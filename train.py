"""
Optimized training script for the GPU MODE hackathon.

Tier 1: Gradient accumulation, scaled LR, WSD schedule
Tier 2: FSDP2 (shards params+optim across GPUs), torch.compile max-autotune
Tier 3: FP8 via torchao for ~1.3x throughput on B300

Only requires: torch (nightly), numpy, torchao
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import time
import glob
import math
import argparse
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group

from model import get_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_dir:    str   = "/home/data/"
    token_dtype: str   = "uint16"
    seq_len:     int   = 2048

    vocab_size: int   = 32768
    n_layer:    int   = 20
    n_head:     int   = 16
    n_kv_head:  int   = 4
    dim:        int   = 2048
    ffn_hidden: int   = 5632
    norm_eps:   float = 1e-5
    rope_theta: float = 10000.0

    batch_size:       int   = 64
    grad_accum_steps: int   = 1
    max_lr:           float = 6e-4
    min_lr:           float = 6e-5
    warmup_steps:     int   = 200
    max_steps:        int   = 50_000
    weight_decay:     float = 0.1
    grad_clip:        float = 1.0
    time_limit_seconds: float = 10 * 60

    checkpoint_path: str = "checkpoint.pt"
    compile:      bool = True
    compile_mode: str  = "default"
    use_fsdp:     bool = True
    use_fp8:      bool = False
    use_ac:       bool = False
    chunked_ce:   bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinDataset:
    def __init__(self, data_dir: str, seq_len: int, dtype: str = "uint16"):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(f"No *.bin files found in '{data_dir}'")
        self.seq_len = seq_len
        np_dtype = np.dtype(dtype)
        self.shards = [np.memmap(p, dtype=np_dtype, mode="r") for p in paths]
        self.lengths = [len(s) for s in self.shards]
        self.total = sum(self.lengths)
        self.weights = np.array([l / self.total for l in self.lengths])
        self.num_shards = len(self.shards)
        print(f"[data] {len(paths)} shard(s), {self.total:,} tokens total")

    def get_batch(self, batch_size: int, device):
        sl = self.seq_len
        shard_idxs = np.random.choice(self.num_shards, size=batch_size, p=self.weights)
        xs, ys = [], []
        for si in shard_idxs:
            shard = self.shards[si]
            start = np.random.randint(0, len(shard) - sl - 1)
            chunk = torch.from_numpy(shard[start : start + sl + 1].astype(np.int64))
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        return torch.stack(xs).to(device, non_blocking=True), \
               torch.stack(ys).to(device, non_blocking=True)


# ---------------------------------------------------------------------------
# LR schedule: warmup -> stable -> cosine decay (WSD-like)
# ---------------------------------------------------------------------------

def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.max_lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.max_lr - cfg.min_lr)


# ---------------------------------------------------------------------------
# Checkpoint (FSDP2-aware)
# ---------------------------------------------------------------------------

def save_checkpoint(model, step: int, cfg: Config, use_fsdp: bool):
    rank = dist.get_rank() if dist.is_initialized() else 0

    if use_fsdp:
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict, StateDictOptions,
        )
        sd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
    else:
        raw = model.module if hasattr(model, "module") else model
        sd = {k: v.cpu() for k, v in raw.state_dict().items()}

    if rank == 0:
        clean_sd = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor):
                clean_sd[k] = v.contiguous()
        torch.save({
            "step":   step,
            "model":  clean_sd,
            "config": asdict(cfg),
        }, cfg.checkpoint_path)
        print(f"[ckpt] saved -> {cfg.checkpoint_path}  (step {step})")

    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Accept all args the starter submit.sh passes, but override with best config
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",          default="/home/data/")
    parser.add_argument("--checkpoint_path",   default="checkpoint.pt")
    parser.add_argument("--time_limit_min",    type=float, default=10.0)
    # Accept (and ignore) starter submit.sh args so it doesn't error
    parser.add_argument("--seq_len",           type=int,   default=0)
    parser.add_argument("--batch_size",        type=int,   default=0)
    parser.add_argument("--grad_accum_steps",  type=int,   default=0)
    parser.add_argument("--max_steps",         type=int,   default=0)
    parser.add_argument("--vocab_size",        type=int,   default=0)
    parser.add_argument("--n_layer",           type=int,   default=0)
    parser.add_argument("--n_head",            type=int,   default=0)
    parser.add_argument("--n_embd",            type=int,   default=0)
    args, _ = parser.parse_known_args()

    # ── Hardcoded best config (ignores submit.sh overrides) ──
    cfg = Config(
        data_dir           = args.data_dir,
        checkpoint_path    = args.checkpoint_path,
        time_limit_seconds = args.time_limit_min * 60,
    )

    # ------------------------------------------------------------------ Dist
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
        world_size = dist.get_world_size()
    else:
        rank = 0; master = True; world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.use_fsdp = False

    torch.manual_seed(1337 + rank)
    if "cuda" in device:
        torch.cuda.manual_seed(1337 + rank)

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) \
              if "cuda" in device else nullcontext()

    # ------------------------------------------------------------------ Model
    model = get_model(asdict(cfg)).to(device)
    if master:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] {n_params/1e6:.1f}M parameters  |  "
              f"dim={cfg.dim} layers={cfg.n_layer} heads={cfg.n_head}/{cfg.n_kv_head}")

    # ---- Activation checkpointing + chunked CE ----
    if cfg.use_ac:
        model.use_ac = True
        if master:
            print("[ac] activation checkpointing enabled")
    if cfg.chunked_ce:
        model.use_chunked_ce = True
        if master:
            print("[chunked_ce] chunked cross-entropy enabled")

    # ---- Tier 3: FP8 via torchao ----
    if cfg.use_fp8 and "cuda" in device:
        if master:
            print("[fp8] converting linear layers to float8 training...")
        try:
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            fp8_config = Float8LinearConfig()
            convert_to_float8_training(model, config=fp8_config)
            if master:
                print("[fp8] done")
        except Exception as e:
            if master:
                print(f"[fp8] FAILED: {e} — falling back to BF16")
            cfg.use_fp8 = False

    # ---- Tier 2b: torch.compile (regional, per-block) ----
    if cfg.compile and "cuda" in device:
        if master:
            print(f"[compile] mode={cfg.compile_mode}, applying to each block...")
        for key in list(model.layers.keys()):
            model.layers[key] = torch.compile(
                model.layers[key], mode=cfg.compile_mode
            )
        if master:
            print("[compile] done (kernels JIT on first step)")

    # ---- Tier 2a: FSDP2 (replaces DDP) ----
    if cfg.use_fsdp and ddp:
        if master:
            print("[fsdp2] applying fully_shard to model...")
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        for key in model.layers:
            fully_shard(model.layers[key], mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
        if master:
            print("[fsdp2] done")
    elif ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
        if master:
            print("[ddp] wrapped model in DDP")

    # ------------------------------------------------------------------ Optimizer
    raw_model = model
    decay_params   = [p for _, p in raw_model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for _, p in raw_model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": cfg.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=cfg.max_lr, betas=(0.9, 0.95), fused="cuda" in device,
    )

    # ------------------------------------------------------------------ Data
    dataset = BinDataset(cfg.data_dir, cfg.seq_len, cfg.token_dtype)

    tokens_per_step = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps * world_size
    if master:
        mode_str = "FSDP2" if cfg.use_fsdp else "DDP"
        fp8_str = "+FP8" if cfg.use_fp8 else ""
        compile_str = f"+compile({cfg.compile_mode})" if cfg.compile else ""
        print(f"[train] {mode_str}{fp8_str}{compile_str}  |  "
              f"{tokens_per_step:,} tok/step  "
              f"(bs={cfg.batch_size} x accum={cfg.grad_accum_steps} "
              f"x world={world_size} x seq={cfg.seq_len})")

    # ------------------------------------------------------------------ Train
    step = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()

    while step < cfg.max_steps:
        elapsed = time.time() - train_start
        stop = torch.tensor(int(elapsed >= cfg.time_limit_seconds), device=device)
        if ddp:
            dist.broadcast(stop, src=0)
        if stop.item():
            if master:
                print(f"\n[time] {elapsed/60:.1f} min elapsed — time limit reached.")
            save_checkpoint(model, step, cfg, cfg.use_fsdp)
            break

        step_start = time.time()
        lr = get_lr(step, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        accumulated_loss = 0.0
        for micro_step in range(cfg.grad_accum_steps):
            x, y = dataset.get_batch(cfg.batch_size, device)

            is_last_micro = (micro_step == cfg.grad_accum_steps - 1)

            if cfg.use_fsdp and cfg.grad_accum_steps > 1:
                model.set_requires_gradient_sync(is_last_micro)
                sync_ctx = nullcontext()
            elif ddp and not is_last_micro:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = nullcontext()

            with sync_ctx, amp_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

        if cfg.grad_clip > 0:
            if cfg.use_fsdp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1

        if master and step % 10 == 0:
            elapsed_total = time.time() - train_start
            remaining = max(0, cfg.time_limit_seconds - elapsed_total)
            step_ms = (time.time() - step_start) * 1000
            tps = tokens_per_step / (step_ms / 1000) if step_ms > 0 else 0
            print(f"step {step:6d} | loss {accumulated_loss:.4f} | "
                  f"lr {lr:.2e} | "
                  f"{step_ms:.0f}ms/step | "
                  f"{tps/1e6:.2f}M tok/s | "
                  f"elapsed {elapsed_total/60:.1f}m | "
                  f"left {remaining/60:.1f}m")

    if step >= cfg.max_steps and master:
        print(f"\n[done] Reached max_steps={cfg.max_steps}.")
        save_checkpoint(model, step, cfg, cfg.use_fsdp)

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
