"""
Competition eval model: Llama-like GPT (~950M params).

Parameter names match torchtitan's Llama3Model exactly so that
checkpoints can be loaded without any key remapping.

CONTRACT
--------
Must expose exactly one function:

    get_model(config: dict) -> torch.nn.Module

The returned model's forward method must have the signature:

    forward(idx: LongTensor[B, T], targets: LongTensor[B, T] | None = None)
        -> (logits: Tensor[B, T, vocab_size], loss: Tensor | None)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    seq_len: int = 2048
    n_layer: int = 20
    n_head: int = 16
    n_kv_head: int = 4
    dim: int = 2048
    ffn_hidden: int = 5632
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def _precompute_rope(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    B, nh, T, hd = x.shape
    xc = torch.view_as_complex(x.float().reshape(B, nh, T, hd // 2, 2))
    fc = freqs_cis[:T].unsqueeze(0).unsqueeze(1)
    return torch.view_as_real(xc * fc).reshape(B, nh, T, hd).type_as(x)


class GQAttention(nn.Module):
    """Grouped-query attention with separate wq/wk/wv/wo matching torchtitan."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.dim // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head

        self.wq = nn.Linear(cfg.dim, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, freqs_cis)
        k = _apply_rope(k, freqs_cis)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(B, T, -1))


class FeedForward(nn.Module):
    """SwiGLU MLP with w1/w2/w3 naming matching torchtitan."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attention = GQAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = FeedForward(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleDict(
            {str(i): TransformerBlock(cfg) for i in range(cfg.n_layer)}
        )
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        head_dim = cfg.dim // cfg.n_head
        self.register_buffer(
            "freqs_cis",
            _precompute_rope(head_dim, cfg.seq_len, cfg.rope_theta),
            persistent=False,
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_embeddings(idx)
        for layer in self.layers.values():
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_model(config: dict) -> nn.Module:
    """
    Instantiate and return the model from a config dict.
    Called by both train.py (before training) and eval.py (to load a checkpoint).
    """
    cfg = ModelConfig(
        vocab_size=config.get("vocab_size", 32768),
        seq_len=config.get("seq_len", 2048),
        n_layer=config.get("n_layer", 20),
        n_head=config.get("n_head", 16),
        n_kv_head=config.get("n_kv_head", 4),
        dim=config.get("dim", 2048),
        ffn_hidden=config.get("ffn_hidden", 5632),
        norm_eps=config.get("norm_eps", 1e-5),
        rope_theta=config.get("rope_theta", 10000.0),
    )
    return LlamaModel(cfg)
