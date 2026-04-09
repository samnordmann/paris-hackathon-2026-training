#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=logs/%j_%N.out
#SBATCH --error=logs/%j_%N.err

set -euo pipefail

mkdir -p logs

# ── Rendezvous info derived from SLURM ──────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500
NNODES=$SLURM_NNODES
NPROC_PER_NODE=$SLURM_GPUS_PER_NODE   # GPUs per node

echo "Master: $MASTER_ADDR:$MASTER_PORT  |  Nodes: $NNODES  |  GPUs/node: $NPROC_PER_NODE"

source .venv/bin/activate

# ── Launch one torchrun per node via srun ────────────────────────────────────
srun python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    --rdzv_id="$SLURM_JOB_ID" \
    train.py \
        --data_dir      /home/data/ \
        --checkpoint_path checkpoint.pt \
        --seq_len       1024 \
        --batch_size    8 \
        --grad_accum_steps 4 \
        --max_steps        5000 \
        --time_limit_min   

