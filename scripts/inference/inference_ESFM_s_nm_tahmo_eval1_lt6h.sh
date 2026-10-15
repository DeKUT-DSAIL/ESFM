#!/bin/bash

# Copyright (c) 2026 ETH Zurich and DeKUT-DSAIL
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

#SBATCH --job-name=zero_shot_tahmo_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 72
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --time=02:30:00
#SBATCH --account=g164
#SBATCH --output=logs/zero_shot_tahmo_full_%j.out
#SBATCH --error=logs/zero_shot_tahmo_full_%j.err

# ------------------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------------------
[ -f /users/$USER/.wandb_env ] && source /users/$USER/.wandb_env
export OMP_NUM_THREADS=4
ulimit -c 0  # disable core dumps
ulimit -t unlimited
export NCCL_CROSS_NIC=1
export NCCL_DEBUG="INFO"

# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501

node_rank=$SLURM_NODEID
nnodes=$SLURM_JOB_NUM_NODES

echo "================= SLURM Info ================="
echo "Run started at:    $(date)"
echo "SLURM_JOB_ID:      $SLURM_JOB_ID"
echo "SLURM_NODELIST:    $SLURM_JOB_NODELIST"
echo "MASTER_ADDR:       $MASTER_ADDR"
echo "MASTER_PORT:       $MASTER_PORT"
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "node_rank:         $node_rank"
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "================================================"
set -x

# Set the working directory
workdir="/users/$USER/ESFM"
export WORKDIR=${WORKDIR:-$workdir}
tomlpath="$workdir/scripts/torchcontainer_clariden.toml"
export savedir="/ritom/scratch/cscs/$USER/ai-4-weather/experiments/ESFMs/ESFM_outputs"
echo "workdir: $workdir"
echo "Run started at:" $(date)

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="6000"
export NCCL_BLOCKING_WAIT="1"
export NCCL_TIMEOUT="6000"

export FI_CXI_RDZV_GET_MIN=0 
export FI_CXI_RDZV_THRESHOLD=0 
export FI_CXI_RDZV_EAGER_SIZE=0

srun --ntasks=$nnodes \
     --ntasks-per-node=1 \
     --cpus-per-task=72 \
     --export=ALL \
     --environment=$tomlpath \
     -u -l \
  bash -c "
    echo \"[Node \${SLURM_PROCID}] Starting torchrun ...\"
    echo \"[Node \${SLURM_PROCID}] Environment:\"
    echo \"  MASTER_ADDR=\$MASTER_ADDR\"
    echo \"  MASTER_PORT=\$MASTER_PORT\"
    echo \"  SLURM_JOB_NUM_NODES=\$SLURM_JOB_NUM_NODES\"
    echo \"  SLURM_PROCID=\$SLURM_PROCID\"
    echo \"  PWD=\$(pwd)\"
    echo \"  PYTHONPATH=\$PYTHONPATH\"

    set -x

    cd \$WORKDIR
    python3 -c \"import torch; print(f'Torch version: {torch.__version__}')\"
    python3 -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"
    torchrun \\
      --nnodes=\$SLURM_JOB_NUM_NODES \\
      --node_rank=\$SLURM_PROCID \\
      --nproc_per_node=4 \\
      --rdzv_id=42 \\
      --rdzv_backend=c10d \\
      --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
      inference_rollout.py \\
        --config ./configs/config_ESFM_s_nm_e11k_lt6h.yaml \\
        --NUM_ROLLOUT_STEPS 40 \\
        --test_time_intervals tahmo_full \\
        --grid_deg_delta 0.00001 \\
        2>&1
  "
echo "Run finished at: $(date)"
