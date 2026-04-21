#!/bin/bash
# Copyright (c) 2026 ETH Zurich
# Authors: see CONTRIBUTORS.md
# Licensed under the MIT License. See the LICENSE file in the repository root.

#SBATCH --job-name=train_ESFM_s_wm_ri          # short name for the job
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=4           # GPUs per node
#SBATCH -c 72                       # CPU cores per task
#SBATCH --mem=460000                # memory per node
#SBATCH --exclusive
#SBATCH --time=1:00:00             # total run time (HH:MM:SS)
#SBATCH --account=a122
#SBATCH --output=logs/train_ESFM_s_wm_ri%j.out  # output log file


# ------------------------------------------------------------------------------
# Environment setup
# (Adjust or remove lines to match your environment)
# ------------------------------------------------------------------------------
source /users/$USER/.wandb_env
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

# Set the working directory (users can modify this environment variable)
workdir="/users/$USER/projects/ESFM"
export WORKDIR=${WORKDIR:-$workdir}
tomlpath="$workdir/scripts/torchcontainer_clariden.toml"
export savedir="/iopsstor/scratch/cscs/$USER/ESFM_outputs"
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
  bash -c '
    cd $WORKDIR
    
    echo "[Node ${SLURM_PROCID}] Starting torchrun ..."
    echo "[Node ${SLURM_PROCID}] Environment:"
    echo "  MASTER_ADDR=$MASTER_ADDR"
    echo "  MASTER_PORT=$MASTER_PORT"
    echo "  SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
    echo "  SLURM_PROCID=$SLURM_PROCID"
    echo "  PWD=$(pwd)"
    echo "  PYTHONPATH=$PYTHONPATH"

    set -x  # Enable command printing

    # Patch throughput.py to map GH200 to H100 SXM
    sed -i '\''/chip = device_name.lower()/a\        # map GH200 Superchip to H100 SXM so we pick up HBM3 flops\n        if "gh200" in chip:\n            chip = "h100 sxm"'\'' /usr/local/lib/python3.10/dist-packages/lightning/fabric/utilities/throughput.py


    python3 -c "import torch; print(f\"Torch version: {torch.__version__}\")"
    python3 -c "import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")"
    torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$SLURM_PROCID \
      --nproc_per_node=4 \
      --rdzv_id=42 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      train.py \
        --config ./configs/config_ESFM_s_wm_ri.yaml \
        --num_nodes $SLURM_JOB_NUM_NODES \
        2>&1
'

echo "Run finished at: $(date)"