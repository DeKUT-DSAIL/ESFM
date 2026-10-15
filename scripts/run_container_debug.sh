#!/bin/bash
# Copyright (c) 2026 DeKUT-DSAIL Research Lab
# Licensed under the MIT License.
# Do not add emojis to the code
#
# Helper script to launch commands or an interactive session in debug mode on CSCS Clariden
# using the Pyxis container configuration defined at scripts/torchcontainer_clariden.toml.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOML_PATH="${REPO_ROOT}/scripts/torchcontainer_clariden.toml"

if [ ! -f "${TOML_PATH}" ]; then
  echo "Error: Container configuration not found at ${TOML_PATH}" >&2
  exit 1
fi

ACCOUNT="${SLURM_ACCOUNT:-g164}"
PARTITION="${SLURM_PARTITION:-debug}"
TIME_LIMIT="${SLURM_TIME:-00:30:00}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
GPUS="${SLURM_GPUS:-1}"

if [ "$#" -gt 0 ]; then
  CMD="$*"
  echo "[HPC Debug] Executing command inside container on (Account: ${ACCOUNT}, Time: ${TIME_LIMIT}):"
  echo "            ${CMD}"
  srun --account="${ACCOUNT}" \
       --partition="${PARTITION}" \
       --time="${TIME_LIMIT}" \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task="${CPUS_PER_TASK}" \
       --gpus="${GPUS}" \
       --environment="${TOML_PATH}" \
       --chdir="${REPO_ROOT}" \
       bash -c "${CMD}"
else
  echo "[HPC Debug] Launching interactive debug shell inside container on (Account: ${ACCOUNT}, Time: ${TIME_LIMIT}) ..."
  srun --account="${ACCOUNT}" \
       --partition="${PARTITION}" \
       --time="${TIME_LIMIT}" \
       --nodes=1 \
       --ntasks=1 \
       --cpus-per-task="${CPUS_PER_TASK}" \
       --gpus="${GPUS}" \
       --environment="${TOML_PATH}" \
       --chdir="${REPO_ROOT}" \
       --pty bash
fi
