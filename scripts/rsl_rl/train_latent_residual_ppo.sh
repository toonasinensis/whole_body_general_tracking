#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RSL_RL_DIR="$(cd "${PROJECT_DIR}/../rsl_rl" && pwd)"
export PYTHONPATH="${RSL_RL_DIR}:${PYTHONPATH:-}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
PYTHON_CMD=(${ISAAC_PYTHON})

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29510}"
TASK="${TASK:-LatentResidual-Flat-G1-v0}"
REGISTRY_NAME="${REGISTRY_NAME:-test1}"
RESUME_PRIMITIVE_PATH="${RESUME_PRIMITIVE_PATH:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_latent_distill/2026-06-09_16-47-56_latent_vib_student/model_500.pt}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/lafan.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
HEADLESS="${HEADLESS:-1}"

if [[ -z "${RESUME_PRIMITIVE_PATH}" ]]; then
  echo "RESUME_PRIMITIVE_PATH must point to a distilled LatentVIB checkpoint." >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${MOTION_FILE}" ]]; then
  EXTRA_ARGS+=(--motion_file "${MOTION_FILE}")
fi
if [[ -n "${DATASET_TXT}" ]]; then
  EXTRA_ARGS+=(--dataset_txt "${DATASET_TXT}")
fi
if [[ -n "${SMPL_FILE_PATH}" ]]; then
  EXTRA_ARGS+=(--smpl_file_path "${SMPL_FILE_PATH}")
fi
if [[ -n "${MAX_ITERATIONS}" ]]; then
  EXTRA_ARGS+=(--max_iterations "${MAX_ITERATIONS}")
fi
if [[ "${HEADLESS}" == "1" ]]; then
  EXTRA_ARGS+=(--headless)
fi

"${PYTHON_CMD[@]}" -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/scripts/rsl_rl/train.py" \
  --registry_name="${REGISTRY_NAME}" \
  --task="${TASK}" \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --resume_primitive_path="${RESUME_PRIMITIVE_PATH}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
