#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RSL_RL_DIR="$(cd "${PROJECT_DIR}/../rsl_rl" && pwd)"
export PYTHONPATH="${RSL_RL_DIR}:${PYTHONPATH:-}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
PYTHON_CMD=(${ISAAC_PYTHON})

TASK="${TASK:-LatentResidual-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-40}"
RESUME_PATH="${RESUME_PATH:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_1000.pt}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/lafan.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
HEADLESS="${HEADLESS:-0}"
MAX_STEPS="${MAX_STEPS:-}"

if [[ -z "${RESUME_PATH}" ]]; then
  echo "RESUME_PATH must point to a trained latent residual PPO checkpoint." >&2
  exit 1
fi

ARGS=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --resume_path="${RESUME_PATH}"
)

if [[ -n "${MOTION_FILE}" ]]; then
  ARGS+=(--motion_file="${MOTION_FILE}")
fi
if [[ -n "${DATASET_TXT}" ]]; then
  ARGS+=(--dataset_txt="${DATASET_TXT}")
fi
if [[ -n "${SMPL_FILE_PATH}" ]]; then
  ARGS+=(--smpl_file_path="${SMPL_FILE_PATH}")
fi
if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi
if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max_steps="${MAX_STEPS}")
fi

"${PYTHON_CMD[@]}" "${PROJECT_DIR}/scripts/rsl_rl/play.py" "${ARGS[@]}" "$@"
