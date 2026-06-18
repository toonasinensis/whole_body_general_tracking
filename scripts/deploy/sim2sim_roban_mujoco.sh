#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-mimic}"
ISAACLAB_SH="${ISAACLAB_SH:-/home/leju/IsaacLab/isaaclab.sh}"
TASK="${TASK:-AMP-RobanS22}"
CHECKPOINT="${CHECKPOINT:-${ROOT_DIR}/logs/rsl_rl/model_16000.pt}"
MOTION_FILE="${MOTION_FILE:-/home/leju/Documents/zjl/test-motion}"
DATASET_TXT="${DATASET_TXT:-}"
ONNX_PATH="${ONNX_PATH:-${ROOT_DIR}/logs/rsl_rl/exported/roban_policy.onnx}"
EXPORT_ONNX="${EXPORT_ONNX:-0}"
RENDER="${RENDER:-1}"

cd "${ROOT_DIR}"

DATASET_ARGS=()
if [[ -n "${DATASET_TXT}" ]]; then
  DATASET_ARGS=(--dataset_txt "${DATASET_TXT}")
fi

if [[ "${EXPORT_ONNX}" == "1" ]]; then
  mkdir -p "$(dirname "${ONNX_PATH}")"
  conda run -n "${CONDA_ENV}" "${ISAACLAB_SH}" -p scripts/rsl_rl/play.py \
    --task="${TASK}" \
    --num_envs=1 \
    --motion_file="${MOTION_FILE}" \
    "${DATASET_ARGS[@]}" \
    --resume_path="${CHECKPOINT}" \
    --encoder_mode=robot \
    --export_onnx \
    --export_only \
    --onnx_dir="$(dirname "${ONNX_PATH}")" \
    --onnx_filename="$(basename "${ONNX_PATH}")" \
    --headless
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[ERROR] ONNX not found: ${ONNX_PATH}" >&2
  echo "        Run with EXPORT_ONNX=1 first." >&2
  exit 1
fi

RENDER_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  RENDER_ARGS=(--render)
fi

conda run -n "${CONDA_ENV}" python scripts/deploy/sim2sim_roban_mujoco.py \
  --onnx_path="${ONNX_PATH}" \
  --motion_file="${MOTION_FILE}" \
  "${DATASET_ARGS[@]}" \
  "${RENDER_ARGS[@]}" \
  "$@"
