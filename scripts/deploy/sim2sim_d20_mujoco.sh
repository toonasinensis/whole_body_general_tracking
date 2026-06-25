#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/kiki/IsaacLab/isaaclab.sh -p}"
SIM2SIM_PYTHON="${SIM2SIM_PYTHON:-python}"
TASK="${TASK:-Tracking-Flat-D20-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
RESUME_PATH="${RESUME_PATH:-${ROOT_DIR}/logs/rsl_rl/d20_flat/2026-06-02_15-13-36/model_103500.pt}"
MOTION_FILE="${MOTION_FILE:-/home/kiki/workspace/motion_data/mimic_data/d20v2/bones_seed}"
DATASET_TXT="${DATASET_TXT:-dataset_txt/bones_seed/test.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
ENCODER_MODE="${ENCODER_MODE:-robot}"
ONNX_DIR="${ONNX_DIR:-$(dirname "${RESUME_PATH}")/exported}"
ONNX_FILENAME="${ONNX_FILENAME:-policy.onnx}"
ONNX_PATH="${ONNX_PATH:-${ONNX_DIR}/${ONNX_FILENAME}}"
EXPORT_ONNX="${EXPORT_ONNX:-0}"
RENDER="${RENDER:-1}"
SHOW_REFERENCE="${SHOW_REFERENCE:-1}"
METRICS_CSV="${METRICS_CSV:-}"
METRICS_TAG="${METRICS_TAG:-}"
MOTION_INDEX="${MOTION_INDEX:-}"
MOTION_PATH="${MOTION_PATH:-}"
MOTION_START_FRAME="${MOTION_START_FRAME:-}"
LOG_INTERVAL="${LOG_INTERVAL:-}"

cd "${ROOT_DIR}"

DATASET_ARGS=()
if [[ -n "${DATASET_TXT}" ]]; then
  DATASET_ARGS=(--dataset_txt "${DATASET_TXT}")
else
  echo "[INFO] DATASET_TXT is empty. Using all .npz files under MOTION_FILE: ${MOTION_FILE}"
fi

if [[ "${EXPORT_ONNX}" == "1" ]]; then
  echo "[INFO] EXPORT_ONNX=1, launching IsaacLab to export ONNX: ${ONNX_PATH}"
  ${ISAAC_PYTHON} scripts/rsl_rl/play.py \
    --task="${TASK}" \
    --num_envs="${NUM_ENVS}" \
    --resume_path="${RESUME_PATH}" \
    --motion_file="${MOTION_FILE}" \
    "${DATASET_ARGS[@]}" \
    --smpl_file_path="${SMPL_FILE_PATH}" \
    --encoder_mode="${ENCODER_MODE}" \
    --export_onnx \
    --export_only \
    --onnx_dir="${ONNX_DIR}" \
    --onnx_filename="${ONNX_FILENAME}"
else
  echo "[INFO] Skipping IsaacLab export. Using existing ONNX: ${ONNX_PATH}"
  if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "[ERROR] ONNX not found: ${ONNX_PATH}" >&2
    echo "        Run with EXPORT_ONNX=1 to export it first." >&2
    exit 1
  fi
fi

RENDER_ARGS=()
if [[ "${RENDER}" == "1" ]]; then
  RENDER_ARGS=(--render)
fi

REFERENCE_ARGS=()
if [[ "${SHOW_REFERENCE}" == "0" ]]; then
  REFERENCE_ARGS=(--no_show_reference)
fi

METRICS_ARGS=()
if [[ -n "${METRICS_CSV}" ]]; then
  METRICS_ARGS=(--metrics_csv "${METRICS_CSV}")
fi
if [[ -n "${METRICS_TAG}" ]]; then
  METRICS_ARGS+=(--metrics_tag "${METRICS_TAG}")
fi

MOTION_SELECT_ARGS=()
if [[ -n "${MOTION_INDEX}" ]]; then
  MOTION_SELECT_ARGS+=(--motion_index "${MOTION_INDEX}")
fi
if [[ -n "${MOTION_PATH}" ]]; then
  MOTION_SELECT_ARGS+=(--motion_path "${MOTION_PATH}")
fi
if [[ -n "${MOTION_START_FRAME}" ]]; then
  MOTION_SELECT_ARGS+=(--motion_start_frame "${MOTION_START_FRAME}")
fi
if [[ -n "${LOG_INTERVAL}" ]]; then
  MOTION_SELECT_ARGS+=(--log_interval "${LOG_INTERVAL}")
fi

${SIM2SIM_PYTHON} scripts/deploy/sim2sim_d20_mujoco.py \
  --onnx_path "${ONNX_PATH}" \
  --motion_file "${MOTION_FILE}" \
  "${DATASET_ARGS[@]}" \
  "${MOTION_SELECT_ARGS[@]}" \
  "${RENDER_ARGS[@]}" \
  "${REFERENCE_ARGS[@]}" \
  "${METRICS_ARGS[@]}" \
  "$@"
