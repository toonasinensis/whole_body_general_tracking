#!/usr/bin/env bash
set -e

ROOT_DIR="${ROOT_DIR:-/home/thl/wt_wbc/wbc_parkour}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
SIM2SIM_PYTHON="${SIM2SIM_PYTHON:-python}"
TASK="${TASK:-AMP-G1}"
NUM_ENVS="${NUM_ENVS:-1}"
RESUME_PATH="${RESUME_PATH:-${ROOT_DIR}/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/exported/policy.onnx}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/lafan.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
ENCODER_MODE="${ENCODER_MODE:-robot}"
ONNX_DIR="${ONNX_DIR:-$(dirname "${RESUME_PATH}")/exported}"
ONNX_FILENAME="${ONNX_FILENAME:-policy.onnx}"
ONNX_PATH="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/exported/policy.onnx"
EXPORT_ONNX="${EXPORT_ONNX:-0}"
RENDER="${RENDER:-1}"
SHOW_REFERENCE="${SHOW_REFERENCE:-1}"
METRICS_CSV="${METRICS_CSV:-}"
METRICS_TAG="${METRICS_TAG:-}"
# Optional single-motion/debug controls.
# Example:
#   MOTION_INDEX=3 MOTION_START_FRAME=120 LOG_INTERVAL=10 ./scripts/deploy/sim2sim_g1_mujoco.sh
#   MOTION_PATH=walk2_subject1.named.npz MOTION_START_FRAME=80 ./scripts/deploy/sim2sim_g1_mujoco.sh
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
  ${ISAAC_PYTHON} whole_body_tracking/scripts/rsl_rl/play.py \
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

${SIM2SIM_PYTHON} whole_body_tracking/scripts/deploy/sim2sim_g1_mujoco.py \
  --onnx_path "${ONNX_PATH}" \
  --motion_file "${MOTION_FILE}" \
  "${DATASET_ARGS[@]}" \
  "${MOTION_SELECT_ARGS[@]}" \
  "${RENDER_ARGS[@]}" \
  "${IMU_DEBUG_ARGS[@]}" \
  "${REFERENCE_ARGS[@]}" \
  "${METRICS_ARGS[@]}" \
  "$@"
