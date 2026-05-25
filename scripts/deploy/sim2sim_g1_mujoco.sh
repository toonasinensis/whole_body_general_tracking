#!/usr/bin/env bash
set -e

ROOT_DIR="${ROOT_DIR:-/home/thl/wt_wbc/wbc_parkour}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
SIM2SIM_PYTHON="${SIM2SIM_PYTHON:-python}"
TASK="${TASK:-FM-G1}"
NUM_ENVS="${NUM_ENVS:-1}"
RESUME_PATH="${RESUME_PATH:-${ROOT_DIR}/whole_body_tracking/logs/rsl_rl/g1_flat/2026-05-22_23-15-00/model_39000.pt}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/fall.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-/home/thl/Downloads/data/output/smpl_filtered}"
ENCODER_MODE="${ENCODER_MODE:-robot}"
ONNX_DIR="${ONNX_DIR:-$(dirname "${RESUME_PATH}")/exported}"
ONNX_FILENAME="${ONNX_FILENAME:-policy.onnx}"
ONNX_PATH="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/exported/policy.onnx"
EXPORT_ONNX="${EXPORT_ONNX:-0}"
RENDER="${RENDER:-1}"
DEBUG_IMU="${DEBUG_IMU:-0}"
SHOW_REFERENCE="${SHOW_REFERENCE:-1}"

cd "${ROOT_DIR}"

if [[ "${EXPORT_ONNX}" == "1" ]]; then
  echo "[INFO] EXPORT_ONNX=1, launching IsaacLab to export ONNX: ${ONNX_PATH}"
  ${ISAAC_PYTHON} whole_body_tracking/scripts/rsl_rl/play.py \
    --task="${TASK}" \
    --num_envs="${NUM_ENVS}" \
    --resume_path="${RESUME_PATH}" \
    --motion_file="${MOTION_FILE}" \
    --dataset_txt="${DATASET_TXT}" \
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

IMU_DEBUG_ARGS=()
if [[ "${DEBUG_IMU}" == "1" ]]; then
  IMU_DEBUG_ARGS=(--debug_imu)
fi

REFERENCE_ARGS=()
if [[ "${SHOW_REFERENCE}" == "0" ]]; then
  REFERENCE_ARGS=(--no_show_reference)
fi

${SIM2SIM_PYTHON} whole_body_tracking/scripts/deploy/sim2sim_g1_mujoco.py \
  --onnx_path "${ONNX_PATH}" \
  --motion_file "${MOTION_FILE}" \
  --dataset_txt "${DATASET_TXT}" \
  "${RENDER_ARGS[@]}" \
  "${IMU_DEBUG_ARGS[@]}" \
  "${REFERENCE_ARGS[@]}" \
  "$@"
