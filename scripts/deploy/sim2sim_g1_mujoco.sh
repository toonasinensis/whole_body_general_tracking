#!/usr/bin/env bash
set -e

ROOT_DIR="${ROOT_DIR:-/home/thl/wt_wbc/wbc_parkour}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
TASK="${TASK:-FM-G1}"
NUM_ENVS="${NUM_ENVS:-1}"
RESUME_PATH="${RESUME_PATH:-${ROOT_DIR}/whole_body_tracking/logs/rsl_rl/g1_flat/model_95400.pt}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-${ROOT_DIR}/whole_body_tracking/dataset_txt/mini_test.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-/home/thl/Downloads/data/output/smpl_filtered}"
ENCODER_MODE="${ENCODER_MODE:-robot}"
ONNX_DIR="${ONNX_DIR:-$(dirname "${RESUME_PATH}")/exported}"
ONNX_FILENAME="${ONNX_FILENAME:-policy.onnx}"
ONNX_PATH="${ONNX_PATH:-${ONNX_DIR}/${ONNX_FILENAME}}"

cd "${ROOT_DIR}"

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
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
fi

python whole_body_tracking/scripts/deploy/sim2sim_g1_mujoco.py \
  --onnx_path "${ONNX_PATH}" \
  --motion_file "${MOTION_FILE}" \
  --dataset_txt "${DATASET_TXT}" \
  "$@"
