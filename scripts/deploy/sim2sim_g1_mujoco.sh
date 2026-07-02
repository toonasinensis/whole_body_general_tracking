#!/usr/bin/env bash
set -e

ROOT_DIR="${ROOT_DIR:-/home/thl/wt_wbc/wbc_parkour}"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
SIM2SIM_PYTHON="${SIM2SIM_PYTHON:-python}"
TASK="${TASK:-AMP-G1}"
NUM_ENVS="${NUM_ENVS:-1}"
RESUME_PATH="${RESUME_PATH:-${ROOT_DIR}/whole_body_tracking/logs/rsl_rl/exported/policy.onnx}"
MOTION_FILE="${MOTION_FILE:-whole_body_tracking/data/mesh_flatwbc_model151000_z10cm/selected_pairs.jsonl}"
DATASET_TXT="${DATASET_TXT:-}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
ENCODER_MODE="${ENCODER_MODE:-robot}"
ONNX_DIR="${ONNX_DIR:-$(dirname "${RESUME_PATH}")/exported}"
ONNX_FILENAME="${ONNX_FILENAME:-policy.onnx}"
ONNX_PATH="${ONNX_PATH:-${ROOT_DIR}/whole_body_tracking/logs/rsl_rl/exported/policy.onnx}"
EXPORT_ONNX="${EXPORT_ONNX:-0}"
RENDER="${RENDER:-1}"
SHOW_REFERENCE="${SHOW_REFERENCE:-1}"
SHOW_VELOCITY_COMMAND="${SHOW_VELOCITY_COMMAND:-1}"
SHOW_HEIGHT_SCAN="${SHOW_HEIGHT_SCAN:-1}"
HEIGHT_SCAN_VIS_POINTS_ONLY="${HEIGHT_SCAN_VIS_POINTS_ONLY:-1}"
SPAWN_HEIGHT_OFFSET="${SPAWN_HEIGHT_OFFSET:-0.1}"
CAMERA_FOLLOW="${CAMERA_FOLLOW:-1}"
CAMERA_BODY="${CAMERA_BODY:-pelvis}"
CAMERA_DISTANCE="${CAMERA_DISTANCE:-3.0}"
CAMERA_AZIMUTH="${CAMERA_AZIMUTH:-135.0}"
CAMERA_ELEVATION="${CAMERA_ELEVATION:--18.0}"
VELOCITY_COMMAND_SOURCE="${VELOCITY_COMMAND_SOURCE:-keyboard}"
METRICS_CSV="${METRICS_CSV:-}"
METRICS_TAG="${METRICS_TAG:-}"
# Optional single-motion/debug controls.
# Example:
#   MOTION_INDEX=3 MOTION_START_FRAME=120 LOG_INTERVAL=10 ./scripts/deploy/sim2sim_g1_mujoco.sh
#   MOTION_PATH=walk2_subject1.named.npz MOTION_START_FRAME=80 ./scripts/deploy/sim2sim_g1_mujoco.sh
MOTION_INDEX="${MOTION_INDEX:-0}"
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

VELOCITY_COMMAND_VIS_ARGS=()
if [[ "${SHOW_VELOCITY_COMMAND}" == "1" ]]; then
  VELOCITY_COMMAND_VIS_ARGS=(--show_velocity_command)
fi

HEIGHT_SCAN_ARGS=()
if [[ "${SHOW_HEIGHT_SCAN}" == "1" ]]; then
  HEIGHT_SCAN_ARGS=(--show_height_scan)
fi
if [[ "${HEIGHT_SCAN_VIS_POINTS_ONLY}" == "1" ]]; then
  HEIGHT_SCAN_ARGS+=(--height_scan_vis_points_only)
fi

CAMERA_ARGS=()
if [[ "${CAMERA_FOLLOW}" == "0" ]]; then
  CAMERA_ARGS=(--no_camera_follow)
else
  CAMERA_ARGS=(
    --camera_body "${CAMERA_BODY}"
    --camera_distance "${CAMERA_DISTANCE}"
    --camera_azimuth "${CAMERA_AZIMUTH}"
    --camera_elevation "${CAMERA_ELEVATION}"
  )
fi

METRICS_ARGS=()
METRICS_ARGS=(--metrics_csv "${METRICS_CSV}")
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
  "${REFERENCE_ARGS[@]}" \
  "${VELOCITY_COMMAND_VIS_ARGS[@]}" \
  "${HEIGHT_SCAN_ARGS[@]}" \
  "${CAMERA_ARGS[@]}" \
  --velocity_command_source "${VELOCITY_COMMAND_SOURCE}" \
  --spawn_height_offset "${SPAWN_HEIGHT_OFFSET}" \
  "${METRICS_ARGS[@]}" \
  "$@"
