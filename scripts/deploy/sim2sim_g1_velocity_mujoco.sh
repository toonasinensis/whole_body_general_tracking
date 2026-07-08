#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/thl/wt_wbc/wbc_parkour}"
if [[ -z "${SIM2SIM_PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    SIM2SIM_PYTHON="$(command -v python)"
  elif [[ -x "/home/thl/miniconda3/bin/python" ]]; then
    SIM2SIM_PYTHON="/home/thl/miniconda3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    SIM2SIM_PYTHON="$(command -v python3)"
  else
    echo "[ERROR] Could not find python. Set SIM2SIM_PYTHON=/path/to/python." >&2
    exit 127
  fi
fi

ONNX_PATH="${ONNX_PATH:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/MLP/2026-07-08_17-07-44_use_velocity_amp_env_cfg_resume_add_curri/exported/policy.onnx}"
XML_PATH="${XML_PATH:-${ROOT_DIR}/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml}"


# python scripts/deploy/debug_joystick.py --device /dev/input/js0

# Velocity command in robot yaw frame: [vx, vy, yaw_rate].
CMD_VX="${CMD_VX:-0.4}"
CMD_VY="${CMD_VY:-0.0}"
CMD_YAW="${CMD_YAW:-0.0}"
KEYBOARD="${KEYBOARD:-0}"
JOYSTICK="${JOYSTICK:-1}"
JOYSTICK_DEVICE="${JOYSTICK_DEVICE:-/dev/input/js0}"
JOYSTICK_AXIS_VX="${JOYSTICK_AXIS_VX:-1}"
JOYSTICK_AXIS_VY="${JOYSTICK_AXIS_VY:-0}"
JOYSTICK_AXIS_YAW="${JOYSTICK_AXIS_YAW:-3}"
JOYSTICK_DEADZONE="${JOYSTICK_DEADZONE:-0.08}"
JOYSTICK_SCALE_VX="${JOYSTICK_SCALE_VX:-1.2}"
JOYSTICK_SCALE_VY="${JOYSTICK_SCALE_VY:-0.5}"
JOYSTICK_SCALE_YAW="${JOYSTICK_SCALE_YAW:-1.0}"
JOYSTICK_INVERT_VX="${JOYSTICK_INVERT_VX:--1.0}"
JOYSTICK_INVERT_VY="${JOYSTICK_INVERT_VY:--1.0}"
JOYSTICK_INVERT_YAW="${JOYSTICK_INVERT_YAW:--1.0}"
DEBUG_JOYSTICK="${DEBUG_JOYSTICK:-0}"
DEBUG_JOYSTICK_INTERVAL="${DEBUG_JOYSTICK_INTERVAL:-10}"
CMD_VX_LIMIT="${CMD_VX_LIMIT:-1.5}"
CMD_VY_LIMIT="${CMD_VY_LIMIT:-1.0}"
CMD_YAW_LIMIT="${CMD_YAW_LIMIT:-1.5}"

# Simulation controls.
STEPS="${STEPS:-20000000000000}"
DECIMATION="${DECIMATION:-}"
SPAWN_HEIGHT_OFFSET="${SPAWN_HEIGHT_OFFSET:-0.05}"
FALL_HEIGHT="${FALL_HEIGHT:-0.45}"
FAIL_ON_FALL="${FAIL_ON_FALL:-1}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
DRY_RUN="${DRY_RUN:-0}"
RENDER="${RENDER:-1}"

# Fixed mixed-policy task inputs for velocity-only inference.
VEL_TASK_MASK_VALUE="${VEL_TASK_MASK_VALUE:-1.0}"
AUX_MASK_VALUE="${AUX_MASK_VALUE:-0.0}"
WBC_CMD_VALUE="${WBC_CMD_VALUE:-0.0}"
TERRAIN_MODE="${TERRAIN_MODE:-mj_ray}"
TERRAIN_FLAT_VALUE="${TERRAIN_FLAT_VALUE:-0.0}"
HEIGHT_SCAN_BODY="${HEIGHT_SCAN_BODY:-torso_link}"
HEIGHT_SCAN_OFFSET="${HEIGHT_SCAN_OFFSET:-0.0 0.0 20.0}"
HEIGHT_SCAN_GROUND_Z="${HEIGHT_SCAN_GROUND_Z:-0.0}"
HEIGHT_SCAN_OBS_OFFSET="${HEIGHT_SCAN_OBS_OFFSET:-0.5}"
HEIGHT_SCAN_GEOM_GROUPS="${HEIGHT_SCAN_GEOM_GROUPS:-0}"

# Optional PD override. Empty means use ONNX metadata.
KP="${KP:-}"
KD="${KD:-}"

# Viewer controls. Red arrow is command velocity; blue arrow is measured robot velocity.
CAMERA_BODY="${CAMERA_BODY:-pelvis}"
CAMERA_DISTANCE="${CAMERA_DISTANCE:-3.0}"
CAMERA_AZIMUTH="${CAMERA_AZIMUTH:-135.0}"
CAMERA_ELEVATION="${CAMERA_ELEVATION:--18.0}"
SHOW_CMD_ARROW="${SHOW_CMD_ARROW:-1}"

cd "${ROOT_DIR}"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[ERROR] ONNX not found: ${ONNX_PATH}" >&2
  exit 1
fi

ARGS=(
  --onnx_path "${ONNX_PATH}"
  --xml_path "${XML_PATH}"
  --steps "${STEPS}"
  --cmd_vx "${CMD_VX}"
  --cmd_vy "${CMD_VY}"
  --cmd_yaw "${CMD_YAW}"
  --cmd_vx_limit "${CMD_VX_LIMIT}"
  --cmd_vy_limit "${CMD_VY_LIMIT}"
  --cmd_yaw_limit "${CMD_YAW_LIMIT}"
  --joystick_device "${JOYSTICK_DEVICE}"
  --joystick_axis_vx "${JOYSTICK_AXIS_VX}"
  --joystick_axis_vy "${JOYSTICK_AXIS_VY}"
  --joystick_axis_yaw "${JOYSTICK_AXIS_YAW}"
  --joystick_deadzone "${JOYSTICK_DEADZONE}"
  --joystick_scale_vx "${JOYSTICK_SCALE_VX}"
  --joystick_scale_vy "${JOYSTICK_SCALE_VY}"
  --joystick_scale_yaw "${JOYSTICK_SCALE_YAW}"
  --joystick_invert_vx "${JOYSTICK_INVERT_VX}"
  --joystick_invert_vy "${JOYSTICK_INVERT_VY}"
  --joystick_invert_yaw "${JOYSTICK_INVERT_YAW}"
  --debug_joystick_interval "${DEBUG_JOYSTICK_INTERVAL}"
  --spawn_height_offset "${SPAWN_HEIGHT_OFFSET}"
  --fall_height "${FALL_HEIGHT}"
  --log_interval "${LOG_INTERVAL}"
  --vel_task_mask_value "${VEL_TASK_MASK_VALUE}"
  --aux_mask_value "${AUX_MASK_VALUE}"
  --wbc_cmd_value "${WBC_CMD_VALUE}"
  --terrain_mode "${TERRAIN_MODE}"
  --terrain_flat_value "${TERRAIN_FLAT_VALUE}"
  --height_scan_body "${HEIGHT_SCAN_BODY}"
  --height_scan_offset ${HEIGHT_SCAN_OFFSET}
  --height_scan_ground_z "${HEIGHT_SCAN_GROUND_Z}"
  --height_scan_obs_offset "${HEIGHT_SCAN_OBS_OFFSET}"
  --height_scan_geom_groups ${HEIGHT_SCAN_GEOM_GROUPS}
  --camera_body "${CAMERA_BODY}"
  --camera_distance "${CAMERA_DISTANCE}"
  --camera_azimuth "${CAMERA_AZIMUTH}"
  --camera_elevation "${CAMERA_ELEVATION}"
)

if [[ -n "${DECIMATION}" ]]; then
  ARGS+=(--decimation "${DECIMATION}")
fi
if [[ -n "${KP}" ]]; then
  ARGS+=(--kp "${KP}")
fi
if [[ -n "${KD}" ]]; then
  ARGS+=(--kd "${KD}")
fi
if [[ "${FAIL_ON_FALL}" == "1" ]]; then
  ARGS+=(--fail_on_fall)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry_run)
fi
if [[ "${RENDER}" == "1" ]]; then
  ARGS+=(--render)
fi
if [[ "${KEYBOARD}" == "1" ]]; then
  ARGS+=(--keyboard)
fi
if [[ "${JOYSTICK}" == "1" ]]; then
  ARGS+=(--joystick)
fi
if [[ "${DEBUG_JOYSTICK}" == "1" ]]; then
  ARGS+=(--debug_joystick)
fi
if [[ "${SHOW_CMD_ARROW}" == "1" ]]; then
  ARGS+=(--show_cmd_arrow)
fi

echo "[INFO] Velocity-only MuJoCo sim2sim"
echo "[INFO] SIM2SIM_PYTHON=${SIM2SIM_PYTHON}"
echo "[INFO] ONNX_PATH=${ONNX_PATH}"
echo "[INFO] CMD=[${CMD_VX}, ${CMD_VY}, ${CMD_YAW}] STEPS=${STEPS} RENDER=${RENDER} JOYSTICK=${JOYSTICK}"
echo "[INFO] JOYSTICK_DEVICE=${JOYSTICK_DEVICE} axes=[vx:${JOYSTICK_AXIS_VX}, vy:${JOYSTICK_AXIS_VY}, yaw:${JOYSTICK_AXIS_YAW}] scales=[${JOYSTICK_SCALE_VX}, ${JOYSTICK_SCALE_VY}, ${JOYSTICK_SCALE_YAW}] invert=[${JOYSTICK_INVERT_VX}, ${JOYSTICK_INVERT_VY}, ${JOYSTICK_INVERT_YAW}]"

"${SIM2SIM_PYTHON}" whole_body_tracking/scripts/deploy/sim2sim_g1_velocity_mujoco.py "${ARGS[@]}" "$@"
