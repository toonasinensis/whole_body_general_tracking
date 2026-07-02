#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMixed-G1}"
NUM_ENVS="${NUM_ENVS:-20}"
DEFAULT_RESUME_PATH="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_13000.pt"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  RESUME_PATH="${RESUME_PATH:-${1}}"
  shift
else
  RESUME_PATH="${RESUME_PATH:-${DEFAULT_RESUME_PATH}}"
fi
PAIRS_JSONL="${PAIRS_JSONL:-${WBT_ROOT}/data/mesh_flatwbc_model151000_z10cm/selected_pairs.jsonl}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-data/tracking_npz_data/walk_run1.txt}"
FLAT_WBC_ENV_RATIO="${FLAT_WBC_ENV_RATIO:-0.0}"
FLAT_VELOCITY_ENV_RATIO="${FLAT_VELOCITY_ENV_RATIO:-0.0}"
VELOCITY_TERRAIN_ENV_RATIO="${VELOCITY_TERRAIN_ENV_RATIO:-0.0}"
VELOCITY_TERRAIN_CELL_COUNT="${VELOCITY_TERRAIN_CELL_COUNT:-8}"
VELOCITY_TERRAIN_PROFILE="${VELOCITY_TERRAIN_PROFILE:-velocity_runway_steps}"
MESH_ENV_RATIO="${MESH_ENV_RATIO:-1.00}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-32}"
DEBUG_VELCOMMAND="${DEBUG_VELCOMMAND:-0}"
HEADLESS="${HEADLESS:-0}"
MAX_STEPS="${MAX_STEPS:-100000}"
export WBT_DEBUG_VELCOMMAND="${DEBUG_VELCOMMAND}"

if [[ ! -f "${RESUME_PATH}" ]]; then
  echo "RESUME_PATH does not exist: ${RESUME_PATH}" >&2
  exit 1
fi

ARGS=(
  scripts/rsl_rl/play.py
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --resume_path="${RESUME_PATH}"
  --pairs_jsonl="${PAIRS_JSONL}"
  --flat_dataset_txt="${FLAT_DATASET_TXT}"
  --domain_separator_cell_count="${DOMAIN_SEPARATOR_CELLS}"
  --flat_wbc_env_ratio="${FLAT_WBC_ENV_RATIO}"
  --flat_velocity_env_ratio="${FLAT_VELOCITY_ENV_RATIO}"
  --velocity_terrain_env_ratio="${VELOCITY_TERRAIN_ENV_RATIO}"
  --velocity_terrain_cell_count="${VELOCITY_TERRAIN_CELL_COUNT}"
  --velocity_terrain_profile="${VELOCITY_TERRAIN_PROFILE}"
  --mesh_env_ratio="${MESH_ENV_RATIO}"
  --encoder_mode=robot
#   --export_onnx \
#   --use_onnx_policy \
#   --export_only \
)

if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max_steps="${MAX_STEPS}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
  ARGS+=(--headless)
fi

"${PYTHON_BIN}" "${ARGS[@]}" "$@"
