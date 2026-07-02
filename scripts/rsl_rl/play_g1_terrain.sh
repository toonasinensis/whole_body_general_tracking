#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMixed-G1}"
NUM_ENVS="${NUM_ENVS:-40}"
DEFAULT_RESUME_PATH="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_paired_terrain_height_scan/2026-06-23_19-00-16_with_run_walk/model_128500.pt"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  RESUME_PATH="${RESUME_PATH:-${1}}"
  shift
else
  RESUME_PATH="${RESUME_PATH:-${DEFAULT_RESUME_PATH}}"
fi
PAIRS_JSONL="${PAIRS_JSONL:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/omniretarget/g1_terrain/cropped_pairs/pairs_cropped.jsonl}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-data/tracking_npz_data/walk_run.txt}"
FLAT_ENV_RATIO="${FLAT_ENV_RATIO:-0.25}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-32}"
HEADLESS="${HEADLESS:-0}"
MAX_STEPS="${MAX_STEPS:-100000}"

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
  --flat_env_ratio="${FLAT_ENV_RATIO}"
  --domain_separator_cell_count="${DOMAIN_SEPARATOR_CELLS}"
)

if [[ -n "${MAX_STEPS}" ]]; then
  ARGS+=(--max_steps="${MAX_STEPS}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
  ARGS+=(--headless)
fi

"${PYTHON_BIN}" "${ARGS[@]}" "$@"
