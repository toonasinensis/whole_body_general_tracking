#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMixed-G1}"
if [[ $# -gt 0 && "${1}" != --* ]]; then
  RESUME_PATH="${RESUME_PATH:-${1}}"
  shift
else
  RESUME_PATH="${RESUME_PATH:-${WBT_ROOT}/logs/rsl_rl/model_151000.pt}"
fi
PAIRS_JSONL="${PAIRS_JSONL:-${WBT_ROOT}/data/omniretarget/g1_terrain/cropped_pairs/pairs_cropped.jsonl}"
COLLECT_DOMAINS="${COLLECT_DOMAINS:-mesh,flat_wbc}"
MESH_PAIR_INDICES="${MESH_PAIR_INDICES:-all}"
FLAT_MOTION_FILE="${FLAT_MOTION_FILE:-data/g1_amp/WalkandRun_isaaclab}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-dataset_txt/WalkandRun.txt}"
FLAT_MOTION_INDICES="${FLAT_MOTION_INDICES:-all}"
RUN_NAME="${RUN_NAME:-play_collect_${COLLECT_DOMAINS//,/_}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WBT_ROOT}/logs/play_collect_mesh_rollout/${RUN_NAME}}"
MAX_STEPS="${MAX_STEPS:-0}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-0}"
MOTION_RESET_Z_OFFSET="${MOTION_RESET_Z_OFFSET:-0.10}"
HEADLESS="${HEADLESS:-1}"

if [[ ! -f "${RESUME_PATH}" ]]; then
  echo "RESUME_PATH does not exist: ${RESUME_PATH}" >&2
  exit 1
fi

ARGS=(
  scripts/rsl_rl/play_collect_mesh_rollout_npz.py
  --task="${TASK}"
  --resume_path="${RESUME_PATH}"
  --pairs_jsonl="${PAIRS_JSONL}"
  --collect_domains="${COLLECT_DOMAINS}"
  --mesh_pair_indices="${MESH_PAIR_INDICES}"
  --flat_motion_file="${FLAT_MOTION_FILE}"
  --flat_dataset_txt="${FLAT_DATASET_TXT}"
  --flat_motion_indices="${FLAT_MOTION_INDICES}"
  --output_dir="${OUTPUT_DIR}"
  --run_name="${RUN_NAME}"
  --max_steps="${MAX_STEPS}"
  --encoder_mode=robot
  --domain_separator_cell_count="${DOMAIN_SEPARATOR_CELLS}"
  --motion_reset_z_offset="${MOTION_RESET_Z_OFFSET}"
)

if [[ "${HEADLESS}" != "0" ]]; then
  ARGS+=(--headless)
fi

"${PYTHON_BIN}" "${ARGS[@]}" "$@"

echo
echo "Visualize with:"
echo "python scripts/omniretarget/visualize_g1_terrain_pairs_viser.py \\"
echo "  --pairs_jsonl ${OUTPUT_DIR}/pairs_rollout.jsonl \\"
echo "  --show_height_scan"
