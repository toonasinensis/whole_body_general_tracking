#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMixedModal-G1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ENVS="${NUM_ENVS:-4000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200000000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29520}"
RUN_NAME="${RUN_NAME:-amp_tf_walk}"
PAIRS_JSONL="${PAIRS_JSONL:-${WBT_ROOT}/data/mesh_flatwbc_model151000_z10cm/selected_pairs.jsonl}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-${WBT_ROOT}/data/tracking_npz_data/walk_run1.txt}"
FLAT_WBC_ENV_RATIO="${FLAT_WBC_ENV_RATIO:-0.0}"
FLAT_VELOCITY_ENV_RATIO="${FLAT_VELOCITY_ENV_RATIO:-1.0}"
VELOCITY_TERRAIN_ENV_RATIO="${VELOCITY_TERRAIN_ENV_RATIO:-0.0}"
VELOCITY_TERRAIN_CELL_COUNT="${VELOCITY_TERRAIN_CELL_COUNT:-8}"
VELOCITY_TERRAIN_PROFILE="${VELOCITY_TERRAIN_PROFILE:-velocity_runway_steps}"
MESH_ENV_RATIO="${MESH_ENV_RATIO:-0.0}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-32}"
DEBUG_VELCOMMAND="${DEBUG_VELCOMMAND:-0}"
HEADLESS="${HEADLESS:-1}"
export WBT_DEBUG_VELCOMMAND="${DEBUG_VELCOMMAND}"
# RESUME_PATH="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_156000.pt"

echo "[INFO] Training task: ${TASK}"
echo "[INFO] Runner: TerrainPairMixedModal-G1 uses G1MixedTerrainHeightScanAMPModalityFusionRunnerCfg"
echo "[INFO] Modal tokens: prop, terrain, wbc_cmd(mask=aux_mask), velcommand(mask=vel_task_mask)"

COMMON_ARGS=(
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task="${TASK}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --pairs_jsonl="${PAIRS_JSONL}" \
  --flat_dataset_txt="${FLAT_DATASET_TXT}" \
  --flat_wbc_env_ratio="${FLAT_WBC_ENV_RATIO}" \
  --flat_velocity_env_ratio="${FLAT_VELOCITY_ENV_RATIO}" \
  --velocity_terrain_env_ratio="${VELOCITY_TERRAIN_ENV_RATIO}" \
  --velocity_terrain_cell_count="${VELOCITY_TERRAIN_CELL_COUNT}" \
  --velocity_terrain_profile="${VELOCITY_TERRAIN_PROFILE}" \
  --mesh_env_ratio="${MESH_ENV_RATIO}" \
  --domain_separator_cell_count="${DOMAIN_SEPARATOR_CELLS}" \
#   --resume=true \
#   --resume_path="${RESUME_PATH}" \
)

if [[ "${HEADLESS}" != "0" ]]; then
  COMMON_ARGS+=(--headless)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" \
    --distributed
else
  python "${COMMON_ARGS[@]}"
fi
