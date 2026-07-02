#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMulti-G1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ENVS="${NUM_ENVS:-4000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200000000}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29520}"
RUN_NAME="${RUN_NAME:-mixed_flat_mesh_smoke}"
PAIRS_JSONL="${PAIRS_JSONL:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/omniretarget/g1_terrain/cropped_pairs/pairs_cropped.jsonl}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-data/tracking_npz_data/walk_run.txt}"
FLAT_ENV_RATIO="${FLAT_ENV_RATIO:-0.75}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-32}"
HEADLESS="${HEADLESS:-1}"
RESUME_PATH="${RESUME_PATH:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_paired_terrain_height_scan/2026-06-22_23-22-06_mixed_flat_mesh_smoke/model_97000.pt}"

COMMON_ARGS=(
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task="${TASK}" \
  --resume=true \
  --resume_path="${RESUME_PATH}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --pairs_jsonl="${PAIRS_JSONL}"
  --flat_dataset_txt="${FLAT_DATASET_TXT}" \
  --flat_env_ratio="${FLAT_ENV_RATIO}" \
  --domain_separator_cell_count="${DOMAIN_SEPARATOR_CELLS}"
)

if [[ "${HEADLESS}" != "0" ]]; then
  COMMON_ARGS+=(--headless)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${COMMON_ARGS[@]}" \
    --distributed
else
  "${PYTHON_BIN}" "${COMMON_ARGS[@]}"
fi
