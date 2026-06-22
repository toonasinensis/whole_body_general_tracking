#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-TerrainPairMixed-G1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ENVS="${NUM_ENVS:-32}"
MAX_ITERATIONS="${MAX_ITERATIONS:-200}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29520}"
RUN_NAME="${RUN_NAME:-mixed_flat_mesh_smoke}"
PAIRS_JSONL="${PAIRS_JSONL:-data/omniretarget/g1_terrain/pairs.jsonl}"
FLAT_DATASET_TXT="${FLAT_DATASET_TXT:-data/tracking_npz_data/lafan_named.txt}"
FLAT_ENV_RATIO="${FLAT_ENV_RATIO:-0.25}"
DOMAIN_SEPARATOR_CELLS="${DOMAIN_SEPARATOR_CELLS:-32}"
HEADLESS="${HEADLESS:-1}"

COMMON_ARGS=(
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task="${TASK}" \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --pairs_jsonl="${PAIRS_JSONL}" \
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
