#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WBT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${WBT_ROOT}"
export WBT_ROOT

export CUDA_VISIBLE_DEVICES="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="${WBT_ROOT}/../rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-${WBT_ROOT}/scripts/omniretarget/run_with_isaaclab_python.sh}"

TASK="${TASK:-HeadingWalkAMP-Modal-G1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NUM_ENVS="${NUM_ENVS:-4000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29330}"
REGISTRY_NAME="${REGISTRY_NAME:-test1}"
RUN_NAME="${RUN_NAME:-heading_walk_amp_mlp}"
HEADLESS="${HEADLESS:-1}"

COMMON_ARGS=(
  scripts/rsl_rl/train.py
  --registry_name="${REGISTRY_NAME}"
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --run_name="${RUN_NAME}"
)

if [[ -n "${MAX_ITERATIONS}" ]]; then
  COMMON_ARGS+=(--max_iterations "${MAX_ITERATIONS}")
fi

if [[ "${HEADLESS}" != "0" ]]; then
  COMMON_ARGS+=(--headless)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --kit_args="--/physics/collisionApproximateCylinders=true" \
    "${COMMON_ARGS[@]}" \
    --distributed \
    "$@"
else
  python "${COMMON_ARGS[@]}" "$@"
fi
