#!/usr/bin/env bash
# Replay the top-K adaptive bins for visualization:
#   scripts/eval/replay_adaptive_bins.py
#
# Usage examples:
#   bash scripts/eval/run_replay_adaptive_bins.sh
#   TOP_K=32 START_RANK=64 bash scripts/eval/run_replay_adaptive_bins.sh
#   BINS_JSON=data/eval_results/adaptive_bins_roban.json bash scripts/eval/run_replay_adaptive_bins.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-Tracking-Flat-RobanS22-v0}"
# JSON produced by scripts/eval/eval_adaptive_bins.py
BINS_JSON="${BINS_JSON:-data/eval_results/adaptive_bins_roban.json}"
TOP_K="${TOP_K:-16}"
START_RANK="${START_RANK:-0}"
ENV_SPACING="${ENV_SPACING:-2.5}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

# Visualization script: default is NOT headless.
EXTRA_ARGS=()
if [[ "${HEADLESS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--headless)
fi

python "${REPO_ROOT}/scripts/eval/replay_adaptive_bins.py" \
  --task="${TASK}" \
  --bins_json="${BINS_JSON}" \
  --top_k="${TOP_K}" \
  --start_rank="${START_RANK}" \
  --env_spacing="${ENV_SPACING}" \
  --seed="${SEED}" \
  --device="${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

