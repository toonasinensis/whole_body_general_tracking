#!/usr/bin/env bash
# Run adaptive-bin evaluation (scripts/eval/eval_adaptive_bins.py) with the same
# motion flags as training. Override any variable below or export before calling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASK="${TASK:-Tracking-Flat-RobanS22-v0}"
# Set to your checkpoint (relative to repo root or absolute).
RESUME_PATH="${RESUME_PATH:-logs/rsl_rl/roban_flat/2026-04-17_21-27-35/model_25600.pt}"
MOTION_FILE="${MOTION_FILE:-data/roban_motions}"
# Optional: same as train.py --motion_file_txt (unset = rglob all .npz under MOTION_FILE).
# Example: export MOTION_FILE_TXT=data/roban_motions_list/quick_test.txt
MOTION_FILE_TXT="${MOTION_FILE_TXT:-data/roban_motions_list/motions_main_kept_500.txt}"
MAX_MOTION_NUM="${MAX_MOTION_NUM:-5000}"
STEPS="${STEPS:-8000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
OUT="${OUT:-data/eval_results/adaptive_bins_run.json}"
OUT_TXT="${OUT_TXT:-data/eval_results/adaptive_bins_run_motions_sorted.txt}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"

EXTRA_ARGS=()
if [[ "${HEADLESS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--headless)
fi
if [[ -n "${MOTION_FILE_TXT:-}" ]]; then
  EXTRA_ARGS+=(--motion_file_txt="${MOTION_FILE_TXT}")
fi

python "${REPO_ROOT}/scripts/eval/eval_adaptive_bins.py" \
  --task="${TASK}" \
  --resume_path="${RESUME_PATH}" \
  --motion_file="${MOTION_FILE}" \
  --max_motion_num="${MAX_MOTION_NUM}" \
  --steps="${STEPS}" \
  --warmup_steps="${WARMUP_STEPS}" \
  --out="${OUT}" \
  --seed="${SEED}" \
  --device="${DEVICE}" \
  --out_txt="${OUT_TXT}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
