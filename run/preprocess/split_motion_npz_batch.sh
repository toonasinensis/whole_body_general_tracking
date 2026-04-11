#!/usr/bin/env bash
# Batch-split NPZs listed in a kept file (see preprocess/split_motion_npz.py split-batch).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- edit or override with environment variables ---
ENV_PATH="${ENV_PATH:-}" # e.g. $HOME/miniforge3/envs/env_isaaclab — leave empty to use `python` on PATH

INPUT_DIR="${INPUT_DIR:-assets/roban_motions}"
KEPT_LIST="${KEPT_LIST:-kept_and_passed_motions.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-assets/roban_motions_bins_50}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-5}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.1}"
# ----------------------------------------------------

if [[ -n "$ENV_PATH" ]]; then
  conda run -p "$ENV_PATH" --no-capture-output python preprocess/split_motion_npz.py split-batch \
    --input-dir "$INPUT_DIR" \
    --kept-list "$KEPT_LIST" \
    --output-dir "$OUTPUT_DIR" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --overlap-ratio "$OVERLAP_RATIO"
else
  python preprocess/split_motion_npz.py split-batch \
    --input-dir "$INPUT_DIR" \
    --kept-list "$KEPT_LIST" \
    --output-dir "$OUTPUT_DIR" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --overlap-ratio "$OVERLAP_RATIO"
fi
