#!/usr/bin/env bash
# Split one motion NPZ into clips (see preprocess/split_motion_npz.py split).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- edit or override with environment variables ---
ENV_PATH="${ENV_PATH:-}" # leave empty to use `python` on PATH

# First argument overrides INPUT_NPZ if given.
INPUT_NPZ="${INPUT_NPZ:-}"
if [[ $# -ge 1 ]]; then
  INPUT_NPZ="$1"
fi
OUTPUT_DIR="${OUTPUT_DIR:-assets/roban_motions_bins_50}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-4}"
OVERLAP_RATIO="${OVERLAP_RATIO:-0.5}"
# ----------------------------------------------------

if [[ -z "$INPUT_NPZ" ]]; then
  echo "Usage: $0 <path-to-motion.npz>" >&2
  echo "   or: INPUT_NPZ=path/to/file.npz $0" >&2
  exit 1
fi

if [[ -n "$ENV_PATH" ]]; then
  conda run -p "$ENV_PATH" --no-capture-output python preprocess/split_motion_npz.py split \
    --input "$INPUT_NPZ" \
    --output-dir "$OUTPUT_DIR" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --overlap-ratio "$OVERLAP_RATIO"
else
  python preprocess/split_motion_npz.py split \
    --input "$INPUT_NPZ" \
    --output-dir "$OUTPUT_DIR" \
    --segment-seconds "$SEGMENT_SECONDS" \
    --overlap-ratio "$OVERLAP_RATIO"
fi
