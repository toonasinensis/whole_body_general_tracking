#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root (two levels up from this script).
# This makes paths like preprocess/... and data/... resolve correctly
# no matter where the script is launched from.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="data/roban_soma_motions"
OUTPUT_DIR="data/roban_motions"
PATTERN="*.npz"
OUTPUT_FPS=50
NUM_ENVS=64
DEVICE="cuda:0"   # change to cpu or cuda:1 if needed

mkdir -p "$OUTPUT_DIR"

python preprocess/soma_npz_to_mimic_npz.py \
  --input_dir "$INPUT_DIR" \
  --pattern "$PATTERN" \
  --output_dir "$OUTPUT_DIR" \
  --output_fps "$OUTPUT_FPS" \
  --num_envs "$NUM_ENVS" \
  --device "$DEVICE" \
  --headless \
  --overwrite \
  --no_wandb