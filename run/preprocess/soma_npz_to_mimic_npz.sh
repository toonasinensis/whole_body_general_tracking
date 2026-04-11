#!/usr/bin/env bash
set -euo pipefail

# --- edit these if needed ---
ENV_PATH="$HOME/miniforge3/envs/env_isaaclab"
INPUT_DIR="/home/leju/workspace/leju_soma_retarget/outputs/roban/soma_uniform"
OUTPUT_DIR="./assets/roban_motions"

OUTPUT_FPS=50
NUM_ENVS=64
DEVICE="cuda:0"
PATTERN="*.npz"
# ---------------------------

conda run -p "$ENV_PATH" python preprocess/soma_npz_to_mimic_npz.py \
  --input_dir "$INPUT_DIR" \
  --pattern "$PATTERN" \
  --output_fps "$OUTPUT_FPS" \
  --num_envs "$NUM_ENVS" \
  --headless \
  --device "$DEVICE" \
  --overwrite \
  --no_wandb \
  --output_dir "$OUTPUT_DIR"