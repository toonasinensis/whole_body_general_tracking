#!/usr/bin/env bash
set -euo pipefail

# Runs scripts/roban_tracking/play.py with RobanS22 task.
# Uses conda run so you don't need to manually activate the env.

ENV_PATH="${ENV_PATH:-$HOME/miniforge3/envs/env_isaaclab}"

TASK="${TASK:-Tracking-Flat-RobanS22-v0}"
MOTION_FILE="${MOTION_FILE:-assets/roban_motions/change_idle_left_to_idle_001__A021.npz}"
MODEL_PATH="${MODEL_PATH:-logs/rsl_rl/robanS22_flat/2026-04-10_17-57-46/model_13000.pt}"

NUM_ENVS="${NUM_ENVS:-16}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-40.0}"

HEADLESS="${HEADLESS:---headless}"
DEVICE="${DEVICE:-cuda}"

conda run -p "$ENV_PATH" python scripts/roban_tracking/play.py \
  --task "$TASK" \
  --motion_file "$MOTION_FILE" \
  --model_path "$MODEL_PATH" \
  --num_envs "$NUM_ENVS" \
  --episode_length_s "$EPISODE_LENGTH_S" \
  $HEADLESS \
  --device "$DEVICE"

