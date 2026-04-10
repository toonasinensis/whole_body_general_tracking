#!/usr/bin/env bash
set -euo pipefail

# Runs scripts/roban_tracking/train.py with RobanS22 task.
# Uses conda run so you don't need to manually activate the env.

ENV_PATH="${ENV_PATH:-$HOME/miniforge3/envs/env_isaaclab}"

TASK="${TASK:-Tracking-Flat-RobanS22-v0}"
MOTION_FILE="${MOTION_FILE:-assets/roban_motions/looking_in_the_mirror_exaggerated_002__A001.npz}"

HEADLESS="${HEADLESS:---headless}"
DEVICE="${DEVICE:-cuda}"

LOGGER="${LOGGER:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-roban_flat}"

conda run -p "$ENV_PATH" python scripts/roban_tracking/train.py \
  --task "$TASK" \
  --motion_file "$MOTION_FILE" \
  $HEADLESS \
  --device "$DEVICE" \
  --logger "$LOGGER" \
  --log_project_name "$WANDB_PROJECT"

