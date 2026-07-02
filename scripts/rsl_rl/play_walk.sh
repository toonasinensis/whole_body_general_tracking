#!/usr/bin/env bash

set -e

ARGS=(
  --task=HeadingWalkAMP-Modal-G1
  --num_envs=40
  --resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_2350.pt

  # --headless

  # Video recording. Output goes to <checkpoint_dir>/videos/play.
  # --video
  # --video_length=200

  # Exit automatically after N env steps.
  # --max_steps=1000

  # Export/load ONNX.
  # --export_onnx
  # --use_onnx_policy
  # --export_only
)

python scripts/rsl_rl/play.py "${ARGS[@]}" "$@"
