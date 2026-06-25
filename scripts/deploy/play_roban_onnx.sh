#!/usr/bin/env bash
set -euo pipefail

python scripts/deploy/play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --onnx_path logs/rsl_rl/roban_flat/2026-04-15_22-34-18/exported/policy.onnx \
  --motion_file data/roban_motions_quick_test \
  --num_envs 1