#!/usr/bin/env bash
set -euo pipefail

python scripts/deploy/play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --onnx_path logs/rsl_rl/roban_flat/0428_quick_test/exported/policy.onnx \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/v_exp.txt \
  --num_envs 1

  # --onnx_path logs/rsl_rl/roban_flat/0422_all_kept/exported/policy.onnx \
