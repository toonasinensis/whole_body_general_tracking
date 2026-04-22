#!/usr/bin/env bash
set -euo pipefail

python scripts/deploy/play_onnx.py \
  --task Tracking-Flat-RobanS22-v0 \
  --onnx_path logs/rsl_rl/roban_flat/0420_crawl_kept/exported/policy.onnx \
  --motion_file data/roban_motions \
  --motion_file_txt data/roban_motions_list/motions_crawl_kept.txt \
  --num_envs 1