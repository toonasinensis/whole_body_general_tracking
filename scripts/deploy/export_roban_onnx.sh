#!/usr/bin/env bash

python scripts/deploy/export_onnx.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=1 \
--resume_path=logs/rsl_rl/roban_flat/0428_quick_test/model_13200.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/quick_test.txt \
--export_onnx

# --resume_path=logs/rsl_rl/roban_flat/0420_crawl_kept/model_13200.pt \
