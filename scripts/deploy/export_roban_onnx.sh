#!/usr/bin/env bash

python scripts/deploy/export_onnx.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=1 \
--resume_path=logs/rsl_rl/roban_flat/2026-05-05_13-06-33_motions_all_kept/model_17000.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/exp.txt \
--export_onnx \
--headless

