#!/usr/bin/env bash

python scripts/deploy/export_onnx.py \
--task=AMP-RobanS22 \
--num_envs=1 \
--resume_path=/home/zhangqiqi/Downloads/whole_body_general_tracking/logs/rsl_rl/roban22_amp/2026-06-09_14-26-15/model_9500.pt \
--motion_file=data/Roban22_amp/Recovery/npz \
--export_onnx \
--headless