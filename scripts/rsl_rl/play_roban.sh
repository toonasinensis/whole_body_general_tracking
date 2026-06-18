#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=AMP-RobanS22 \
--num_envs=4 \
--export_onnx \
--export_only \
--motion_file=/home/zhangqiqi/Downloads/whole_body_general_tracking/data/Roban22_amp/Recovery/npz \
--resume_path=/home/zhangqiqi/Downloads/whole_body_general_tracking/logs/rsl_rl/roban22_amp/2026-06-09_14-26-15/model_9500.pt \
--headless 