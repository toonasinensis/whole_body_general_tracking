#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=FM-G1 \
--num_envs=50 \
--resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_59500.pt \
--motion_file=/home/thl/Documents/g1-mimic-npz \
--dataset_txt=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/all_top_files.txt \
--export_onnx \
--use_onnx_policy \
--export_only
# --encoder_mode=robot \
