#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=FM-G1 \
--num_envs=30 \
--resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_31000.pt \
--motion_file=/home/thl/Documents/g1-mimic-npz \
--dataset_txt=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/fall.txt \
# --smpl_file_path="/home/thl/Documents/smpl/smpl_out/smpl_filtered"  \
# --encoder_mode="encoder_g1" \
# --export_onnx
