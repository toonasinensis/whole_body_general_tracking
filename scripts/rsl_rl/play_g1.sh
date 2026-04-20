#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=Tracking-Flat-G1-v0 \
--num_envs=10 \
--resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_18500.pt \
--motion_file=/home/thl/Documents/g1-mimic-npz \
--dataset_txt=whole_body_tracking/dataset_txt/hard.txt
