#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=FM-G1 \
--num_envs=40 \
--resume_path=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_22000.pt \
--motion_file=/home/thl/Documents/g1-mimic-npz \
--dataset_txt=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/tracking_npz_data/fall.txt
