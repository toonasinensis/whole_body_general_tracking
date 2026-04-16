#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=4 \
--resume_path=logs/rsl_rl/roban_flat/2026-04-14_18-51-52/model_35500.pt \
--motion_file=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/roban/roban2
