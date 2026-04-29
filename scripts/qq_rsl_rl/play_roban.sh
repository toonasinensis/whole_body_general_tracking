#!/usr/bin/env bash

python scripts/qq_rsl_rl/play.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=16 \
--resume_path=logs/rsl_rl/roban_flat/0422_all_kept/model_79000.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/model_79000_failed.txt \
--max_motion_num=160 \
--rendering_mode=performance
# --resume_path=logs/rsl_rl/roban_flat/0419_squat_kept/model_17000.pt \
# --resume_path=logs/rsl_rl/roban_flat/0419_main_kept/model_16300.pt \
