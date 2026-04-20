#!/usr/bin/env bash

python scripts/qq_rsl_rl/play.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=16 \
--resume_path=logs/rsl_rl/roban_flat/0419_main_kept/model_16300.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/motions_main_kept.txt \
--max_motion_num=160 \
# --resume_path=logs/rsl_rl/roban_flat/2026-04-19_13-07-20/model_17000.pt \
