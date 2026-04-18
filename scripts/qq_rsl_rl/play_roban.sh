#!/usr/bin/env bash

python scripts/qq_rsl_rl/play.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=40 \
--resume_path=logs/rsl_rl/roban_flat/2026-04-17_21-27-35/model_25600.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/motions_failed.txt \
--max_motion_num=500 \
