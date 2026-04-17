#!/usr/bin/env bash

python scripts/v_rsl_rl/play.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=40 \
--resume_path=logs/rsl_rl/roban_flat/2026-04-15_22-34-18/model_3500.pt \
--motion_file=assets/roban_motions_quick_test \
