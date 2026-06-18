#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=AMP-RobanS22 \
--num_envs=4 \
--motion_file /home/leju/Documents/zjl/test-motion \
--resume_path /home/leju/Documents/zjl/whole_body_general_tracking/logs/rsl_rl/model_16000.pt