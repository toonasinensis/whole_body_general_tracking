#!/usr/bin/env bash

python scripts/rsl_rl/play.py \
--task=Tracking-Flat-G1-v0 \
--num_envs=20 \
--resume_path=/home/leju/workspace/whole_body_general_tracking/wandb/run-20260406_205309-dojb8cuj/files/model_31000.pt \
--motion_file=/home/leju/workspace/whole_body_general_tracking/data/g1-mimic-npz \
