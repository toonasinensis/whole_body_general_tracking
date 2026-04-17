#!/usr/bin/env bash

python scripts/deploy/export_onnx.py \
--task=Tracking-Flat-RobanS22-v0 \
--num_envs=1 \
--resume_path=logs/rsl_rl/roban_flat/2026-04-15_22-34-18/model_3500.pt \
--motion_file=data/roban_motions_quick_test \
--export_onnx
