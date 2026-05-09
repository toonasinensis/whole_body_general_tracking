#!/usr/bin/env bash

python scripts/deploy/export_onnx.py \
--task=Tracking-Flat-RobanS22-v0 \
--resume_path=logs/rsl_rl/roban_flat/0505_crawl_kept/model_41500.pt \
--motion_file=data/roban_motions \
--motion_file_txt=data/roban_motions_list/exp.txt \
--headless
