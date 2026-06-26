#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/rsl_rl/play.py \
--task=AMP-G1 \
--num_envs=40 \
--resume_path=logs/rsl_rl/g1_amp/2026-06-23_10-50-47/model_62500.pt \
--motion_file=/home/kiki/workspace/motion_data/mimic_data/g1/lafan \
--dataset_txt=dataset_txt/fall.txt \
--export_onnx \
--export_only
# --motion_file=/home/kiki/workspace/motion_data/mimic_data/g1/bones_seed \
# --dataset_txt=dataset_txt/quick.txt
