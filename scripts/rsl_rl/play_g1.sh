#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/rsl_rl/play.py \
--task=TR-G1 \
--num_envs=40 \
--resume_path=logs/rsl_rl/g1_flat/2026-06-22_17-17-23/model_21000.pt \
--motion_file=/home/kiki/workspace/motion_data/mimic_data/g1/bones_seed \
--dataset_txt=dataset_txt/quick.txt
