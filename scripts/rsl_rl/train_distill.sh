#!/usr/bin/env bash

NPROC_PER_NODE=1
NUM_ENVS="${NUM_ENVS:-4096}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/rsl_rl/train_distill.py \
  --registry_name=test1 \
  --task=Distill-Flat-G1-v0 \
  --headless \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --resume_teacher_path="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_25000.pt" \
  # --resume_student=true \
  # --resume_student="/home/xiechunyang/wt_ws/wt_wbc/wbc_parkour/wbc_parkour/whole_body_tracking/logs/model_2000.pt"

#  --resume=true \
#  --resume_path="/home/xiechunyang/wt_ws/wt_wbc/wbc_parkour/wbc_parkour/whole_body_tracking/logs/model_2000.pt"
