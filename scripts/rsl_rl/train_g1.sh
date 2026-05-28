#!/usr/bin/env bash
set -euo pipefail

# Always run under wt_env unless caller already activated an environment.
# if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "wt_env" ]]; then
#   source /home/xiechunyang/miniforge3/etc/profile.d/conda.sh
#   conda activate wt_env
# fi
#/home/thl/Documents/g1-mimic-npz
#/home/xiechunyang/wt_ws/wt_wbc/dataset/g1-mimic-npz
# Isaac-based training commonly runs one simulation process per GPU.
# Override when needed, e.g. NPROC_PER_NODE=2 ./scripts/rsl_rl/train.sh
#/home/xiechunyang/wt_ws/wt_wbc/dataset/smpl/smpl_filtered
NPROC_PER_NODE=1
NUM_ENVS="${NUM_ENVS:-8000}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=AMP-G1 \
  --headless \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --motion_file="/home/thl/Documents/g1-mimic-npz" \
  --dataset_txt="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/walk2_subject1.txt"  \
  --resume=True \
  --resume_path="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_amp/2026-05-27_11-56-18/model_23000.pt" \
  # --encoder_mode=robot \
