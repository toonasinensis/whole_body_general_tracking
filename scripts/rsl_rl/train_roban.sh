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
# Override when needed, e.g. NPROC_PER_NODE=4 ./scripts/rsl_rl/train.sh
#/home/xiechunyang/wt_ws/wt_wbc/dataset/smpl/smpl_filtered
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NUM_ENVS="${NUM_ENVS:-8192}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29520}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=AMP-RobanS22 \
  --headless \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --motion_file="/home/zhangqiqi/Downloads/data/roban_motions" \
  --dataset_txt="/home/zhangqiqi/Downloads/data/roban_motions_list/motions_all_kept.txt"  \
  --logger wandb \
  --log_project_name=roban_amp \
  # --resume=True \
  # --resume_path="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/model_81000.pt" \
  # --encoder_mode=robot \


