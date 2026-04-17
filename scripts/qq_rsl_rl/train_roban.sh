#!/usr/bin/env bash
set -euo pipefail

# Always run under wt_env unless caller already activated an environment.
# if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "wt_env" ]]; then
#   source /home/xiechunyang/miniforge3/etc/profile.d/conda.sh
#   conda activate wt_env
# fi

# Isaac-based training commonly runs one simulation process per GPU.
# Override when needed, e.g. NPROC_PER_NODE=2 ./scripts/rsl_rl/train.sh
NPROC_PER_NODE=1
NUM_ENVS="${NUM_ENVS:-4096}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/v_rsl_rl/train.py \
  --registry_name=test1 \
  --task=Tracking-Flat-RobanS22-v0 \
  --headless \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --motion_file=data/roban_motions \
  --motion_file_txt=data/roban_motions_list/quick_test.txt \
  # --logger wandb \
  # --log_project_name=roban_flat
#  --resume=true \
#  --resume_path="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_94500.pt"
