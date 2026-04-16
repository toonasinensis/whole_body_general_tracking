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
NUM_ENVS="${NUM_ENVS:-16000}"

python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/rsl_rl/train.py \
  --registry_name=test1 \
  --task=Tracking-Flat-RobanS22-v0 \
  --headless \
  --kit_args="--/physics/collisionApproximateCylinders=true" \
  --distributed \
  --num_envs="${NUM_ENVS}" \
  --motion_file=/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/roban/roban2 \
#  --resume=true \
#  --resume_path="/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/model_94500.pt"
