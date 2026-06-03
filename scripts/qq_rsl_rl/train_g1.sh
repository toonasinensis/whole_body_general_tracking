#!/usr/bin/env bash
set -euo pipefail

# torch.distributed.run listens on MASTER_PORT (default 29500). If you see
# EADDRINUSE, another job holds that port — kill it or set MASTER_PORT, e.g.:
#   MASTER_PORT=29511 bash scripts/qq_rsl_rl/train_roban.sh
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  # Pick a port in a high range to reduce collisions with the default 29500.
  export MASTER_PORT=$((29501 + RANDOM % 2000))
fi
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# Always run under wt_env unless caller already activated an environment.
# if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "wt_env" ]]; then
#   source /home/xiechunyang/miniforge3/etc/profile.d/conda.sh
#   conda activate wt_env
# fi

# Isaac-based training commonly runs one simulation process per GPU.
# Override when needed, e.g. NPROC_PER_NODE=2 ./scripts/rsl_rl/train.sh
NPROC_PER_NODE=1
# PPO memory scales ~linearly with NUM_ENVS; 12288 is very likely to OOM on 24GB.
# Override at runtime: NUM_ENVS=4096 bash scripts/qq_rsl_rl/train_roban.sh
NUM_ENVS="${NUM_ENVS:-8192}"
# Cap number of motions loaded to avoid dataset OOM (override if needed).
MAX_MOTION_NUM="${MAX_MOTION_NUM:-25000}"
MOTION_FILE_TXT="${MOTION_FILE_TXT:-dataset_txt/bones_seed/squat.txt}"
# Default run name = motion list filename without extension.
RUN_NAME="${RUN_NAME:-$(basename "${MOTION_FILE_TXT}" .txt)}"

# Reduce allocator fragmentation for long training runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# W&B: log curves/metrics but avoid uploading every model_*.pt (large). Checkpoints stay under logs/.
# Override: WANDB_LOG_CHECKPOINTS=1 to upload checkpoints again. See MotionOnPolicyRunner.
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
# Preflight: avoid Isaac Sim startup failures when GPU is already full.
if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_MB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "")"
  if [[ -n "${FREE_MB}" ]] && [[ "${FREE_MB}" -lt 2048 ]]; then
    echo "[ERROR] GPU free memory is too low (${FREE_MB} MiB). Isaac Sim may fail to create CUDA context."
    echo "[INFO] Top GPU processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | head -n 20 || true
    echo ""
    echo "Please stop the above process(es) or reboot, then rerun."
    exit 1
  fi
fi

# Pass --master_port/--master_addr here: MASTER_PORT in the environment alone
# does not change the elastic rendezvous listener (it still defaulted to 29500).
python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  scripts/qq_rsl_rl/train.py \
  --registry_name=test1 \
  --task=Tracking-Flat-G1-v0 \
  --headless \
  --num_envs="${NUM_ENVS}" \
  --motion_file=/home/kiki/workspace/motion_data/mimic_data/g1/bones_seed \
  --motion_file_txt="${MOTION_FILE_TXT}" \
  --run_name="${RUN_NAME}" \
  --max_motion_num="${MAX_MOTION_NUM}" \
  --logger tensorboard \
#   --log_project_name=g1_flat \
#   --resume=true \
#   --resume_path="logs/rsl_rl/roban_flat/0509_crawl_squat_kept/model_76500.pt"
#   --distributed
