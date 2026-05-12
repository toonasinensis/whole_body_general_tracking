#!/usr/bin/env bash
set -euo pipefail

# torch.distributed.run listens on MASTER_PORT (default 29500). If you see
# EADDRINUSE, another job holds that port; kill it or set MASTER_PORT, e.g.:
#   MASTER_PORT=29511 bash scripts/dps_rsl_rl/train_roban.sh
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  # Pick a port in a high range to reduce collisions with the default 29500.
  export MASTER_PORT=$((29501 + RANDOM % 2000))
fi
echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
# NOTE USE your customized rsl_rl path here
export PYTHONPATH="/home/leju/workspace/sonic/rsl_rl:${PYTHONPATH:-}"

# Always run under wt_env unless caller already activated an environment.
# if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "wt_env" ]]; then
#   source /home/xiechunyang/miniforge3/etc/profile.d/conda.sh
#   conda activate wt_env
# fi

# Isaac-based training commonly runs one simulation process per GPU.
# Override when needed, e.g. NPROC_PER_NODE=2 bash scripts/dps_rsl_rl/train_roban.sh
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# PPO memory scales ~linearly with NUM_ENVS; 12288 is very likely to OOM on 24GB.
# Override at runtime: NUM_ENVS=4096 bash scripts/dps_rsl_rl/train_roban.sh
NUM_ENVS="${NUM_ENVS:-8192}"
# Cap number of motions loaded to avoid dataset OOM (override if needed).
MAX_MOTION_NUM="${MAX_MOTION_NUM:-5000}"
MOTION_FILE_TXT="${MOTION_FILE_TXT:-data/roban_motions_list/motions_all_kept.txt}"
# Optional pointer file for distributed loading. Each line should be a dataset txt path;
# rank i uses line i through commands.motion.dataset_txt_pointer.
MOTION_FILE_TXT_POINTER="${MOTION_FILE_TXT_POINTER:-data/roban_motions_list/motion_txt_list.txt}"
# Default run name = motion list filename without extension.
if [[ -n "${MOTION_FILE_TXT_POINTER}" ]]; then
  RUN_NAME="${RUN_NAME:-$(basename "${MOTION_FILE_TXT_POINTER}" .txt)}"
else
  RUN_NAME="${RUN_NAME:-$(basename "${MOTION_FILE_TXT}" .txt)}"
fi
FAIL_COUNT_SAVE_INTERVAL="${FAIL_COUNT_SAVE_INTERVAL:-12000}"

# Reduce allocator fragmentation for long training runs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_LOG_CHECKPOINTS="${WANDB_LOG_CHECKPOINTS:-0}"
export WANDB_LOG_GIT_FILES="${WANDB_LOG_GIT_FILES:-0}"

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

CMD=(
  python -m torch.distributed.run
  --nnodes=1
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  scripts/dps_rsl_rl/train.py
  --registry_name="${REGISTRY_NAME:-test1}"
  --task="${TASK_NAME:-Tracking-Flat-RobanS22-v0}"
  --headless
  --num_envs="${NUM_ENVS}"
  --motion_file="${MOTION_FILE:-data/roban_motions}"
  --run_name="${RUN_NAME}"
  --max_motion_num="${MAX_MOTION_NUM}"
  --fail_count_save_interval="${FAIL_COUNT_SAVE_INTERVAL}"
  # --logger "${LOGGER:-wandb}"
  # --log_project_name="${LOG_PROJECT_NAME:-roban_flat}"
  --distributed
)

if [[ -n "${MOTION_FILE_TXT_POINTER}" ]]; then
  CMD+=(--motion_file_txt_pointer="${MOTION_FILE_TXT_POINTER}")
else
  CMD+=(--motion_file_txt="${MOTION_FILE_TXT}")
fi

if [[ "${DISABLE_ADAPTIVE_BINS:-0}" == "1" ]]; then
  CMD+=(--disable_adaptive_bins)
fi

if [[ -n "${ADAPTIVE_BINS_LOG_DIR:-}" ]]; then
  CMD+=(--adaptive_bins_log_dir="${ADAPTIVE_BINS_LOG_DIR}")
fi

if [[ -n "${RESUME_PATH:-}" ]]; then
  CMD+=(--resume=true --resume_path="${RESUME_PATH}")
fi

"${CMD[@]}"
