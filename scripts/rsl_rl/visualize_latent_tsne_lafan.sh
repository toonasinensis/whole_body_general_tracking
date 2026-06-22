#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/thl/wt_wbc/IsaacLab/isaaclab.sh -p}"
PYTHON_CMD=(${ISAAC_PYTHON})

TASK="${TASK:-LatentDistill-Flat-G1-v0}"
NUM_ENVS="${NUM_ENVS:-64}"
RESUME_STUDENT_PATH="${RESUME_STUDENT_PATH:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_latent_distill/2026-06-09_16-47-56_latent_vib_student/model_500.pt}"
MOTION_FILE="${MOTION_FILE:-/home/thl/Documents/g1-mimic-npz}"
DATASET_TXT="${DATASET_TXT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/dataset_txt/lafan.txt}"
SMPL_FILE_PATH="${SMPL_FILE_PATH:-}"
OUT_DIR="${OUT_DIR:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/latent_tsne/lafan}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-lafan_latent_tsne}"
NUM_SAMPLES="${NUM_SAMPLES:-4096}"
MAX_TSNE_POINTS="${MAX_TSNE_POINTS:-4096}"
PERPLEXITY="${PERPLEXITY:-30}"
LABEL_MODE="${LABEL_MODE:-lafan_action}"
HEADLESS="${HEADLESS:-1}"

if [[ -z "${RESUME_STUDENT_PATH}" ]]; then
  echo "RESUME_STUDENT_PATH must point to a distilled student checkpoint." >&2
  exit 1
fi

ARGS=(
  --task="${TASK}"
  --num_envs="${NUM_ENVS}"
  --resume_path="${RESUME_STUDENT_PATH}"
  --motion_file="${MOTION_FILE}"
  --dataset_txt="${DATASET_TXT}"
  --out_dir="${OUT_DIR}"
  --output_prefix="${OUTPUT_PREFIX}"
  --num_samples="${NUM_SAMPLES}"
  --max_tsne_points="${MAX_TSNE_POINTS}"
  --perplexity="${PERPLEXITY}"
  --label_mode="${LABEL_MODE}"
)

if [[ -n "${SMPL_FILE_PATH}" ]]; then
  ARGS+=(--smpl_file_path="${SMPL_FILE_PATH}")
fi
if [[ "${HEADLESS}" == "1" ]]; then
  ARGS+=(--headless)
fi

"${PYTHON_CMD[@]}" "${PROJECT_DIR}/scripts/rsl_rl/visualize_latent_tsne.py" "${ARGS[@]}" "$@"
