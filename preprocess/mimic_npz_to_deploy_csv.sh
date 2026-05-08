#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

input="data/roban_motions_debug/mimic_npz/concated_motion_simple.npz"
output="data/roban_motions_debug/deploy_csv/concated_motion_simple_mimic.csv"


cmd=(
  python "${REPO_ROOT}/preprocess/mimic_npz_to_deploy_csv.py"
)
cmd+=(--input "${input}")
cmd+=(--output "${output}")

echo "[INFO] Running converter:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
