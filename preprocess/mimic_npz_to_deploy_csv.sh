#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

input="data/roban_motions/230131/body_check_001__A163.npz"
output="data/roban_deploy_motions/body_check_001__A163.csv"


cmd=(
  python "${REPO_ROOT}/preprocess/mimic_npz_to_deploy_csv.py"
)
cmd+=(--input "${input}")
cmd+=(--output "${output}")

echo "[INFO] Running converter:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"
