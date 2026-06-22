#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/thl/wt_wbc/IsaacLab}"
WBT_ROOT="${WBT_ROOT:-/home/thl/wt_wbc/wbc_parkour/whole_body_tracking}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/home/thl/isaacsim5.1}"
ISAACLAB_CONDA_PREFIX="${ISAACLAB_CONDA_PREFIX:-}"
MY_ENV_PREFIX="${MY_ENV_PREFIX:-/home/thl/miniconda3/envs/my_env}"
MY_ENV_SITE_PACKAGES="${MY_ENV_SITE_PACKAGES:-}"
OMNIRETARGET_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OMNIRETARGET_BOOTSTRAP_DIR="${OMNIRETARGET_SCRIPT_DIR}/python_bootstrap"

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi

if [[ -z "${MY_ENV_SITE_PACKAGES}" && -x "${MY_ENV_PREFIX}/bin/python" ]]; then
  MY_ENV_SITE_PACKAGES="$("${MY_ENV_PREFIX}/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
fi

if [[ -n "${MY_ENV_SITE_PACKAGES}" ]]; then
  export OMNIRETARGET_EXTRA_SITE_PACKAGES="${MY_ENV_SITE_PACKAGES}${OMNIRETARGET_EXTRA_SITE_PACKAGES:+:${OMNIRETARGET_EXTRA_SITE_PACKAGES}}"
fi

PYTHONPATH_ENTRIES=(
  "${OMNIRETARGET_BOOTSTRAP_DIR}"
  "${ISAACLAB_ROOT}/source/isaaclab"
  "${ISAACLAB_ROOT}/source/isaaclab_assets"
  "${ISAACLAB_ROOT}/source/isaaclab_contrib"
  "${ISAACLAB_ROOT}/source/isaaclab_mimic"
  "${ISAACLAB_ROOT}/source/isaaclab_rl"
  "${ISAACLAB_ROOT}/source/isaaclab_tasks"
  "${WBT_ROOT}/source/whole_body_tracking"
  "${WBT_ROOT}/source/whole_body_tracking/whole_body_tracking/utils/motionlib/smpl_math_utils"
  "${WBT_ROOT}/source/whole_body_tracking/whole_body_tracking/utils/motionlib/smpl_motion_lib"
)
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ISAACLAB_CONDA_PREFIX}" ]]; then
  CONDA_PREFIX="${ISAACLAB_CONDA_PREFIX}" VIRTUAL_ENV= "${ISAACLAB_ROOT}/isaaclab.sh" -p "$@"
else
  CARB_APP_PATH="${ISAACSIM_ROOT}/kit" \
  ISAAC_PATH="${ISAACSIM_ROOT}" \
  EXP_PATH="${ISAACSIM_ROOT}/apps" \
  CONDA_PREFIX= \
  VIRTUAL_ENV= \
  "${ISAACSIM_ROOT}/python.sh" "$@"
fi
