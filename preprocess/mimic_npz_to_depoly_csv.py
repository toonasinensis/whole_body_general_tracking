"""
Convert mimic / tracking NPZ motions to tab-separated deploy CSV (MuJoCo / hardware replay).

NPZ layout (this repo’s Roban mimic exports, e.g. ``data/roban_motions/...``)
----------------------------------------------------------------------------
Arrays (typical shapes):

- ``fps`` — scalar (or length-1 array), Hz.
- ``joint_pos`` — ``(T, 21)``, ``joint_vel`` — ``(T, 21)``.
- ``body_pos_w`` — ``(T, B, 3)``, ``body_quat_w`` — ``(T, B, 4)`` world-frame;
  ``body_lin_vel_w``, ``body_ang_vel_w`` — ``(T, B, 3)`` (not written to CSV).

**Joint column order in the NPZ** is not stored in every file. Two conventions are used:

1. **deploy** — Same 21-name order as the deploy CSV (see below). Matches the *source*
   layout in ``utilities/fix_joint_order.py`` (folder ``210531`` style exports).
2. **isaac** — Interleaved order produced as the *destination* layout by
   ``utilities/fix_joint_order.py`` (``dst_joint_order``), i.e. Isaac Lab–style
   column ordering.

If the NPZ contains a ``joint_names`` array (length 21, same names as below), the
script builds the permutation to **deploy** order automatically and ignores
``--npz-joint-layout``.

Deploy CSV layout (output)
--------------------------
Tab-separated columns:

- ``body_pos_x``, ``body_pos_y``, ``body_pos_z`` — root / anchor translation.
- ``body_quat_w``, ``body_quat_x``, ``body_quat_y``, ``body_quat_z`` — quaternion **wxyz**.
- ``joint_pos00`` … ``joint_pos20`` — positions in this fixed order:

  * ``waist_yaw_joint``, ``leg_l1_joint`` … ``leg_l6_joint``, ``leg_r1_joint`` … ``leg_r6_joint``,
    ``zarm_l1_joint`` … ``zarm_l4_joint``, ``zarm_r1_joint`` … ``zarm_r4_joint``.
- ``joint_vel_00`` … ``joint_vel_20`` — velocities in the **same** joint order as ``joint_pos*``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Column order for deploy CSV / joint_posNN (matches preprocess/test_depoly_motions.py).
DEPLOY_JOINT_ORDER: list[str] = [
    "waist_yaw_joint",
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l3_joint",
    "leg_l4_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
]

# NPZ columns after ``utilities.fix_joint_order`` destination reorder (Isaac-style interleaving).
ISAAC_NPZ_JOINT_ORDER: list[str] = [
    "waist_yaw_joint",
    "zarm_l1_joint",
    "zarm_r1_joint",
    "leg_l1_joint",
    "leg_r1_joint",
    "zarm_l2_joint",
    "zarm_r2_joint",
    "leg_l2_joint",
    "leg_r2_joint",
    "zarm_l3_joint",
    "zarm_r3_joint",
    "leg_l3_joint",
    "leg_r3_joint",
    "zarm_l4_joint",
    "zarm_r4_joint",
    "leg_l4_joint",
    "leg_r4_joint",
    "leg_l5_joint",
    "leg_r5_joint",
    "leg_l6_joint",
    "leg_r6_joint",
]


def _build_reorder_index(src_order: list[str], dst_order: list[str]) -> np.ndarray:
    """Indices so ``out[:, j] = inp[:, idx[j]]`` with ``dst_order`` column semantics."""
    missing = [j for j in dst_order if j not in src_order]
    if missing:
        raise ValueError(f"Target joints missing in source order: {missing}")
    extra = [j for j in src_order if j not in dst_order]
    if extra:
        raise ValueError(f"Source has joints not in target order: {extra}")
    src_to_idx = {name: i for i, name in enumerate(src_order)}
    return np.asarray([src_to_idx[name] for name in dst_order], dtype=np.int64)


def _parse_joint_names_array(raw: np.ndarray) -> list[str]:
    if raw.shape != (len(DEPLOY_JOINT_ORDER),):
        raise ValueError(
            f"joint_names must have shape ({len(DEPLOY_JOINT_ORDER)},), got {raw.shape}"
        )
    if raw.dtype.kind in "SU":
        return [raw[i].decode("utf-8") if isinstance(raw[i], bytes) else str(raw[i]) for i in range(raw.size)]
    return [str(raw[i]) for i in range(int(raw.size))]


def _joint_reorder_index(npz: Any, layout: str) -> np.ndarray:
    if "joint_names" in npz.files:
        names = _parse_joint_names_array(np.asarray(npz["joint_names"]).reshape(-1))
        return _build_reorder_index(names, DEPLOY_JOINT_ORDER)
    if layout == "deploy":
        return _build_reorder_index(DEPLOY_JOINT_ORDER, DEPLOY_JOINT_ORDER)
    if layout == "isaac":
        return _build_reorder_index(ISAAC_NPZ_JOINT_ORDER, DEPLOY_JOINT_ORDER)
    raise ValueError(f"Unknown --npz-joint-layout: {layout!r} (use deploy or isaac)")


def npz_to_deploy_rows(npz: Any, anchor_body_index: int, layout: str) -> np.ndarray:
    """Return ``(T, 7 + 21 + 21)`` float array: pos(3), quat wxyz(4), joint_pos(21), joint_vel(21)."""
    j_idx = _joint_reorder_index(npz, layout)

    joint_pos = np.asarray(npz["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(npz["joint_vel"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_vel.ndim != 2:
        raise ValueError(f"joint_pos / joint_vel must be 2D, got {joint_pos.shape}, {joint_vel.shape}")
    if joint_pos.shape != joint_vel.shape:
        raise ValueError(f"joint_pos {joint_pos.shape} != joint_vel {joint_vel.shape}")
    if joint_pos.shape[1] != len(DEPLOY_JOINT_ORDER):
        raise ValueError(
            f"Expected {len(DEPLOY_JOINT_ORDER)} joints, got {joint_pos.shape[1]}"
        )

    joint_pos_d = joint_pos[:, j_idx]
    joint_vel_d = joint_vel[:, j_idx]

    pos_w = np.asarray(npz["body_pos_w"], dtype=np.float64)
    quat_w = np.asarray(npz["body_quat_w"], dtype=np.float64)
    if pos_w.ndim != 3 or quat_w.ndim != 3:
        raise ValueError(f"body_pos_w / body_quat_w must be 3D, got {pos_w.shape}, {quat_w.shape}")
    t = joint_pos_d.shape[0]
    if pos_w.shape[0] != t or quat_w.shape[0] != t:
        raise ValueError("Time dimension mismatch between joints and bodies.")
    if anchor_body_index < 0 or anchor_body_index >= pos_w.shape[1]:
        raise ValueError(
            f"anchor_body_index {anchor_body_index} out of range for B={pos_w.shape[1]}"
        )

    root_p = pos_w[:, anchor_body_index, :]
    root_q = quat_w[:, anchor_body_index, :]

    return np.concatenate([root_p, root_q, joint_pos_d, joint_vel_d], axis=1)


def _csv_header() -> list[str]:
    h = ["body_pos_x", "body_pos_y", "body_pos_z", "body_quat_w", "body_quat_x", "body_quat_y", "body_quat_z"]
    h += [f"joint_pos{k:02d}" for k in range(len(DEPLOY_JOINT_ORDER))]
    h += [f"joint_vel_{k:02d}" for k in range(len(DEPLOY_JOINT_ORDER))]
    return h


def write_deploy_csv(path: Path, rows: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(_csv_header())
    # Match existing deploy files: 6 decimal places, tab-separated.
    fmt = ["%.6f"] * rows.shape[1]
    np.savetxt(path, rows, fmt=fmt, delimiter="\t", header=header, comments="")


def convert_file(
    npz_path: Path,
    csv_path: Path,
    anchor_body_index: int,
    layout: str,
) -> None:
    with np.load(npz_path, allow_pickle=True) as npz:
        rows = npz_to_deploy_rows(npz, anchor_body_index, layout)
    write_deploy_csv(csv_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Roban mimic NPZ to deploy tab-CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "data/roban_motions/210531/jump_and_land_heavy_001__A001_M.npz",
        help="Input .npz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data/roban_deploy_motions/jump_and_land_heavy_001__A001_M.csv",
        help="Output .csv path.",
    )
    parser.add_argument(
        "--npz-joint-layout",
        choices=("deploy", "isaac"),
        default="deploy",
        help="Column order of joint_pos in the NPZ if 'joint_names' is absent. "
        "'deploy' matches DEPLOY_JOINT_ORDER; 'isaac' matches utilities/fix_joint_order dst order.",
    )
    parser.add_argument(
        "--anchor-body-index",
        type=int,
        default=0,
        help="Index into body_pos_w / body_quat_w for root pose written to the CSV (default 0 = first body).",
    )
    args = parser.parse_args()

    inp = args.input if args.input.is_absolute() else _REPO_ROOT / args.input
    out = args.output if args.output.is_absolute() else _REPO_ROOT / args.output
    convert_file(inp, out, args.anchor_body_index, args.npz_joint_layout)
    print(f"Wrote {out} ({inp.name})")


if __name__ == "__main__":
    main()
