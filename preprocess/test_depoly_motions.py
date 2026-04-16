"""
Load assigned deployed motions for testing joints orders
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

# MuJoCo Python bindings are mostly untyped; treat as Any for static analysis.
mj: Any = mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]

joints_order = [
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


def _load_motion_csv(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load tab-separated motion CSV; returns (data, column names)."""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        raise ValueError(f"Empty file: {path}")
    header = text[0].split("\t")
    rows = [[float(x) for x in line.split("\t")] for line in text[1:] if line.strip()]
    if not rows:
        raise ValueError(f"No data rows in: {path}")
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != len(header):
        raise ValueError(f"Column count mismatch: header {len(header)} vs row {data.shape[1]}")
    return data, header


def _joint_qpos_indices(model: Any, joint_names: list[str]) -> list[int]:
    """ MuJoCo qpos index for each 1-DoF joint (address of scalar component). """
    out: list[int] = []
    for name in joint_names:
        jid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        adr = int(model.jnt_qposadr[jid])
        jtype = int(model.jnt_type[jid])
        if jtype != int(mj.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected hinge joint, got type {jtype} for {name}")
        out.append(adr)
    return out


def _joint_dof_indices(model: Any, joint_names: list[str]) -> list[int]:
    """ MuJoCo qvel index for each 1-DoF joint (address of scalar component). """
    out: list[int] = []
    for name in joint_names:
        jid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise ValueError(f"Joint not found in model: {name}")
        adr = int(model.jnt_dofadr[jid])
        jtype = int(model.jnt_type[jid])
        if jtype != int(mj.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"Expected hinge joint, got type {jtype} for {name}")
        out.append(adr)
    return out


def apply_motion_row(
    data: Any,
    row: np.ndarray,
    col: dict[str, int],
    joint_qpos_idx: list[int],
    joint_dof_idx: list[int],
) -> None:
    """Write one CSV row into mjData qpos / qvel."""
    data.qpos[0] = row[col["body_pos_x"]]
    data.qpos[1] = row[col["body_pos_y"]]
    data.qpos[2] = row[col["body_pos_z"]]
    data.qpos[3] = row[col["body_quat_w"]]
    data.qpos[4] = row[col["body_quat_x"]]
    data.qpos[5] = row[col["body_quat_y"]]
    data.qpos[6] = row[col["body_quat_z"]]

    n_j = len(joint_qpos_idx)
    for k in range(n_j):
        key_pos = f"joint_pos{k:02d}"
        key_vel = f"joint_vel_{k:02d}"
        if key_pos not in col or key_vel not in col:
            raise KeyError(f"Missing columns {key_pos} or {key_vel}")
        data.qpos[joint_qpos_idx[k]] = row[col[key_pos]]
        data.qvel[joint_dof_idx[k]] = row[col[key_vel]]


def visualize_motion(
    model: Any,
    motion: np.ndarray,
    col: dict[str, int],
    joint_names: list[str],
    fps: float,
) -> None:
    joint_qpos_idx = _joint_qpos_indices(model, joint_names)
    joint_dof_idx = _joint_dof_indices(model, joint_names)

    data = mj.MjData(model)
    frame_dt = 1.0 / fps
    n_frames = motion.shape[0]

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            apply_motion_row(data, motion[frame], col, joint_qpos_idx, joint_dof_idx)
            mj.mj_forward(model, data)
            viewer.sync()
            frame = (frame + 1) % n_frames
            time.sleep(frame_dt)


def main() -> None:
    p = argparse.ArgumentParser(description="Replay deploy motion CSV in MuJoCo.")
    p.add_argument(
        "--motion",
        type=Path,
        default=_REPO_ROOT / "data/roban_deploy_motions/newdance_03_Skeleton.csv",
        help="Tab-separated motion CSV (body pose + joint_posNN + joint_vel_NN).",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=_REPO_ROOT / "source/whole_body_tracking/whole_body_tracking/assets/roban_s22/xml/scene.xml",
        help="MuJoCo scene XML.",
    )
    p.add_argument("--fps", type=float, default=50.0, help="Playback rate (CSV is usually 50 Hz).")
    args = p.parse_args()

    motion_path = args.motion if args.motion.is_absolute() else _REPO_ROOT / args.motion
    model_path = args.model if args.model.is_absolute() else _REPO_ROOT / args.model

    motion, header = _load_motion_csv(motion_path)
    col = {name: i for i, name in enumerate(header)}

    n_csv_joints = sum(1 for name in header if name.startswith("joint_pos"))
    if n_csv_joints != len(joints_order):
        raise ValueError(
            f"CSV has {n_csv_joints} joint_pos* columns but `joints_order` has {len(joints_order)} names."
        )

    model = mj.MjModel.from_xml_path(str(model_path))
    visualize_motion(model, motion, col, joints_order, args.fps)


if __name__ == "__main__":
    main()
