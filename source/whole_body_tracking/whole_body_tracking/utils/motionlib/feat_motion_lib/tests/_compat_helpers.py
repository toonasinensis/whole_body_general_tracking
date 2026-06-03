from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np


def build_cfg(
    motion_file: str,
    *,
    smpl_file_path: str | None = None,
    max_motion_num: int = -1,
    dataset_txt: str | None = None,
    eval_mode: bool = False,
    distributed: bool = False,
    local_rank: int = 0,
    total_rank: int = 1,
    target_fps: float = 50.0,
    up_axis: str = "yup",
    max_frame_diff: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        motion_file=motion_file,
        max_motion_num=max_motion_num,
        dataset_txt=dataset_txt,
        eval_mode=eval_mode,
        distributed=distributed,
        local_rank=local_rank,
        total_rank=total_rank,
        smpl_file_path=smpl_file_path,
        target_fps=target_fps,
        up_axis=up_axis,
        max_frame_diff=max_frame_diff,
    )


def write_robot_npz(
    path: str | Path,
    *,
    num_frames: int,
    joint_dim: int,
    body_count: int,
    fps: float = 50.0,
    frame_offset: int = 0,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
) -> None:
    path = Path(path)
    joint_pos = (
        np.arange(frame_offset, frame_offset + num_frames * joint_dim, dtype=np.float32).reshape(num_frames, joint_dim)
    )
    joint_vel = 1000.0 + joint_pos
    body_pos_w = (
        np.arange(
            frame_offset * 10,
            frame_offset * 10 + num_frames * body_count * 3,
            dtype=np.float32,
        ).reshape(num_frames, body_count, 3)
    )
    body_quat_w = np.zeros((num_frames, body_count, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    body_lin_vel_w = body_pos_w + 2000.0
    body_ang_vel_w = body_pos_w + 3000.0

    kwargs: dict[str, np.ndarray] = {}
    if joint_names is not None:
        kwargs["joint_names"] = np.asarray(joint_names)
    if body_names is not None:
        kwargs["body_names"] = np.asarray(body_names)

    np.savez(
        path,
        fps=np.asarray([fps], dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        **kwargs,
    )


def write_smpl_pkl(
    path: str | Path,
    *,
    num_frames: int,
    fps: float = 50.0,
    frame_offset: int = 0,
) -> None:
    path = Path(path)
    pose_aa = np.zeros((num_frames, 72), dtype=np.float32)
    root_pose = np.asarray(
        [
            [0.10, -0.20, 0.05],
            [0.15, -0.10, 0.07],
            [0.20, -0.05, 0.09],
            [0.25, 0.00, 0.11],
            [0.30, 0.05, 0.13],
            [0.35, 0.10, 0.15],
        ],
        dtype=np.float32,
    )
    pose_aa[:, :3] = root_pose[frame_offset : frame_offset + num_frames]
    pose_aa[:, 3:] = (
        0.001
        * np.arange(
            frame_offset * 69,
            frame_offset * 69 + num_frames * 69,
            dtype=np.float32,
        ).reshape(num_frames, 69)
    )

    smpl_joints = (
        0.01
        * np.arange(
            frame_offset * 24 * 3,
            frame_offset * 24 * 3 + num_frames * 24 * 3,
            dtype=np.float32,
        ).reshape(num_frames, 24, 3)
    )
    transl = np.asarray(
        [[0.5 + i, -1.0 + 0.25 * i, 2.0 - 0.1 * i] for i in range(frame_offset, frame_offset + num_frames)],
        dtype=np.float32,
    )

    joblib.dump(
        {
            "pose_aa": pose_aa,
            "smpl_joints": smpl_joints,
            "transl": transl,
            "fps": fps,
        },
        path,
    )
