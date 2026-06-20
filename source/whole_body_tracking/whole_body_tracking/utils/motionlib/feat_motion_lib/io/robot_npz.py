from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..errors import MotionFileError, MotionValidationError
from ..transform import align_body_tensors, align_joint_tensors, resample_robot_motion
from ..types import RobotMotionClip

_ROBOT_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

def read_npz_fps(raw: np.lib.npyio.NpzFile, path: str) -> float:
    fps = np.asarray(raw["fps"], dtype=np.float32).reshape(-1)
    if fps.size == 0:
        raise MotionValidationError(f"{path}: empty fps field")
    return float(fps[0])


def load_robot_npz_raw(path: str) -> tuple[np.lib.npyio.NpzFile, float]:
    try:
        raw = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise MotionFileError(f"{path}: failed to read npz file: {exc}") from exc
    return raw, read_npz_fps(raw, path)


def parse_robot_npz(
    raw: np.lib.npyio.NpzFile,
    path: str,
    source_fps: float,
    joint_names: Sequence[str] | None = None,
    motion_body_names: Sequence[str] | None = None,
    all_body_names: Sequence[str] | None = None,
    body_indexes: Sequence[int] | None = None,
) -> RobotMotionClip:
    try:
        tensors = {key: torch.from_numpy(np.asarray(raw[key], dtype=np.float32)) for key in _ROBOT_KEYS}
    except KeyError as exc:
        raise MotionValidationError(f"{path}: missing required robot tensor key {exc.args[0]!r}.") from exc

    # 关节和身体数据对齐检测，若 npz 文件中没有包含关节的名称信息，
    # 则默认关节和身体数据的维度顺序与 joint_names 和 motion_body_names 中的顺序一致
    tensors, loaded_joint_names = align_joint_tensors(tensors, raw, path, joint_names)
    tensors, loaded_body_names = align_body_tensors(
        tensors,
        raw,
        path,
        motion_body_names,
        all_body_names,
        body_indexes,
    )

    num_frames = int(tensors["joint_pos"].shape[0])
    if num_frames <= 0:
        raise MotionValidationError(f"{path}: empty motion frames")

    return RobotMotionClip(
        joint_pos=tensors["joint_pos"],
        joint_vel=tensors["joint_vel"],
        body_pos_w=tensors["body_pos_w"],
        body_quat_w=tensors["body_quat_w"],
        body_lin_vel_w=tensors["body_lin_vel_w"],
        body_ang_vel_w=tensors["body_ang_vel_w"],
        fps=source_fps,
        source_fps=source_fps,
        num_frames=num_frames,
        duration=(num_frames - 1) / source_fps if num_frames > 1 else 0.0,
        joint_names=loaded_joint_names,
        body_names=loaded_body_names,
        path=Path(path),
    )


def load_robot_motion_file(
    path: str,
    target_fps: float | None = None,
    joint_names: Sequence[str] | None = None,
    motion_body_names: Sequence[str] | None = None,
    all_body_names: Sequence[str] | None = None,
    body_indexes: Sequence[int] | None = None,
) -> RobotMotionClip:
    """
        从指定路径加载机器人运动数据，返回 RobotMotionClip 对象
    """
    raw, source_fps = load_robot_npz_raw(path)
    clip = parse_robot_npz(
        raw,
        path,
        source_fps,
        joint_names=joint_names,
        motion_body_names=motion_body_names,
        all_body_names=all_body_names,
        body_indexes=body_indexes,
    )
    
    # 如果指定了 target_fps 且与 source_fps 不同，则进行重采样
    if target_fps is not None and abs(source_fps - target_fps) > 1e-3:
        return resample_robot_motion(clip, target_fps)
    
    return clip
