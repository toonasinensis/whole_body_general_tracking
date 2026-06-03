from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from feat_motion_lib.transform.pairing import pair_motion_paths, trim_paired_clips
from feat_motion_lib.types import MotionData, RobotMotionData


def _robot_clip(num_frames: int, path: str = "robot.npz") -> RobotMotionData:
    return RobotMotionData(
        joint_pos=torch.zeros((num_frames, 2), dtype=torch.float32),
        joint_vel=torch.zeros((num_frames, 2), dtype=torch.float32),
        body_pos_w=torch.zeros((num_frames, 3, 3), dtype=torch.float32),
        body_quat_w=torch.zeros((num_frames, 3, 4), dtype=torch.float32),
        body_lin_vel_w=torch.zeros((num_frames, 3, 3), dtype=torch.float32),
        body_ang_vel_w=torch.zeros((num_frames, 3, 3), dtype=torch.float32),
        fps=50.0,
        source_fps=50.0,
        num_frames=num_frames,
        duration=(num_frames - 1) / 50.0 if num_frames > 1 else 0.0,
        path=Path(path),
    )


def _smpl_clip(num_frames: int) -> MotionData:
    return MotionData(
        pose_aa=torch.zeros((num_frames, 72), dtype=torch.float32),
        smpl_joints=torch.zeros((num_frames, 24, 3), dtype=torch.float32),
        transl=torch.zeros((num_frames, 3), dtype=torch.float32),
        fps=50.0,
        source_fps=50.0,
        num_frames=num_frames,
        duration=(num_frames - 1) / 50.0 if num_frames > 1 else 0.0,
    )


def test_pair_motion_paths_matches_on_stem(tmp_path) -> None:
    robot_a = tmp_path / "a.npz"
    robot_b = tmp_path / "b.npz"
    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()
    (smpl_dir / "b.pkl").touch()
    (smpl_dir / "a.pkl").touch()
    robot_a.touch()
    robot_b.touch()

    pairs = pair_motion_paths([robot_b, robot_a], smpl_dir)
    assert [pair.stem for pair in pairs] == ["a", "b"]
    assert pairs[0].robot_path.name == "a.npz"
    assert pairs[0].smpl_path.name == "a.pkl"


def test_trim_paired_clips_returns_none_when_frame_diff_too_large() -> None:
    trimmed = trim_paired_clips("sample", _robot_clip(10), _smpl_clip(13), max_frame_diff=2)
    assert trimmed is None


def test_trim_paired_clips_trims_to_shared_min_frame_count() -> None:
    trimmed = trim_paired_clips("sample", _robot_clip(10), _smpl_clip(11), max_frame_diff=2)
    assert trimmed is not None
    assert trimmed.robot.num_frames == 10
    assert trimmed.smpl.num_frames == 10
    assert trimmed.robot.joint_pos.shape[0] == 10
    assert trimmed.smpl.pose_aa.shape[0] == 10
