from __future__ import annotations

import torch

from feat_motion_lib.errors import MotionValidationError
from feat_motion_lib.store.paired_store import PairedMotionStore
from feat_motion_lib.store.robot_store import RobotMotionStore
from feat_motion_lib.store.smpl_store import SmplMotionStore
from feat_motion_lib.types import SmplMotionClip, RobotMotionClip


def _robot_clip(num_frames: int) -> RobotMotionClip:
    return RobotMotionClip(
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
    )


def _smpl_clip(num_frames: int) -> SmplMotionClip:
    return SmplMotionClip(
        pose_aa=torch.zeros((num_frames, 72), dtype=torch.float32),
        smpl_joints=torch.zeros((num_frames, 24, 3), dtype=torch.float32),
        transl=torch.zeros((num_frames, 3), dtype=torch.float32),
        fps=50.0,
        source_fps=50.0,
        num_frames=num_frames,
        duration=(num_frames - 1) / 50.0 if num_frames > 1 else 0.0,
    )


def test_paired_store_validates_shared_index_success() -> None:
    robot_store = RobotMotionStore([_robot_clip(3), _robot_clip(2)])
    smpl_store = SmplMotionStore([_smpl_clip(3), _smpl_clip(2)])
    paired_store = PairedMotionStore(robot_store, smpl_store)
    assert paired_store.robot_store.index.total_frames == 5


def test_paired_store_raises_on_frame_count_mismatch() -> None:
    robot_store = RobotMotionStore([_robot_clip(3), _robot_clip(2)])
    smpl_store = SmplMotionStore([_smpl_clip(3), _smpl_clip(4)])
    try:
        PairedMotionStore(robot_store, smpl_store)
    except MotionValidationError as exc:
        assert "frame_counts mismatch" in str(exc)
    else:
        raise AssertionError("Expected MotionValidationError")
