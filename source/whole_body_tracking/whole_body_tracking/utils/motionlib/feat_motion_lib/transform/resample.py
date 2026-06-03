from __future__ import annotations

import numpy as np
import torch

try:
    from whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils import (
        interpolate_linear,
        interpolate_pose,
    )
except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback without importing whole_body_tracking package
    from smpl_math_utils import interpolate_linear, interpolate_pose

from ..types import MotionData, RobotMotionData


def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def resample_motion(raw: dict, target_fps: float) -> MotionData:
    src_fps = float(raw["fps"])

    pose_t = _to_tensor(raw["pose_aa"])
    joints_t = _to_tensor(raw["smpl_joints"])
    transl_t = _to_tensor(raw["transl"])

    pose_r = interpolate_pose(pose_t, src_fps, target_fps, interpolation_type="slerp")
    joints_r = interpolate_linear(joints_t, src_fps, target_fps)
    transl_r = interpolate_linear(transl_t, src_fps, target_fps)

    num_frames = pose_r.shape[0]
    return MotionData(
        pose_aa=pose_r,
        smpl_joints=joints_r,
        transl=transl_r,
        fps=target_fps,
        source_fps=src_fps,
        num_frames=num_frames,
        duration=(num_frames - 1) / target_fps if num_frames > 1 else 0.0,
    )


def resample_robot_motion(clip: RobotMotionData, target_fps: float) -> RobotMotionData:
    joint_pos = interpolate_linear(clip.joint_pos, clip.fps, target_fps)
    joint_vel = interpolate_linear(clip.joint_vel, clip.fps, target_fps)
    body_pos_w = interpolate_linear(clip.body_pos_w, clip.fps, target_fps)
    body_quat_w = interpolate_linear(clip.body_quat_w, clip.fps, target_fps)
    body_lin_vel_w = interpolate_linear(clip.body_lin_vel_w, clip.fps, target_fps)
    body_ang_vel_w = interpolate_linear(clip.body_ang_vel_w, clip.fps, target_fps)
    num_frames = int(joint_pos.shape[0])
    return RobotMotionData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        fps=target_fps,
        source_fps=clip.source_fps,
        num_frames=num_frames,
        duration=(num_frames - 1) / target_fps if num_frames > 1 else 0.0,
        joint_names=clip.joint_names,
        body_names=clip.body_names,
        path=clip.path,
    )
