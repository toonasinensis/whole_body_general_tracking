from __future__ import annotations

import numpy as np
import torch

try:
    from whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils import (
        interpolate_linear,
        interpolate_pose,
        normalize_quaternion,
        slerp,
    )
except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback without importing whole_body_tracking package
    from smpl_math_utils import interpolate_linear, interpolate_pose, normalize_quaternion, slerp

from ..types import MotionData, RobotMotionData


def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def _interpolate_angular(quat: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Slerp-resample a wxyz quaternion tensor along its time axis."""
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion tensor must have last dimension 4, got {quat.shape}.")
    
    num_frames = quat.shape[0]
    duration = (num_frames - 1) / source_fps
    
    # 生成目标时间点
    num_target = int(duration * target_fps) + 1
    times = torch.linspace(0, duration, num_target, dtype=torch.float32, device=quat.device)
    
    # 计算插值索引和混合因子
    frame_indices = times * source_fps
    idx0 = torch.floor(frame_indices).long().clamp(0, num_frames - 2)  # 保证 idx1 有效
    idx1 = idx0 + 1
    blend = frame_indices - idx0.float()
    
    # 归一化并重塑
    quat_norm = normalize_quaternion(quat)
    original_shape = quat.shape
    quat_flat = quat_norm.reshape(num_frames, -1, 4)
    item_count = quat_flat.shape[1]
    
    # 使用广播避免 repeat_interleave
    q0 = quat_flat[idx0]  # [num_target, item_count, 4]
    q1 = quat_flat[idx1]  # [num_target, item_count, 4]
    blend_expanded = blend.unsqueeze(-1).unsqueeze(-1)  # [num_target, 1, 1]
    
    # Slerp 插值
    result = slerp(q0.reshape(-1, 4), q1.reshape(-1, 4), blend_expanded.expand(-1, item_count, -1).reshape(-1))
    result = normalize_quaternion(result)
    
    # 恢复原始形状（除了时间维度）
    return result.reshape(num_target, *original_shape[1:-1], 4)


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
    body_quat_w = _interpolate_angular(clip.body_quat_w, clip.fps, target_fps)
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
