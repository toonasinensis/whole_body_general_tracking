from __future__ import annotations

import os

import torch

from ..types import RobotMotionData, UnifiedMotionState
from .index import build_motion_index


class RobotMotionStore:
    def __init__(self, clips: list[RobotMotionData], device: str = "cpu") -> None:
        if not clips:
            raise ValueError("clips list is empty")

        self.clips = list(clips)
        self.device = device
        # TODO(kiki): If motion datasets grow large, switch these whole-tensor `.to(device)`
        # transfers to a batched CPU->GPU path like the legacy MotionLoader to reduce load-time
        # peak VRAM usage.
        self.joint_pos = torch.cat([clip.joint_pos for clip in clips], dim=0).to(device)
        self.joint_vel = torch.cat([clip.joint_vel for clip in clips], dim=0).to(device)
        self.body_pos_w = torch.cat([clip.body_pos_w for clip in clips], dim=0).to(device)
        self.body_quat_w = torch.cat([clip.body_quat_w for clip in clips], dim=0).to(device)
        self.body_lin_vel_w = torch.cat([clip.body_lin_vel_w for clip in clips], dim=0).to(device)
        self.body_ang_vel_w = torch.cat([clip.body_ang_vel_w for clip in clips], dim=0).to(device)
        self.file_names = [clip.path.name if clip.path is not None else f"motion_{i}" for i, clip in enumerate(clips)]
        self.fps = float(clips[0].fps)
        # index 类实例，存储连接的 motion tensor 的基础分段信息，如每段的帧数、起止索引等
        self.index = build_motion_index([clip.num_frames for clip in clips], device=device)
        self.motion_num = int(self.index.frame_counts.numel())
        self.frame_list = self.index.frame_counts
        self.time_step_total = int(self.index.total_frames)
        self.time_step_start_idx = self.index.start_idx
        self.time_step_end_idx = self.index.end_idx

    def relative_file_names(self, base_dir: str) -> list[str]:
        return [
            os.path.relpath(str(clip.path), base_dir) if clip.path is not None else f"motion_{i}"
            for i, clip in enumerate(self.clips)
        ]

    def build_state(self, base_dir: str) -> UnifiedMotionState:
        # UnifiedMotionState 是标准的 robot motion 数据结构
        # 是 RobotMotionData list 的统一化封装
        return UnifiedMotionState(
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
            body_pos_w=self.body_pos_w,
            body_quat_w=self.body_quat_w,
            body_lin_vel_w=self.body_lin_vel_w,
            body_ang_vel_w=self.body_ang_vel_w,
            fps=self.fps,
            time_step_total=self.time_step_total,
            file_names=self.relative_file_names(base_dir),
            motion_num=self.motion_num,
            frame_list=self.frame_list,
            time_step_start_idx=self.time_step_start_idx,
            time_step_end_idx=self.time_step_end_idx,
        )

# checked by kiki on 2024-06-10
