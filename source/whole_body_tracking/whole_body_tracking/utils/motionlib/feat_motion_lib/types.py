from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class MotionData:
    pose_aa: torch.Tensor
    smpl_joints: torch.Tensor
    transl: torch.Tensor
    fps: float
    source_fps: float
    num_frames: int
    duration: float


@dataclass
class RobotMotionData:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    fps: float
    source_fps: float
    num_frames: int
    duration: float
    joint_names: list[str] | None = None
    body_names: list[str] | None = None
    path: Path | None = None


@dataclass
class PairedPath:
    stem: str
    robot_path: Path
    smpl_path: Path


@dataclass
class PairedMotionData:
    stem: str
    robot: RobotMotionData
    smpl: MotionData


@dataclass
class MotionIndex:
    frame_counts: torch.Tensor
    start_idx: torch.Tensor
    end_idx: torch.Tensor
    total_frames: int


@dataclass
class UnifiedMotionState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    fps: float
    time_step_total: int
    file_names: list[str]
    motion_num: int
    frame_list: torch.Tensor
    time_step_start_idx: torch.Tensor
    time_step_end_idx: torch.Tensor


@dataclass
class LoadReport:
    loaded_files: list[str]
    skipped_files: list[str]
    warnings: list[str]
    fallback_used: bool = False
    mode: str | None = None
    requested_pairs: int = 0
    loaded_pairs: int = 0


@dataclass
class RobotLoadResult:
    clips: list[RobotMotionData]
    report: LoadReport


@dataclass
class PairedLoadResult:
    clips: list[PairedMotionData]
    report: LoadReport
