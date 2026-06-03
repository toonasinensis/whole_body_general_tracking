from __future__ import annotations

import torch

from ..types import MotionIndex


def build_length_starts(frame_counts: torch.Tensor) -> torch.Tensor:
    shifted = torch.roll(frame_counts, 1)
    shifted[0] = 0
    return torch.cumsum(shifted, dim=0)


# checked
def build_motion_index(frame_counts: list[int] | torch.Tensor, device: str = "cpu") -> MotionIndex:
    if not isinstance(frame_counts, torch.Tensor):
        frame_counts = torch.tensor(frame_counts, dtype=torch.long, device=device)
    else:
        frame_counts = frame_counts.to(device=device, dtype=torch.long)
    end_idx = torch.cumsum(frame_counts, dim=0)
    start_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device), end_idx[:-1]])
    return MotionIndex(
        frame_counts=frame_counts,
        start_idx=start_idx,
        end_idx=end_idx,
        total_frames=int(frame_counts.sum().item()),
    )


# checked
def flatten_indices(
    length_starts: torch.Tensor,
    motion_ids: torch.Tensor,
    motion_steps: torch.Tensor,
) -> torch.Tensor:
    """
    将 motion_ids 和 motion_steps 转换为在连接的 motion tensor 中的全局帧索引
        length_starts 是每段 motion 在连接的 tensor 中的起始帧索引
        motion_ids 是 motion 的 ID
        motion_steps 是 motion 内的帧步数
    """
    starts = length_starts.to(motion_ids.device)
    return motion_steps + starts[motion_ids]


# checked
def motion_ids_from_timestamps(
    timestamps: torch.Tensor,
    end_idx: torch.Tensor,
    total_frames: int,
    motion_num: int,
) -> torch.Tensor:
    """
    从全局时间戳计算对应的 motion_id
        timestamps 是全局时间戳，单位为帧
        end_idx 是每段 motion 在连接的 tensor 中的结束帧索引
        total_frames 是连接的 motion tensor 的总帧数
        motion_num 是 motion 的总数
    """
    if timestamps.dtype != torch.long:
        timestamps = timestamps.long()
    timestamps = torch.clamp(timestamps, min=0, max=total_frames - 1)
    motion_ids = torch.bucketize(timestamps, end_idx, right=True)
    return torch.clamp(motion_ids, min=0, max=motion_num - 1)
