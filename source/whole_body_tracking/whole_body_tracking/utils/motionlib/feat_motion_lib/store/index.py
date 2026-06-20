from __future__ import annotations

import torch

from ..types import MotionIndex


def build_length_starts(frame_counts: torch.Tensor) -> torch.Tensor:
    """
    根据每段 motion 的帧数，计算每段 motion 在 flat tensor 中的起始帧索引。
    """
    previous_frame_counts = torch.roll(frame_counts, 1)
    previous_frame_counts[0] = 0
    return torch.cumsum(previous_frame_counts, dim=0)


def build_motion_index(
    frame_counts: list[int] | torch.Tensor, 
    device: str = "cpu"
) -> MotionIndex:
    """
    根据每段 motion 的帧数构建 flat tensor 的 per-motion 索引。
    """
    if not isinstance(frame_counts, torch.Tensor):
        frame_counts_tensor = torch.tensor(frame_counts, dtype=torch.long, device=device)
    else:
        frame_counts_tensor = frame_counts.to(device=device, dtype=torch.long)

    end_idx = torch.cumsum(frame_counts_tensor, dim=0)
    start_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=device), end_idx[:-1]])
    total_frames = int(frame_counts_tensor.sum().item())
    return MotionIndex(
        frame_counts=frame_counts_tensor,
        start_idx=start_idx,
        end_idx=end_idx,
        total_frames=total_frames,
    )


def flatten_indices(
    start_idx: torch.Tensor,
    motion_ids: torch.Tensor,
    motion_steps: torch.Tensor,
) -> torch.Tensor:
    """
    将 motion id 和 motion-local frame index 转换为 flat tensor 的全局帧索引。

    Args:
        start_idx: 每段 motion 在 flat tensor 中的起始帧索引。
        motion_ids: motion id。
        motion_steps: motion 内的 local frame index。
    """
    start_idx_on_device = start_idx.to(motion_ids.device)
    return motion_steps + start_idx_on_device[motion_ids]


def motion_ids_from_timestamps(
    timestamps: torch.Tensor,
    end_idx: torch.Tensor,
    total_frames: int,
    motion_num: int,
) -> torch.Tensor:
    """
    timestamps -> motion_ids

    从全局时间戳计算对应的 motion_id
        timestamps 是全局时间戳，单位为帧
        end_idx 是每段 motion 在连接的 tensor 中的结束帧索引
        total_frames 是连接的 motion tensor 的总帧数
        motion_num 是 motion 的总数
    
    return:
       motion_ids: torch tensor  
    """
    if timestamps.dtype != torch.long:
        global_frame_idx = timestamps.long()
    else:
        global_frame_idx = timestamps

    global_frame_idx = torch.clamp(global_frame_idx, min=0, max=total_frames - 1)
    motion_ids = torch.bucketize(global_frame_idx, end_idx, right=True)
    return torch.clamp(motion_ids, min=0, max=motion_num - 1)
