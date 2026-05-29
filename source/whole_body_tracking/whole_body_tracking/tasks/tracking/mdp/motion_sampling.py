from __future__ import annotations

import torch


def sample_rewinded_motion_local_times(
    timestamps: torch.Tensor,
    time_step_start_idx: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    motion_ids: torch.Tensor,
    time_step_total: int,
    min_local_frame: int,
    max_future_step: int,
    rewind_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert sampled global frames to safe motion-local start frames.

    失败 bin 表示“失败发生点”，不是“重生点”。如果直接从失败 bin 重生，
    机器人可能已经在空中或姿态很差；这里先定位到同一个 motion，再在 local
    frame 上往前回退，给策略一个恢复窗口。clamp 也必须在 motion 内完成，
    不能直接全局减帧数，否则靠近 motion 边界时会串到前一个动作。
    """
    if timestamps.dtype != torch.long:
        timestamps = timestamps.long()
    timestamps = torch.clamp(timestamps, min=0, max=int(time_step_total) - 1)

    start = time_step_start_idx[motion_ids]
    end = time_step_end_idx[motion_ids]
    motion_lengths = end - start
    min_local = int(max(0, min_local_frame))
    max_local = motion_lengths - int(max_future_step) - 1
    invalid = max_local < min_local
    if torch.any(invalid):
        bad_motion_ids = torch.unique(motion_ids[invalid]).detach().cpu().tolist()
        raise ValueError(
            "Motion sampling start frame/future window leaves no valid frame for motions "
            f"{bad_motion_ids}. motion_sampling_start_frame={min_local}, "
            f"max_future_step={int(max_future_step)}. "
            "Use longer motions or lower motion_sampling_start_frame/max_future_step."
        )

    local_t = timestamps - start - int(max(0, rewind_frames))
    local_t = torch.clamp(local_t, min=min_local)
    local_t = torch.minimum(local_t, max_local)
    return motion_ids, local_t
