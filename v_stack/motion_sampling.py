from __future__ import annotations

import torch


def validate_motion_local_frame_bounds(
    time_step_start_idx: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    min_local_frame: int,
    max_future_step: int,
) -> tuple[int, torch.Tensor]:
    """Return valid local frame bounds for every motion."""
    motion_lengths = time_step_end_idx - time_step_start_idx
    min_local = int(max(0, min_local_frame))
    max_local = motion_lengths - int(max_future_step) - 1
    invalid = max_local < min_local
    if torch.any(invalid):
        bad_motion_ids = torch.where(invalid)[0].detach().cpu().tolist()
        raise ValueError(
            "Motion sampling start frame/future window leaves no valid frame for motions "
            f"{bad_motion_ids}. motion_sampling_start_frame={min_local}, "
            f"max_future_step={int(max_future_step)}. "
            "Use longer motions or lower motion_sampling_start_frame/max_future_step."
        )
    return min_local, max_local


def compute_motion_sample_bin_counts(
    time_step_start_idx: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    min_local_frame: int,
    max_future_step: int,
    bin_frame_width: int,
) -> torch.Tensor:
    """Compute how many local sampling bins each motion owns."""
    min_local, max_local = validate_motion_local_frame_bounds(
        time_step_start_idx=time_step_start_idx,
        time_step_end_idx=time_step_end_idx,
        min_local_frame=min_local_frame,
        max_future_step=max_future_step,
    )
    bin_width = max(1, int(bin_frame_width))
    min_bin = min_local // bin_width
    return (max_local // bin_width - min_bin + 1).long()


def accumulate_rewinded_motion_sample_bin_counts(
    fail_motion_ids: torch.Tensor,
    fail_local_frames: torch.Tensor,
    time_step_start_idx: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    min_local_frame: int,
    max_future_step: int,
    bin_frame_width: int,
    rewind_bins: int,
    min_rewind_bins: int = 1,
    max_sample_bin_count: int | None = None,
) -> torch.Tensor:
    """Accumulate spawn-bin weights from terminate frames in motion-local coordinates.

    fail bin 仍然代表“失败发生点”；这里生成的是“重生候选点”。重生候选点
    始终在同一个 motion 内，并落在 terminate bin 前 min_rewind_bins..rewind_bins
    个 local bins 中。也就是 terminate bin 为 m 时，采样窗口为
    [m - rewind_bins, m - min_rewind_bins]。这样既能避开失败瞬间，也不会因为
    全局时间倒减而跨到别的动作。
    """
    device = time_step_start_idx.device
    fail_motion_ids = fail_motion_ids.to(device=device, dtype=torch.long)
    fail_local_frames = fail_local_frames.to(device=device, dtype=torch.long)
    bin_width = max(1, int(bin_frame_width))
    rewind_count = max(0, int(rewind_bins))
    min_rewind_count = 0 if rewind_count == 0 else max(1, int(min_rewind_bins))
    if min_rewind_count > rewind_count:
        raise ValueError(
            "adaptive_sample_rewind_min_bins must be <= adaptive_sample_rewind_bins. "
            f"Got min_rewind_bins={min_rewind_count}, rewind_bins={rewind_count}."
        )
    motion_bin_counts = compute_motion_sample_bin_counts(
        time_step_start_idx=time_step_start_idx,
        time_step_end_idx=time_step_end_idx,
        min_local_frame=min_local_frame,
        max_future_step=max_future_step,
        bin_frame_width=bin_width,
    )
    if max_sample_bin_count is None:
        max_sample_bin_count = int(motion_bin_counts.max().item())
    max_sample_bin_count = int(max(1, max_sample_bin_count))
    counts = torch.zeros(
        (int(time_step_start_idx.numel()), max_sample_bin_count),
        dtype=torch.float32,
        device=device,
    )
    if fail_motion_ids.numel() == 0:
        return counts

    motion_lengths = time_step_end_idx - time_step_start_idx
    fail_local_frames = torch.clamp(fail_local_frames, min=0)
    fail_local_frames = torch.minimum(fail_local_frames, motion_lengths[fail_motion_ids] - 1)
    fail_local_bins = fail_local_frames // bin_width

    min_local = int(max(0, min_local_frame))
    min_sample_bin = min_local // bin_width
    max_sample_bins = min_sample_bin + motion_bin_counts[fail_motion_ids] - 1
    flat_counts = counts.view(-1)

    offsets = tuple(range(min_rewind_count, rewind_count + 1))
    weight = 1.0 / float(len(offsets))
    weight_values = torch.full((fail_motion_ids.numel(),), weight, dtype=counts.dtype, device=device)
    for offset in offsets:
        target_bins = fail_local_bins - int(offset)
        target_bins = torch.clamp(target_bins, min=min_sample_bin)
        target_bins = torch.minimum(target_bins, max_sample_bins)
        local_storage_bins = target_bins - min_sample_bin
        flat_indexes = fail_motion_ids * max_sample_bin_count + local_storage_bins
        flat_counts.scatter_add_(0, flat_indexes, weight_values)
    return counts


def sample_motion_local_times_from_bins(
    motion_ids: torch.Tensor,
    local_bin_ids: torch.Tensor,
    time_step_start_idx: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    min_local_frame: int,
    max_future_step: int,
    bin_frame_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local frame indices inside already-selected motion-local bins."""
    device = time_step_start_idx.device
    motion_ids = motion_ids.to(device=device, dtype=torch.long)
    local_bin_ids = local_bin_ids.to(device=device, dtype=torch.long)
    min_local, max_local = validate_motion_local_frame_bounds(
        time_step_start_idx=time_step_start_idx,
        time_step_end_idx=time_step_end_idx,
        min_local_frame=min_local_frame,
        max_future_step=max_future_step,
    )
    bin_width = max(1, int(bin_frame_width))
    min_bin_id = min_local // bin_width
    local_bin_ids = local_bin_ids + min_bin_id
    max_bin_ids = (max_local[motion_ids] // bin_width).long()
    local_bin_ids = torch.minimum(local_bin_ids, max_bin_ids)

    lower = local_bin_ids * bin_width
    lower = torch.clamp(lower, min=min_local)
    upper = torch.minimum((local_bin_ids + 1) * bin_width, max_local[motion_ids] + 1)
    span = torch.clamp(upper - lower, min=1)
    local_t = lower + (torch.rand(span.shape, device=device) * span.float()).long()
    return motion_ids, local_t


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
    min_local, max_local_all = validate_motion_local_frame_bounds(
        time_step_start_idx=time_step_start_idx,
        time_step_end_idx=time_step_end_idx,
        min_local_frame=min_local_frame,
        max_future_step=max_future_step,
    )
    max_local = max_local_all[motion_ids]

    local_t = timestamps - start - int(max(0, rewind_frames))
    local_t = torch.clamp(local_t, min=min_local)
    local_t = torch.minimum(local_t, max_local)
    return motion_ids, local_t
