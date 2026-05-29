from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "tasks"
    / "tracking"
    / "mdp"
    / "motion_sampling.py"
)
SPEC = importlib.util.spec_from_file_location("motion_sampling", MODULE_PATH)
motion_sampling = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(motion_sampling)
sample_rewinded_motion_local_times = motion_sampling.sample_rewinded_motion_local_times


def _motion_ids_from_timestamps(timestamps: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(timestamps, ends, right=True)


def test_sampled_frame_rewinds_two_bins_inside_same_motion() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([300])
    timestamps = torch.tensor([200])
    motion_ids = _motion_ids_from_timestamps(timestamps, ends)

    resolved_motion_ids, local_t = sample_rewinded_motion_local_times(
        timestamps=timestamps,
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        motion_ids=motion_ids,
        time_step_total=300,
        min_local_frame=5,
        max_future_step=10,
        rewind_frames=100,
    )

    assert resolved_motion_ids.tolist() == [0]
    assert local_t.tolist() == [100]


def test_sampled_frame_near_start_clamps_to_min_without_crossing_motion() -> None:
    starts = torch.tensor([0, 300])
    ends = torch.tensor([300, 600])
    timestamps = torch.tensor([315])
    motion_ids = _motion_ids_from_timestamps(timestamps, ends)

    resolved_motion_ids, local_t = sample_rewinded_motion_local_times(
        timestamps=timestamps,
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        motion_ids=motion_ids,
        time_step_total=600,
        min_local_frame=5,
        max_future_step=10,
        rewind_frames=100,
    )

    assert resolved_motion_ids.tolist() == [1]
    assert local_t.tolist() == [5]


def test_sampled_frame_near_end_clamps_to_future_safe_upper_bound() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([300])
    timestamps = torch.tensor([299])
    motion_ids = _motion_ids_from_timestamps(timestamps, ends)

    _, local_t = sample_rewinded_motion_local_times(
        timestamps=timestamps,
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        motion_ids=motion_ids,
        time_step_total=300,
        min_local_frame=5,
        max_future_step=10,
        rewind_frames=0,
    )

    assert local_t.tolist() == [289]


def test_short_motion_raises_instead_of_sampling_from_zero() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([12])
    timestamps = torch.tensor([6])
    motion_ids = _motion_ids_from_timestamps(timestamps, ends)

    with pytest.raises(ValueError, match="leaves no valid frame"):
        sample_rewinded_motion_local_times(
            timestamps=timestamps,
            time_step_start_idx=starts,
            time_step_end_idx=ends,
            motion_ids=motion_ids,
            time_step_total=12,
            min_local_frame=5,
            max_future_step=10,
            rewind_frames=100,
        )
