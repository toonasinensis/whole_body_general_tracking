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
accumulate_rewinded_motion_sample_bin_counts = motion_sampling.accumulate_rewinded_motion_sample_bin_counts
sample_motion_local_times_from_bins = motion_sampling.sample_motion_local_times_from_bins


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


def test_failure_spawn_bins_stay_on_terminated_motion() -> None:
    starts = torch.tensor([0, 300])
    ends = torch.tensor([300, 600])

    counts = accumulate_rewinded_motion_sample_bin_counts(
        fail_motion_ids=torch.tensor([1]),
        fail_local_frames=torch.tensor([200]),
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        min_local_frame=5,
        max_future_step=10,
        bin_frame_width=50,
        rewind_bins=2,
    )

    assert counts[0].sum().item() == 0.0
    assert counts[1, 2].item() == pytest.approx(0.5)
    assert counts[1, 3].item() == pytest.approx(0.5)
    assert counts[1].sum().item() == pytest.approx(1.0)


def test_rewind_bins_mean_previous_one_through_n_bins() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([300])

    counts = accumulate_rewinded_motion_sample_bin_counts(
        fail_motion_ids=torch.tensor([0]),
        fail_local_frames=torch.tensor([175]),
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        min_local_frame=5,
        max_future_step=10,
        bin_frame_width=50,
        rewind_bins=2,
    )

    # terminate local frame 175 is in bin 3, so rewind_bins=2 samples from bins 2 and 1.
    assert counts[0, 1].item() == pytest.approx(0.5)
    assert counts[0, 2].item() == pytest.approx(0.5)
    assert counts[0, 3].item() == 0.0


def test_rewind_min_bins_skips_bins_close_to_termination() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([500])

    counts = accumulate_rewinded_motion_sample_bin_counts(
        fail_motion_ids=torch.tensor([0]),
        fail_local_frames=torch.tensor([275]),
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        min_local_frame=5,
        max_future_step=10,
        bin_frame_width=50,
        rewind_bins=5,
        min_rewind_bins=2,
    )

    # terminate local frame 275 is in bin 5; min=2,max=5 samples bins m-2..m-5 -> 3,2,1,0.
    assert counts[0, 0].item() == pytest.approx(0.25)
    assert counts[0, 1].item() == pytest.approx(0.25)
    assert counts[0, 2].item() == pytest.approx(0.25)
    assert counts[0, 3].item() == pytest.approx(0.25)
    assert counts[0, 4].item() == 0.0
    assert counts[0, 5].item() == 0.0


def test_rewind_min_bins_cannot_exceed_max_bins() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([300])

    with pytest.raises(ValueError, match="adaptive_sample_rewind_min_bins"):
        accumulate_rewinded_motion_sample_bin_counts(
            fail_motion_ids=torch.tensor([0]),
            fail_local_frames=torch.tensor([175]),
            time_step_start_idx=starts,
            time_step_end_idx=ends,
            min_local_frame=5,
            max_future_step=10,
            bin_frame_width=50,
            rewind_bins=2,
            min_rewind_bins=3,
        )


def test_sampling_from_storage_bin_keeps_min_start_frame() -> None:
    starts = torch.tensor([0])
    ends = torch.tensor([300])
    torch.manual_seed(1)

    motion_ids, local_t = sample_motion_local_times_from_bins(
        motion_ids=torch.tensor([0] * 128),
        local_bin_ids=torch.tensor([0] * 128),
        time_step_start_idx=starts,
        time_step_end_idx=ends,
        min_local_frame=5,
        max_future_step=10,
        bin_frame_width=50,
    )

    assert motion_ids.unique().tolist() == [0]
    assert int(local_t.min().item()) >= 5
    assert int(local_t.max().item()) < 50
