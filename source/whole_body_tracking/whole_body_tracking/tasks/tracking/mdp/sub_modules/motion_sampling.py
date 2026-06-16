from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from typing import Sequence

from .interface import MotionDataSource, MotionSelection
from .motion_timeline import MotionCommandTimeline

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul, sample_uniform, yaw_quat


#负责计算动作的选择方法（包含 adaptive sampling）
#   功能边界：
#   行为：

class AdaptiveMotionSampler:
    """Maintain adaptive-sampling state and produce motion selections."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        self.bin_count = 0  # num of 1s motion bins
        self.bin_failed_count = torch.zeros(0, dtype=torch.float, device=device)
        self._current_bin_failed = torch.zeros(0, dtype=torch.float, device=device)
        self.kernel = torch.zeros(0, dtype=torch.float, device=device)
        self.success_motion = torch.zeros(0, dtype=torch.float32, device=device)  # used for evaluation
        self._last_export_step = -1

    def reset_for_motion_source(self, motion_source: MotionDataSource, *, decimation: int, sim_dt: float) -> None:
        """called when load new motion """
        self.bin_count = int(motion_source.time_step_total // (1 / (decimation * sim_dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
        # used for evaluation
        motion_num = int(getattr(motion_source, "motion_num", 0) or 0)
        self.success_motion = torch.zeros(motion_num, dtype=torch.float32, device=self.device)
        self._last_export_step = -1

    def sample_selection(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        """sample new motions according to failure metrics, and produce a MotionSelection for the given env_ids."""
        if len(env_ids) == 0:
            raise ValueError("env_ids must not be empty for adaptive sampling")

        # Update failed motion bins to _current_bin_failed
        episode_failed = terminated[env_ids]
        if allow_failure_accounting and torch.any(episode_failed):
            global_ts = torch.clamp(timeline.global_time_steps(motion_source) - 1, min=0, max=int(motion_source.time_step_total) - 1)
            current_bin_index = torch.clamp(
                (global_ts * self.bin_count) // max(motion_source.time_step_total, 1),
                0,
                self.bin_count - 1,
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count)

        sampling_probabilities = self._compute_sampling_probabilities(metrics)
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        global_ts = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (motion_source.time_step_total - 1)
        ).long()

        selection = timeline.selection_from_global_timestamps(motion_source, global_ts)
        #TODO think whether there is more clear logic for different sampling modes
        if self.cfg.eval_mode:
            selection.local_time_steps.zero_()
        self._update_sampling_metrics(metrics, sampling_probabilities)
        return selection

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        """
            Called after command.step(). 
            Update current failed bins to bin_failed count.
            Export adaptive bins for analysis.  
        """
        if self.bin_failed_count.numel() == 0:
            return

        self.bin_failed_count[:] = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

        if (
            self.cfg.save_adaptive_bins
            and self.cfg.fail_count_save_interval > 0
            and command_step_count % self.cfg.fail_count_save_interval == 0
            and self._last_export_step != command_step_count
        ):
            self._export_adaptive_bins(motion_source=motion_source, command_step_count=command_step_count)

    def _compute_sampling_probabilities(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        """Hierarchical adaptive sampling"""
        metrics["failures_max"][:] = self.bin_failed_count.max()
        metrics["failures_mean"][:] = self.bin_failed_count.mean()
        metrics["failures_min"][:] = self.bin_failed_count.min()
        metrics["failures_max_over_uniform"][:] = (
            self.bin_failed_count.max() * self.bin_count / (self.bin_failed_count.sum() + 1e-8)
        )

        clipped_bin_failed_count = torch.clamp(
            self.bin_failed_count,
            max=self.cfg.failure_most_hard_cap_beta * self.bin_failed_count.mean(),
        )
        sampling_probabilities = torch.nn.functional.pad(
            clipped_bin_failed_count.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities,
            self.kernel.view(1, 1, -1),
        ).view(-1)
        sampling_probabilities = sampling_probabilities / (sampling_probabilities.sum() + 1e-12)

        sampling_probabilities_middle_hard = torch.clamp(
            sampling_probabilities,
            max=self.cfg.failure_cap_beta * sampling_probabilities.mean(),
        )
        sampling_probabilities_middle_hard = sampling_probabilities_middle_hard / (
            sampling_probabilities_middle_hard.sum() + 1e-12
        )

        sampling_probabilities_most_hard = torch.clamp(
            sampling_probabilities,
            max=self.cfg.failure_most_hard_cap_beta * sampling_probabilities.mean(),
        )
        sampling_probabilities_most_hard = sampling_probabilities_most_hard / (
            sampling_probabilities_most_hard.sum() + 1e-12
        )

        return (
            self.cfg.motion_ratio[0] * (1 / float(self.bin_count))
            + self.cfg.motion_ratio[1] * sampling_probabilities_middle_hard
            + self.cfg.motion_ratio[2] * sampling_probabilities_most_hard
        )

    def _update_sampling_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        sampling_probabilities: torch.Tensor,
    ) -> None:
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        metrics["sampling_entropy"][:] = H_norm
        metrics["sampling_top1_prob_max"][:] = pmax
        metrics["prob_max_over_uniform"][:] = pmax / (1 / self.bin_count)
        metrics["prob_uniform"][:] = 1 / self.bin_count
        metrics["sampling_top1_prob_bin"][:] = imax.float() / self.bin_count
        metrics["sampling_top1_prob_mean"][:] = sampling_probabilities.mean()
        metrics["sampling_top1_prob_min"][:] = sampling_probabilities.min()
        metrics["num_concentrate_bins"][:] = (
            sampling_probabilities > self.cfg.failure_most_hard_cap_beta * 0.5 * sampling_probabilities.mean()
        ).sum()

    #region export high prob bins for analysis 
    # TODO 这一部分过于冗长，后期需要改掉
    def _get_bin_global_frame_range(self, motion_source: MotionDataSource, bin_index: int) -> tuple[int, int]:
        total_frames = max(int(motion_source.time_step_total), 1)
        if total_frames == 1:
            return 0, 1

        scaled_total = total_frames - 1
        global_start = int(math.floor((bin_index * scaled_total) / float(self.bin_count)))
        global_end = int(math.ceil(((bin_index + 1) * scaled_total) / float(self.bin_count)))
        global_end = max(global_start + 1, min(global_end, total_frames))
        return global_start, global_end

    def _build_bin_motion_segments(
        self,
        motion_source: MotionDataSource,
        global_start: int,
        global_end: int,
    ) -> list[dict]:
        start_idx = motion_source.time_step_start_idx.detach().cpu().tolist()
        end_idx = motion_source.time_step_end_idx.detach().cpu().tolist()
        file_names = list(getattr(motion_source, "file_names", []))
        motion_num = int(getattr(motion_source, "motion_num", 0) or 0)
        if len(file_names) == 0:
            file_names = [f"motion_{i:05d}.npz" for i in range(motion_num)]

        segments: list[dict] = []
        for motion_id, (motion_start, motion_end) in enumerate(zip(start_idx, end_idx)):
            overlap_start = max(global_start, int(motion_start))
            overlap_end = min(global_end, int(motion_end))
            if overlap_start >= overlap_end:
                continue
            segments.append(
                {
                    "motion_id": int(motion_id),
                    "motion_file": file_names[motion_id],
                    "global_frame_start": int(overlap_start),
                    "global_frame_end_exclusive": int(overlap_end),
                    "motion_local_frame_start": int(overlap_start - motion_start),
                    "motion_local_frame_end_exclusive": int(overlap_end - motion_start),
                }
            )
            if motion_end >= global_end:
                break
        return segments

    def _export_adaptive_bins(self, *, motion_source: MotionDataSource, command_step_count: int) -> None:
        sampling_probabilities = self._compute_sampling_probabilities_for_export().detach().cpu()
        bin_failed_count = self.bin_failed_count.detach().cpu()
        fps = float(getattr(motion_source, "fps", 0.0))
        rank = int(self.cfg.local_rank) if int(self.cfg.local_rank) >= 0 else 0

        topk = min(100, int(self.bin_count))
        _, top_indices = torch.topk(bin_failed_count, k=topk, largest=True, sorted=True)
        top_indices = top_indices.tolist()

        bins = []
        for bin_index in top_indices:
            global_start, global_end = self._get_bin_global_frame_range(motion_source, bin_index)
            motion_segments = self._build_bin_motion_segments(motion_source, global_start, global_end)
            first_segment = motion_segments[0] if motion_segments else None
            bins.append(
                {
                    "bin_index": int(bin_index),
                    "sampling_probability": float(sampling_probabilities[bin_index].item()),
                    "bin_failed_count": float(bin_failed_count[bin_index].item()),
                    "global_frame_start": int(global_start),
                    "global_frame_end_exclusive": int(global_end),
                    "global_time_s_start": float(global_start / fps) if fps > 0 else 0.0,
                    "global_time_s_end_exclusive": float(global_end / fps) if fps > 0 else 0.0,
                    "segment_count": int(len(motion_segments)),
                    "motion_id_at_bin_start": int(first_segment["motion_id"]) if first_segment else -1,
                    "motion_file": first_segment["motion_file"] if first_segment else None,
                    "motion_local_frame_start": int(first_segment["motion_local_frame_start"]) if first_segment else -1,
                    "motion_local_frame_end_exclusive": (
                        int(first_segment["motion_local_frame_end_exclusive"]) if first_segment else -1
                    ),
                    "motion_segments": motion_segments,
                }
            )

        bins_sorted_indices = sorted(range(len(bins)), key=lambda index: bins[index]["bin_failed_count"], reverse=True)
        payload = {
            "meta": {
                "step": int(command_step_count),
                "rank": rank,
                "distributed": bool(self.cfg.distributed),
                "motions_dir": str(self.cfg.motion_file),
                "log_save_path": str(self.cfg.log_save_path),
                "fps": fps,
                "motion_num": int(getattr(motion_source, "motion_num", 0) or 0),
                "total_frames": int(motion_source.time_step_total),
                "bin_count": int(self.bin_count),
                "adaptive_kernel_size": int(self.cfg.adaptive_kernel_size),
                "adaptive_lambda": float(self.cfg.adaptive_lambda),
                "adaptive_uniform_ratio": float(self.cfg.adaptive_uniform_ratio),
                "adaptive_alpha": float(self.cfg.adaptive_alpha),
            },
            "bins": bins,
            "bins_sorted": bins_sorted_indices,
        }

        save_dir = Path(self.cfg.log_save_path).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{self.cfg.adaptive_bins_file_prefix}_rank_{rank:02d}_step_{command_step_count:09d}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._last_export_step = command_step_count

    def _compute_sampling_probabilities_for_export(self) -> torch.Tensor:
        clipped_bin_failed_count = torch.clamp(
            self.bin_failed_count,
            max=self.cfg.failure_most_hard_cap_beta * self.bin_failed_count.mean(),
        )
        sampling_probabilities = torch.nn.functional.pad(
            clipped_bin_failed_count.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities,
            self.kernel.view(1, 1, -1),
        ).view(-1)
        sampling_probabilities = sampling_probabilities / (sampling_probabilities.sum() + 1e-12)

        sampling_probabilities_middle_hard = torch.clamp(
            sampling_probabilities,
            max=self.cfg.failure_cap_beta * sampling_probabilities.mean(),
        )
        sampling_probabilities_middle_hard = sampling_probabilities_middle_hard / (
            sampling_probabilities_middle_hard.sum() + 1e-12
        )
        sampling_probabilities_most_hard = torch.clamp(
            sampling_probabilities,
            max=self.cfg.failure_most_hard_cap_beta * sampling_probabilities.mean(),
        )
        sampling_probabilities_most_hard = sampling_probabilities_most_hard / (
            sampling_probabilities_most_hard.sum() + 1e-12
        )
        return (
            self.cfg.motion_ratio[0] * (1 / float(self.bin_count))
            + self.cfg.motion_ratio[1] * sampling_probabilities_middle_hard
            + self.cfg.motion_ratio[2] * sampling_probabilities_most_hard
        )
    #endregion export high prob bins for analysis 
