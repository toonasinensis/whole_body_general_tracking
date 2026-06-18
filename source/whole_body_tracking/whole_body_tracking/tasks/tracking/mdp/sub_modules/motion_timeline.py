from __future__ import annotations

from .interface import MotionDataSource, MotionSelection
import torch
from collections.abc import Sequence


#负责仿真步数和动作缓存之间的转换
# TODO the funtion names shall be algined
class MotionCommandTimeline:  # checked
    """Track per-environment motion ids and local frame cursors."""

    def __init__(self, num_envs: int, future_step_offsets: torch.Tensor, device: str):
        self.device = device
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.local_time_steps  = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frame_end_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.future_step_offsets = future_step_offsets
        # used in evaluation
        self.eval_cycle_count  = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def num_future_frames(self) -> int:
        return int(self.future_step_offsets.numel())

    def global_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        return motion_source.time_step_start_idx[self.motion_ids] + self.local_time_steps

    def motion_start_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        return motion_source.time_step_start_idx[self.motion_ids]

    def motion_num_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """frame lengths of motions"""
        start = motion_source.time_step_start_idx[self.motion_ids]
        end = motion_source.time_step_end_idx[self.motion_ids]
        return end - start

    def future_motion_ids(self) -> torch.Tensor:
        """(envs_num, ) -> (envs_num, num_future_frames)"""
        return self.motion_ids[:, None].expand(-1, self.num_future_frames).reshape(-1)

    def future_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """ -> (envs_num, num_future_frames)"""
        start = self.motion_start_time_steps(motion_source)
        local_max = (self.motion_num_steps(motion_source) - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_step_offsets[None, :],
            max=local_max[:, None],
        ) #(motion_num, num_future_frames)
        return (start[:, None] + future_local).long()

    def selection_from_global_timestamps(
        self,
        motion_source: MotionDataSource,
        timestamps: torch.Tensor,
    ) -> MotionSelection:
        """global_timestamps -> motion_ids, local_time_step, motion_length"""
        if timestamps.dtype != torch.long:
            timestamps = timestamps.long()
        timestamps = torch.clamp(timestamps, min=0, max=int(motion_source.time_step_total) - 1)
        motion_ids = motion_source.motion_ids_from_timestamps(timestamps)
        start = motion_source.time_step_start_idx[motion_ids]
        end = motion_source.time_step_end_idx[motion_ids]
        return MotionSelection(
            motion_ids=motion_ids,
            local_time_steps=timestamps - start,
            frame_end=end - start,
        )

    def selection_from_motion_ids(
        self,
        motion_source: MotionDataSource,
        motion_ids: torch.Tensor,
        *,
        local_time_steps: torch.Tensor | None = None,
    ) -> MotionSelection:
        """motion_ids -> motion_ids, local_time_steps, motion_length"""
        motion_ids = motion_ids.long()
        start = motion_source.time_step_start_idx[motion_ids]
        end = motion_source.time_step_end_idx[motion_ids]
        if local_time_steps is None:
            local_time_steps = torch.zeros_like(motion_ids)
        return MotionSelection(
            motion_ids=motion_ids,
            local_time_steps=local_time_steps.long(),
            frame_end=end - start,
        )

    def apply_selection(self, env_ids: Sequence[int], selection: MotionSelection) -> None:
        """refresh MotionCommandTimeline using MotionSelection"""
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = selection.motion_ids
        self.local_time_steps[env_ids] = selection.local_time_steps
        self.frame_end_per_env[env_ids] = selection.frame_end

    def invalidate(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear timeline state so it cannot be interpreted as a valid selection."""
        if env_ids is None:
            self.motion_ids.zero_()
            self.local_time_steps.zero_()
            self.frame_end_per_env.zero_()
            return
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = 0
        self.local_time_steps[env_ids] = 0
        self.frame_end_per_env[env_ids] = 0

    def step(self) -> None:
        self.local_time_steps += 1

    def expired_env_ids(self, max_future_step: int) -> torch.Tensor:
        """(num_envs, )"""
        return torch.where(self.local_time_steps >= self.frame_end_per_env - max_future_step)[0]
