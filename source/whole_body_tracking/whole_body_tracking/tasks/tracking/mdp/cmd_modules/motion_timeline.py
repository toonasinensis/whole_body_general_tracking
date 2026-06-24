from __future__ import annotations

from .interface import MotionDataSource, MotionSelection

import torch
from collections.abc import Sequence


class MotionCommandTimeline:  # checked
    """Tackle with steps and motion ids"""

    def __init__(self, num_envs: int, future_step_offsets: torch.Tensor, device: str):
        self.device = device
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.local_time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_steps_len = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.future_step_offsets = future_step_offsets

        """
        CommandTimeline 
        用于处理 frame steps 到 motion ids 之间的转换
        暂时不负责 和 motion bins 相关的转换

        拥有变量
        motion_ids      : 环境对应的 motion id
        local_time_steps: 环境运动的当前帧
        motion_steps_len: 环境对应的动作的长度
        future_step_offsets: 用于 indexing 未来帧, 保持不变

        外界可以引用变量
        每次 resample 后变量需要同步
        变量的更改只能通过 apply_selection() 实现
        """

    @property
    def num_future_frames(self) -> int:
        return int(self.future_step_offsets.numel())

    def global_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """
        return the global time steps of all the envs

        TODO 此函数在一步内会被调用多次 尝试每一步 reset 后预先计算 避免重复运算
        """
        return motion_source.time_step_start_idx[self.motion_ids] + self.local_time_steps # (num_envs, )

    def global_start_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """
        return the global start steps of all the envs
        """
        return motion_source.time_step_start_idx[self.motion_ids] # (num_envs, )

    def motion_num_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """
        frame lengths of motions of all the envs
        """
        start = motion_source.time_step_start_idx[self.motion_ids]
        end = motion_source.time_step_end_idx[self.motion_ids]
        return end - start

    def expanded_future_motion_ids(self) -> torch.Tensor:
        """
        Current motion ids repeated for each future frame, flattened.
        (envs_num, ) -> (envs_num * num_future_frames, )
        """
        return self.motion_ids[:, None].expand(-1, self.num_future_frames).reshape(-1)

    def global_future_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        """ 
        Get the future steps of all the envs
        return tensor of shape (envs_num, num_future_frames)

        TODO 这里的计算为什么不使用 self.motion_steps_len 有重复计算嫌疑下一轮需要修掉
        """
        start_steps = self.global_start_steps(motion_source)
        motion_len  = (self.motion_num_steps(motion_source) - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_step_offsets[None, :],
            max=motion_len[:, None],
        ) #(motion_num, num_future_frames)
        return (start_steps[:, None] + future_local).long()

    def build_selection_from_global_timestamps(
        self,
        motion_source: MotionDataSource,
        timestamps: torch.Tensor,
    ) -> MotionSelection:
        """
        build MotionSelection from global timestamps
        global_timestamps -> motion_ids, local_time_step, motion_length

        TODO 本函数并未用到 Timeline 中的变量 后续考虑是否从中拆出
        """
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

    def build_selection_from_motion_ids(
        self,
        motion_source: MotionDataSource,
        motion_ids: torch.Tensor,
        *,
        local_time_steps: torch.Tensor | None = None,
    ) -> MotionSelection:
        """
        build MotionSelection according to given motion ids and local time steps
        motion_ids -> motion_ids, local_time_steps, motion_length
        
        TODO 本函数并未用到 Timeline 中的变量 后续考虑是否从中拆出
        """
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

    def global_timestamps_from_sampled_bins(
        self,
        motion_source: MotionDataSource,
        sampled_bins: torch.Tensor,
        *,
        bin_count: int,
        rewind_min_bins: int = 0,
        rewind_max_bins: int = 0,
    ) -> torch.Tensor:
        """Convert sampled failure bins into spawn timestamps.

        The adaptive sampler still samples bins according to failure statistics.
        Optional rewind moves the actual spawn bin before the sampled failure bin,
        so reset starts from a recovery window instead of the failure instant.

        return tensor of shape sampled_bins

        TODO 本函数已经被弃用 后续删除
        """
        bin_count = int(max(1, bin_count))
        sampled_bins = sampled_bins.to(device=self.device, dtype=torch.long)
        sampled_bins = torch.clamp(sampled_bins, min=0, max=bin_count - 1)

        rewind_min_bins = int(max(0, rewind_min_bins))
        rewind_max_bins = int(max(0, rewind_max_bins))
        total_frames = max(int(motion_source.time_step_total), 1)
        if total_frames == 1: return torch.zeros_like(sampled_bins)


        bin_fraction = sampled_bins.float() + torch.rand(sampled_bins.shape, device=self.device)
        sampled_timestamps = (bin_fraction / float(bin_count) * float(total_frames - 1)).long()

        if rewind_max_bins > 0:
            if rewind_min_bins > rewind_max_bins:
                raise ValueError(
                    "adaptive_sample_rewind_min_bins must be <= adaptive_sample_rewind_bins. "
                    f"Got min={rewind_min_bins}, max={rewind_max_bins}."
                )
            offsets = torch.randint(
                rewind_min_bins,
                rewind_max_bins + 1,
                sampled_bins.shape,
                device=self.device,
            )
            sampled_motion_ids = motion_source.motion_ids_from_timestamps(sampled_timestamps)
            motion_start = motion_source.time_step_start_idx[sampled_motion_ids]
            bin_width = float(total_frames - 1) / float(bin_count)
            rewind_frames = (offsets.float() * bin_width).long()
            return torch.maximum(sampled_timestamps - rewind_frames, motion_start)
        else:
            return sampled_timestamps

    def apply_selection(
            self, 
            env_ids: Sequence[int] | torch.Tensor, 
            selection: MotionSelection
        ) -> None:
        """
        refresh MotionCommandTimeline using MotionSelection
        env_ids:   envs to be reset
        selection: motions to be applied to reset envs
        """
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = selection.motion_ids
        self.local_time_steps[env_ids] = selection.local_time_steps
        self.motion_steps_len[env_ids] = selection.frame_end

    def clear_timeline(self) -> None:
        """
        Clear timeline state so it cannot be interpreted as a valid selection.
        Used when the source motion is reloaded.
        """
        self.motion_ids.zero_()
        self.local_time_steps.zero_()
        self.motion_steps_len.zero_()

    def step(self) -> None:
        """
        add local time steps at every sampling step
        """
        self.local_time_steps += 1

    def expired_env_ids(self, max_future_step: int) -> torch.Tensor:
        """
        return envs in which the step is at the end of the motions         
        """
        return torch.where(self.local_time_steps >= self.motion_steps_len - max_future_step)[0] # length of expired envs
