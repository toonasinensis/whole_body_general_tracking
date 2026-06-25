"""
本模块用于替换 @motion_sampling.py 中的 AdaptiveMotionSampler 类
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .interface import MotionDataSource, MotionSelection
from .motion_timeline import MotionCommandTimeline


class AdaptiveMotionSampler:
    """Maintain adaptive-sampling state and produce motion selections."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        
        # 抽取常用的 cfg 数据
        self.adaptive_lambda = cfg.adaptive_lambda
        self.adaptive_kernel_size = cfg.adaptive_kernel_size
        self.eval_mode = cfg.eval_mode
        self.adaptive_alpha = cfg.adaptive_alpha
        self.failure_cap_beta = cfg.failure_cap_beta
        self.failure_most_hard_cap_beta = cfg.failure_most_hard_cap_beta
        self.motion_ratio = cfg.motion_ratio
        self.max_future_step = cfg.max_future_step
        self.motion_sampling_start_frame = cfg.motion_sampling_start_frame

        self._init_sampling_kernel()
        self.bin_frame_width: int = 50 # frames num in general motion bin

        self.max_motion_bin_count = 0
        self.total_motion_bin_num = 0
        self.motion_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)
        self.pending_motion_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)
        self.bin_count_per_motion = torch.zeros(0, dtype=torch.long, device=device)  # bin nums in each motion
        self.bin_start_idx = torch.zeros(0, dtype=torch.long, device=device)
        self.bin_end_idx = torch.zeros(0, dtype=torch.long, device=device)

        """
        AdaptiveMotionSampler 
        
        不对外开放任何自有变量
        
        对外暴露三个方法
        reset_for_motion_source
        sample_selection
        step_post_update

        TODO 把 adaptive_sample_rewind_bins / adaptive_sample_rewind_min_bins 用于 shift frame idx
        TODO 创立自己的 Config
        TODO step_post_update 方法可以收到自有方案
        """

    def _init_sampling_kernel(self) -> None:
        """
        初始化 sampling kernel
        """
        self.sampling_kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.sampling_kernel = self.sampling_kernel / self.sampling_kernel.sum()

    def reset_for_motion_source( #checked
        self, 
        motion_source: MotionDataSource, 
        *, 
        decimation: int, 
        sim_dt: float
    ) -> None:
        """
        Reset adaptive-sampling state after loading a new motion source.
        在重新加载 motion data source 时会调用一次
        """
        self.bin_frame_width = max(1, int(round(1.0 / (decimation * sim_dt))))
        motion_start_idx = motion_source.time_step_start_idx
        motion_end_idx = motion_source.time_step_end_idx 
        motion_num = motion_source.motion_num
        
        # 计算每个 motion 含有的 bins 数量
        motion_lengths = motion_end_idx - motion_start_idx

        motion_bin_count = motion_lengths // self.bin_frame_width + (
            motion_lengths % self.bin_frame_width != 0
        ).long()  # 如果最后一段 bin 不满一帧，则认为它也算一个 bin

        self.max_motion_bin_count = int(motion_bin_count.max().item())
        self.total_motion_bin_num = int(motion_bin_count.sum().item())
        self.bin_count_per_motion = motion_bin_count
        self.bin_start_idx = torch.zeros(int(self.total_motion_bin_num), dtype=torch.long, device=self.device)
        self.bin_end_idx   = torch.zeros(int(self.total_motion_bin_num), dtype=torch.long, device=self.device)
        
        # 计算每个 bin 的起始和结束索引（flatten 后的 global bin id）
        global_bin_offset = 0
        for motion_id in range(motion_num):
            start_idx = motion_start_idx[motion_id]
            end_idx = motion_end_idx[motion_id]
            bin_num = int(self.bin_count_per_motion[motion_id].item())
            bin_ids = torch.arange(bin_num, device=self.device, dtype=torch.long)
            flat_ids = global_bin_offset + bin_ids
            self.bin_start_idx[flat_ids] = start_idx + bin_ids * self.bin_frame_width
            self.bin_end_idx[flat_ids] = torch.minimum(
                start_idx + (bin_ids + 1) * self.bin_frame_width,
                end_idx,
            )
            global_bin_offset += bin_num
        
        # 分配显存空间
        self.motion_failure_bin_counts = torch.zeros(self.total_motion_bin_num, dtype=torch.float, device=self.device)   # num_motions, max_motion_bins_count
        self.pending_motion_failure_bin_counts = torch.zeros(self.total_motion_bin_num, dtype=torch.float, device=self.device)

    def _sample_frame_idx(
        self,
        bin_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        给定 bin_idx 从中随机采样一个 frame_idx
        """
        bin_idx = bin_idx.to(device=self.device, dtype=torch.long)
        start_idx = self.bin_start_idx[bin_idx]
        end_idx = self.bin_end_idx[bin_idx]
        span = (end_idx - start_idx).clamp(min=1)
        return start_idx + (torch.rand(bin_idx.shape, device=self.device) * span.float()).long()

    def _shift_frame_idx(
        self,
        frame_idx: torch.Tensor,
        data_source: MotionDataSource,
    ) -> torch.Tensor:
        """
        根据 self.max_future_step 和 self.motion_sampling_start_frame 调整 frame_idx。
        将给定的 frame_idx 往前随机移动 [0, self.bin_frame_width] 步，
        并确保结果落在对应 motion 的全局 frame 范围
        [motion_start + motion_sampling_start_frame, motion_end - max_future_step - 1] 内。
        """
        device = data_source.time_step_start_idx.device
        frame_idx = frame_idx.to(device=device, dtype=torch.long)

        motion_idx = self._frame_idx_to_motion_idx(frame_idx, data_source)
        motion_start_idx = data_source.time_step_start_idx[motion_idx]
        motion_end_idx = data_source.time_step_end_idx[motion_idx]
        min_frame_idx = motion_start_idx + self.motion_sampling_start_frame
        max_frame_idx = motion_end_idx - self.max_future_step - 1

        invalid_mask = max_frame_idx < min_frame_idx
        max_frame_idx = torch.maximum(max_frame_idx, min_frame_idx)

        offset = torch.randint(0, self.bin_frame_width + 1, frame_idx.shape, device=device)
        frame_idx = (frame_idx - offset).clamp(min=min_frame_idx, max=max_frame_idx)

        # 处理 motion 过短的情况
        # 当 max_frame_idx < min_frame_idx 时
        # 将 对应的 frame_idx 替换为任意 合法的 frame_idx
        if torch.any(invalid_mask):
            valid_mask = ~invalid_mask
            if torch.any(valid_mask):
                valid_frame_idx = frame_idx[valid_mask]
                replacement_idx = torch.randint(
                    0,
                    int(valid_frame_idx.numel()),
                    (int(invalid_mask.sum().item()),),
                    device=device,
                )
                frame_idx[invalid_mask] = valid_frame_idx[replacement_idx]
            
            # 最坏情况下返回不合法的 frame_idx
            # 但影响应该不大

        return frame_idx

    def _frame_idx_to_bin_idx(
        self, 
        frame_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        将 frame idx 映射到 bins 空间
        注意 frame idx 是全局 frame idx
        而 bin idx 是 flatten 后的 global bin id
        bin 的 start idx 和 end idx 已经计算好了
        """
        if self.total_motion_bin_num <= 0:
            return torch.zeros_like(frame_idx, dtype=torch.long, device=self.device)

        frame_idx = frame_idx.to(device=self.device, dtype=torch.long)
        # bin_end_idx 为 exclusive upper bound
        bin_idx = torch.searchsorted(self.bin_end_idx, frame_idx, right=True)
        return bin_idx.clamp(max=int(self.total_motion_bin_num) - 1)

    def _frame_idx_to_motion_idx( 
        self,
        frame_idx: torch.Tensor,
        data_source: MotionDataSource,
    ) -> torch.Tensor:
        """
        将 frame idx 映射到 motion idx
        frame_idx 是全局 frame idx
        """
        return data_source.motion_ids_from_timestamps(frame_idx)

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
        """
        Sample new motions according to failure metrics.
        
        每轮都会被调用
        """
        if len(env_ids) == 0:
            raise ValueError("env_ids must not be empty for adaptive sampling")

        if allow_failure_accounting:
            self._record_failed_bins(
                env_ids=env_ids,
                motion_source=motion_source,
                timeline=timeline,
                terminated=terminated,
            )

        # TODO 考虑将 step_post_update 从外部引用移到此处 当前先保持原有实现

        sampling_prob = self._compute_sampling_prob()
        sampled_bin_idx = torch.multinomial(
            sampling_prob,
            len(env_ids),
            replacement=True,
        )
        sampled_frame_idx = self._sample_frame_idx(bin_idx=sampled_bin_idx)
        sampled_frame_idx = self._shift_frame_idx(
            frame_idx=sampled_frame_idx, 
            data_source=motion_source
        )
        sampled_motion_idx = self._frame_idx_to_motion_idx(
            frame_idx=sampled_frame_idx, 
            data_source=motion_source
        )
        sampled_local_time_steps = sampled_frame_idx - motion_source.time_step_start_idx[sampled_motion_idx]

        self._update_sampling_metrics(metrics, sampling_prob)
        self._update_failure_metrics(metrics)

        selection = timeline.build_selection_from_motion_ids(
            motion_source,
            sampled_motion_idx,
            local_time_steps=sampled_local_time_steps,
        )
        if self.eval_mode:
            selection.local_time_steps[:] = int(max(0, self.motion_sampling_start_frame))
        return selection

    def _record_failed_bins(
        self,
        *,
        env_ids: Sequence[int],
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
    ) -> None:
        """
        将失败的 bins 统计到 pending_motion_failure_bins 中
        """
        failed_env_mask = terminated[env_ids]
        if not torch.any(failed_env_mask):
            return

        current_global_time_steps = torch.clamp(
            timeline.global_time_steps(motion_source) - 1,
            min=0,
            max=int(motion_source.time_step_total) - 1,
        )   # num_envs
        
        failed_frame_idx = current_global_time_steps[env_ids][failed_env_mask] # num failed envs
        failed_bin_idx = self._frame_idx_to_bin_idx(failed_frame_idx) # num failed envs
        self.pending_motion_failure_bin_counts += torch.bincount(
            failed_bin_idx,
            minlength=int(self.total_motion_bin_num),
        ).float()

    def _clip_and_smooth_failure_bin_counts(
        self,
        failure_bin_counts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip failure counts and smooth along the flatten global bin axis.

        NOTE 这里为了兼容 处理速度和显存占用 在做平滑时 没有考虑 motion 的边界
        TODO 考虑削峰操作是否应该删除
        """
        # 处理全零情况
        failure_mean = failure_bin_counts.mean()
        if failure_mean == 0: return failure_bin_counts

        clipped = torch.clamp(
            failure_bin_counts, 
            max=self.failure_most_hard_cap_beta * failure_mean)
        
        kernel_size = int(self.sampling_kernel.numel())
        if kernel_size <= 1:
            return clipped

        # Keep conv output length equal to the input length. Symmetric padding
        # over-pads by one when kernel_size is even.
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size - 1 - pad_left
        pad_mode = "reflect" if failure_bin_counts.numel() > max(pad_left, pad_right) else "replicate"
        padded = torch.nn.functional.pad(
            clipped.unsqueeze(0).unsqueeze(0), 
            (pad_left, pad_right), 
            mode=pad_mode)
        
        smoothed = torch.nn.functional.conv1d(
            padded, 
            self.sampling_kernel.view(1, 1, -1)
            ).squeeze(0).squeeze(0)

        return smoothed

    def _cap_failure_bin_probabilities(
        self,
        probabilities: torch.Tensor,
        cap_beta: float,
    ) -> torch.Tensor:
        """
        Cap probabilities and renormalize over all flatten bins.
        probabilities 是 flatten global bin 的采样概率
        对输入的采样概率削峰，防止某些 bin 的采样概率过大
        """
        capped_probabilities = torch.clamp(
            probabilities,
            max=cap_beta * probabilities.mean(),
        )
        return capped_probabilities / (capped_probabilities.sum() + 1e-12)

    def _compute_sampling_prob(self) -> torch.Tensor:
        """
        根据 self.motion_failure_bin_counts 计算 flatten global bin 的采样概率

        TODO 考虑是否有更好的概率计算方式
        TODO 重新命变量 增强可读性
        """
        if self.total_motion_bin_num <= 0:
            raise RuntimeError("No valid adaptive sampling bins. Check motion lengths and max_future_step.")

        clipped_bin_counts = self._clip_and_smooth_failure_bin_counts(
            self.motion_failure_bin_counts)
        uniform_probs = torch.full_like(
            clipped_bin_counts,
            1.0 / float(self.total_motion_bin_num))

        total = clipped_bin_counts.sum()
        if total <= 1e-12:
            return uniform_probs

        hard_probs = clipped_bin_counts / (total + 1e-12)
        mid_probs = self._cap_failure_bin_probabilities(
            hard_probs,
            cap_beta=float(self.failure_cap_beta))
        top_probs = self._cap_failure_bin_probabilities(
            hard_probs,
            cap_beta=float(self.failure_most_hard_cap_beta))

        probs = (
            self.motion_ratio[0] * uniform_probs
            + self.motion_ratio[1] * mid_probs
            + self.motion_ratio[2] * top_probs
        )

        return probs / (probs.sum() + 1e-12)

    def step_post_update(self) -> None: # checked
        """
        Apply EMA updates to accumulated failure-bin counts.

        每轮调用以更新失败 bins 的统计信息
        TODO 把函数名改得更直观一些
        """
        self.motion_failure_bin_counts[:] = (
            self.adaptive_alpha * self.pending_motion_failure_bin_counts
            + (1.0 - self.adaptive_alpha) * self.motion_failure_bin_counts
        )
        self.pending_motion_failure_bin_counts.zero_()

    def _update_sampling_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        sampling_prob: torch.Tensor,
    ) -> None:
        """Update sampling metrics without mutating sampler state."""
        valid_bin_count = max(int(self.total_motion_bin_num), 1)
        sampling_entropy = -(sampling_prob * (sampling_prob + 1e-12).log()).sum()
        normalized_entropy = sampling_entropy / math.log(max(valid_bin_count, 2))
        top_prob, top_prob_idx = sampling_prob.max(dim=0)

        metrics["sampling_entropy"][:] = normalized_entropy
        metrics["sampling_top1_prob_max"][:] = top_prob
        metrics["prob_max_over_uniform"][:] = top_prob / (1.0 / float(valid_bin_count))
        metrics["prob_uniform"][:] = 1.0 / float(valid_bin_count)
        metrics["sampling_top1_prob_bin"][:] = top_prob_idx.float() / max(
            float(sampling_prob.numel()),
            1.0,
        )
        metrics["sampling_top1_prob_mean"][:] = sampling_prob.mean()
        metrics["sampling_top1_prob_min"][:] = sampling_prob.min()
        metrics["num_concentrate_bins"][:] = (
            sampling_prob > self.failure_most_hard_cap_beta * 0.5 * sampling_prob.mean()
        ).sum()

    def _update_failure_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        """Update metrics that describe failure-bin statistics."""
        failure_counts = self.motion_failure_bin_counts
        metrics["failures_max"][:] = failure_counts.max()
        metrics["failures_mean"][:] = failure_counts.mean()
        metrics["failures_min"][:] = failure_counts.min()
        metrics["failures_max_over_uniform"][:] = (
            failure_counts.max()
            * float(self.total_motion_bin_num)
            / (failure_counts.sum() + 1e-8)
        )
