from __future__ import annotations

import math
from typing import Sequence

import torch

from .interface import MotionDataSource, MotionSelection
from .motion_timeline import MotionCommandTimeline


class AdaptiveMotionSampler:
    """Maintain adaptive-sampling state and produce motion selections."""

    # TODO 把 restart bin offset 写入类的 config
    RESTART_BIN_OFFSET = 1

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device

        self.bin_frame_width: int | None = None # frames num in general motion bin
        self.success_motion = torch.zeros(0, dtype=torch.float32, device=device)
        self.sampling_kernel = torch.zeros(0, dtype=torch.float, device=device)

        # TODO 当前 global bin 只用于更新 metrics 占用大量的计算资源 删掉
        self.global_bin_count = 0
        self.global_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)
        self.pending_global_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)
        
        # TODO 当前 motion_bin 的存储方式和计算方式不合理，要大改
        self.max_motion_bin_count = 0
        self.motion_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)   # num_motions, max_motion_bins_count
        self.pending_motion_failure_bin_counts = torch.zeros(0, dtype=torch.float, device=device)
        self.motion_bin_count_per_motion = torch.zeros(0, dtype=torch.long, device=device)  # bin nums in each motion

        """
        本类只关心 motion source 中的排列结构
        相较而言 timeline 关心 envs 中 motion 的相关属性

        切分结构
            motion bins             |       |       |       |       |       |
            motions -> local bins   |               |                       |

        本类只暴露 reset_for_motion_source  sample_selection  step_post_update 三个函数
        不向外暴露任何变量
        """

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

        # TODO 这里 global_bins 和 motion_bins 的计算方式不一致 后续需要删除或者统一
        self.global_bin_count = int(motion_source.time_step_total // self.bin_frame_width) + 1
        self.global_failure_bin_counts = torch.zeros(self.global_bin_count, dtype=torch.float, device=self.device)
        self.pending_global_failure_bin_counts = torch.zeros(
            self.global_bin_count, dtype=torch.float, device=self.device
        )

        self.motion_bin_count_per_motion = self._compute_motion_bin_count_per_motion(
            time_step_start_idx=motion_source.time_step_start_idx,
            time_step_end_idx=motion_source.time_step_end_idx,
        )
        self.max_motion_bin_count = int(self.motion_bin_count_per_motion.max().item())
        motion_count = int(getattr(motion_source, "motion_num", 0) or 0)
        # TODO 这种 motion_failure_bin_counts 的形式会造成大量的空间浪费
        self.motion_failure_bin_counts = torch.zeros(
            (motion_count, self.max_motion_bin_count),
            dtype=torch.float,
            device=self.device,
        )
        self.pending_motion_failure_bin_counts = torch.zeros_like(self.motion_failure_bin_counts)

        self.sampling_kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.sampling_kernel = self.sampling_kernel / self.sampling_kernel.sum()
        self.success_motion = torch.zeros(motion_count, dtype=torch.float32, device=self.device)

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

        self._update_global_failure_metrics(metrics)
        sampling_probabilities, valid_failure_bin_mask = self._compute_motion_failure_bin_sampling_probabilities()

        sampled_flat_failure_bin_ids = torch.multinomial(
            sampling_probabilities,
            len(env_ids),
            replacement=True,
        )
        sampled_motion_ids = sampled_flat_failure_bin_ids // self.max_motion_bin_count
        sampled_failure_bin_ids = sampled_flat_failure_bin_ids % self.max_motion_bin_count
        sampled_restart_bin_ids = self._shift_failure_bins_to_restart_bins(
            motion_ids=sampled_motion_ids,
            failure_bin_ids=sampled_failure_bin_ids,
        )
        sampled_motion_ids, sampled_local_time_steps = self._sample_local_time_steps_from_motion_bins(
            motion_ids=sampled_motion_ids,
            motion_bin_ids=sampled_restart_bin_ids,
            time_step_start_idx=motion_source.time_step_start_idx,
            time_step_end_idx=motion_source.time_step_end_idx,
        )

        selection = timeline.build_selection_from_motion_ids(
            motion_source,
            sampled_motion_ids,
            local_time_steps=sampled_local_time_steps,
        )
        if self.cfg.eval_mode:
            selection.local_time_steps[:] = int(max(0, getattr(self.cfg, "motion_sampling_start_frame", 0)))
        self._update_sampling_metrics(metrics, sampling_probabilities, valid_failure_bin_mask)
        return selection

    def step_post_update(self) -> None: # checked
        """
        Apply EMA updates to accumulated failure-bin counts.

        每轮调用以更新失败 bins 的统计信息
        TODO 把函数名改得更直观一些
        """
        del motion_source, command_step_count
        if self.global_failure_bin_counts.numel() == 0:
            return

        self.global_failure_bin_counts[:] = (
            self.cfg.adaptive_alpha * self.pending_global_failure_bin_counts
            + (1 - self.cfg.adaptive_alpha) * self.global_failure_bin_counts
        )
        self.pending_global_failure_bin_counts.zero_()

        self.motion_failure_bin_counts[:] = (
            self.cfg.adaptive_alpha * self.pending_motion_failure_bin_counts
            + (1 - self.cfg.adaptive_alpha) * self.motion_failure_bin_counts
        )
        self.pending_motion_failure_bin_counts.zero_()

    def _record_failed_bins(
        self,
        *,
        env_ids: Sequence[int],
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
    ) -> None:
        """
        Accumulate exact global and motion-local failure-bin counts for this resample step.
        """

        failed_env_mask = terminated[env_ids]
        if not torch.any(failed_env_mask):
            return

        # fetch 
        current_global_time_steps = torch.clamp(
            timeline.global_time_steps(motion_source) - 1,
            min=0,
            max=int(motion_source.time_step_total) - 1,
        )
        current_global_bin_ids = torch.clamp(
            (current_global_time_steps * self.global_bin_count) // max(motion_source.time_step_total, 1),
            min=0,
            max=self.global_bin_count - 1,
        )
        failed_global_bin_ids = current_global_bin_ids[env_ids][failed_env_mask]
        self.pending_global_failure_bin_counts += torch.bincount(
            failed_global_bin_ids,
            minlength=self.global_bin_count,
        )

        failed_global_time_steps = current_global_time_steps[env_ids][failed_env_mask]
        failed_motion_ids = motion_source.motion_ids_from_timestamps(failed_global_time_steps)
        failed_local_time_steps = failed_global_time_steps - motion_source.time_step_start_idx[failed_motion_ids]
        self.pending_motion_failure_bin_counts += self._accumulate_motion_failure_bin_counts(
            failed_motion_ids=failed_motion_ids,
            failed_local_time_steps=failed_local_time_steps,
            time_step_start_idx=motion_source.time_step_start_idx,
            time_step_end_idx=motion_source.time_step_end_idx,
            max_motion_bin_count=self.max_motion_bin_count,
        )

    def _compute_motion_failure_bin_sampling_probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return probabilities over valid motion-local failure bins.
        
        """
        valid_failure_bin_mask = (
            torch.arange(self.max_motion_bin_count, device=self.device)[None, :]
            < self.motion_bin_count_per_motion[:, None]
        )
        valid_flat_failure_bin_mask = valid_failure_bin_mask.reshape(-1)
        valid_failure_bin_count = int(valid_flat_failure_bin_mask.sum().item())
        if valid_failure_bin_count <= 0:
            raise RuntimeError("No valid adaptive sampling bins. Check motion lengths and max_future_step.")

        clipped_failure_bin_counts = self._clip_and_smooth_failure_bin_counts(
            failure_bin_counts=self.motion_failure_bin_counts,
            valid_failure_bin_mask=valid_failure_bin_mask,
        )
        flat_failure_bin_probabilities = torch.where(
            valid_failure_bin_mask,
            clipped_failure_bin_counts,
            torch.zeros_like(clipped_failure_bin_counts),
        ).reshape(-1)

        uniform_failure_bin_probabilities = valid_flat_failure_bin_mask.float() / float(valid_failure_bin_count)
        total_probability = flat_failure_bin_probabilities.sum()
        if total_probability > 1e-12:
            flat_failure_bin_probabilities = flat_failure_bin_probabilities / (total_probability + 1e-12)
            middle_hard_probabilities = self._cap_failure_bin_probabilities(
                probabilities=flat_failure_bin_probabilities,
                valid_probability_mask=valid_flat_failure_bin_mask,
                cap_beta=float(self.cfg.failure_cap_beta),
            )
            most_hard_probabilities = self._cap_failure_bin_probabilities(
                probabilities=flat_failure_bin_probabilities,
                valid_probability_mask=valid_flat_failure_bin_mask,
                cap_beta=float(self.cfg.failure_most_hard_cap_beta),
            )
        else:
            middle_hard_probabilities = uniform_failure_bin_probabilities
            most_hard_probabilities = uniform_failure_bin_probabilities

        sampling_probabilities = (
            self.cfg.motion_ratio[0] * uniform_failure_bin_probabilities
            + self.cfg.motion_ratio[1] * middle_hard_probabilities
            + self.cfg.motion_ratio[2] * most_hard_probabilities
        )
        sampling_probabilities = torch.where(
            valid_flat_failure_bin_mask,
            sampling_probabilities,
            torch.zeros_like(sampling_probabilities),
        )
        sampling_probabilities = sampling_probabilities / (sampling_probabilities.sum() + 1e-12)
        return sampling_probabilities, valid_flat_failure_bin_mask

    def _clip_and_smooth_failure_bin_counts(
        self,
        *,
        failure_bin_counts: torch.Tensor,
        valid_failure_bin_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clip counts on valid bins and smooth them along the motion-bin axis.
        """
        valid_failure_mean = failure_bin_counts[valid_failure_bin_mask].mean()
        clipped_failure_bin_counts = torch.clamp(
            failure_bin_counts,
            max=self.cfg.failure_most_hard_cap_beta * valid_failure_mean,
        )
        clipped_failure_bin_counts = torch.where(
            valid_failure_bin_mask,
            clipped_failure_bin_counts,
            torch.zeros_like(clipped_failure_bin_counts),
        )
        smoothed_failure_bin_counts = torch.nn.functional.pad(
            clipped_failure_bin_counts.unsqueeze(1),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="constant",
            value=0.0,
        )
        smoothed_failure_bin_counts = torch.nn.functional.conv1d(
            smoothed_failure_bin_counts,
            self.sampling_kernel.view(1, 1, -1),
        ).squeeze(1)
        return smoothed_failure_bin_counts

    def _cap_failure_bin_probabilities(
        self,
        *,
        probabilities: torch.Tensor,
        valid_probability_mask: torch.Tensor,
        cap_beta: float,
    ) -> torch.Tensor:
        """Cap valid probabilities and renormalize."""
        capped_probabilities = torch.clamp(
            probabilities,
            max=cap_beta * probabilities[valid_probability_mask].mean(),
        )
        capped_probabilities = torch.where(
            valid_probability_mask,
            capped_probabilities,
            torch.zeros_like(capped_probabilities),
        )
        return capped_probabilities / (capped_probabilities.sum() + 1e-12)

    def _accumulate_motion_failure_bin_counts(
        self,
        *,
        failed_motion_ids: torch.Tensor,
        failed_local_time_steps: torch.Tensor,
        time_step_start_idx: torch.Tensor,
        time_step_end_idx: torch.Tensor,
        max_motion_bin_count: int | None = None,
    ) -> torch.Tensor:
        """
        Accumulate exact failed motion-local bin counts.
        把一批失败样本，统计成一个 [motion_id, local_bin_id] 的计数矩阵
        """
        device = time_step_start_idx.device
        failed_motion_ids = failed_motion_ids.to(device=device, dtype=torch.long)
        failed_local_time_steps = failed_local_time_steps.to(device=device, dtype=torch.long)

        motion_bin_count_per_motion = self._compute_motion_bin_count_per_motion(
            time_step_start_idx=time_step_start_idx,
            time_step_end_idx=time_step_end_idx,
        )
        if max_motion_bin_count is None:
            max_motion_bin_count = int(motion_bin_count_per_motion.max().item())
        max_motion_bin_count = int(max(1, max_motion_bin_count))

        motion_failure_bin_counts = torch.zeros(
            (int(time_step_start_idx.numel()), max_motion_bin_count),
            dtype=torch.float32,
            device=device,
        )
        if failed_motion_ids.numel() == 0:
            return motion_failure_bin_counts

        motion_lengths = time_step_end_idx - time_step_start_idx
        failed_local_time_steps = torch.clamp(failed_local_time_steps, min=0)
        failed_local_time_steps = torch.minimum(
            failed_local_time_steps,
            motion_lengths[failed_motion_ids] - 1,
        )
        failed_motion_bin_ids = failed_local_time_steps // self.bin_frame_width

        min_local_time_step, _ = self._compute_valid_local_time_step_bounds(
            time_step_start_idx=time_step_start_idx,
            time_step_end_idx=time_step_end_idx,
        )
        min_motion_bin_id = min_local_time_step // self.bin_frame_width
        max_motion_bin_ids = min_motion_bin_id + motion_bin_count_per_motion[failed_motion_ids] - 1
        failed_motion_bin_ids = torch.clamp(failed_motion_bin_ids, min=min_motion_bin_id)
        failed_motion_bin_ids = torch.minimum(failed_motion_bin_ids, max_motion_bin_ids)

        storage_bin_ids = failed_motion_bin_ids - min_motion_bin_id
        flat_indexes = failed_motion_ids * max_motion_bin_count + storage_bin_ids
        flat_motion_failure_bin_counts = motion_failure_bin_counts.view(-1)
        flat_motion_failure_bin_counts.scatter_add_(
            0,
            flat_indexes,
            torch.ones((failed_motion_ids.numel(),), dtype=motion_failure_bin_counts.dtype, device=device),
        )
        return motion_failure_bin_counts

    def _shift_failure_bins_to_restart_bins( # checked
        self,
        *,
        motion_ids: torch.Tensor,
        failure_bin_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Shift sampled failure bins backward by one bin to get restart bins.
        
        args:
            failure_bin_ids: local bin ids in motions

        return:
            expected local restart bin ids
        """
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
        failure_bin_ids = failure_bin_ids.to(device=self.device, dtype=torch.long)

        restart_bin_ids = failure_bin_ids - self.RESTART_BIN_OFFSET
        max_restart_bin_ids = self.motion_bin_count_per_motion[motion_ids] - 1
        restart_bin_ids = torch.clamp(restart_bin_ids, min=0)
        restart_bin_ids = torch.minimum(restart_bin_ids, max_restart_bin_ids)
        return restart_bin_ids

    def _sample_local_time_steps_from_motion_bins( # checked
        self,
        *,
        motion_ids: torch.Tensor,
        motion_bin_ids: torch.Tensor,
        time_step_start_idx: torch.Tensor,
        time_step_end_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample local time steps inside already-selected motion-local bins.
        bin 已经选好之后，从这个 bin 对应的合法 frame 区间里，再随机采一个最终的参考帧
        """
        device = time_step_start_idx.device
        motion_ids = motion_ids.to(device=device, dtype=torch.long)
        motion_bin_ids = motion_bin_ids.to(device=device, dtype=torch.long)
        min_local_time_step, max_local_time_steps = self._compute_valid_local_time_step_bounds(
            time_step_start_idx=time_step_start_idx,
            time_step_end_idx=time_step_end_idx,
        )

        # allocate usable motion bins
        # TODO 这里的逻辑就是一坨大粪 找时间改掉
        min_motion_bin_id = min_local_time_step // self.bin_frame_width
        motion_bin_ids = motion_bin_ids + min_motion_bin_id
        max_motion_bin_ids = (max_local_time_steps[motion_ids] // self.bin_frame_width).long()
        motion_bin_ids = torch.minimum(motion_bin_ids, max_motion_bin_ids)

        # sample local time steps
        lower_time_steps = motion_bin_ids * self.bin_frame_width
        lower_time_steps = torch.clamp(lower_time_steps, min=min_local_time_step)
        upper_time_steps = torch.minimum(
            (motion_bin_ids + 1) * self.bin_frame_width,
            max_local_time_steps[motion_ids] + 1,
        )
        sample_span = torch.clamp(upper_time_steps - lower_time_steps, min=1)
        sampled_local_time_steps = lower_time_steps + (
            torch.rand(sample_span.shape, device=device) * sample_span.float()
        ).long()
        return motion_ids, sampled_local_time_steps

    def _compute_valid_local_time_step_bounds( # checked
        self,
        *,
        time_step_start_idx: torch.Tensor,
        time_step_end_idx: torch.Tensor,
    ) -> tuple[int, torch.Tensor]:
        """
        Compute valid local time-step bounds for each motion.

        TODO 将 motion_sampling_start_frame 写入私有 config
        TODO 本函数只要在 data source 更换时计算 以免重复计算
        """
        motion_lengths = time_step_end_idx - time_step_start_idx
        min_local_time_step = int(max(0, getattr(self.cfg, "motion_sampling_start_frame", 0)))
        max_local_time_steps = motion_lengths - int(self.cfg.max_future_step) - 1
        return min_local_time_step, max_local_time_steps

    def _compute_motion_bin_count_per_motion( # checked
        self,
        *,
        time_step_start_idx: torch.Tensor,
        time_step_end_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute how many valid motion-local bins each motion owns.

        TODO 保证该函数只在更换 data source 时调用 以避免重复计算
        """
        min_local_time_step, max_local_time_steps = self._compute_valid_local_time_step_bounds(
            time_step_start_idx=time_step_start_idx,
            time_step_end_idx=time_step_end_idx,
        )
        min_motion_bin_id = min_local_time_step // self.bin_frame_width
        return (max_local_time_steps // self.bin_frame_width - min_motion_bin_id + 1).long()

    def _update_global_failure_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        """Update metrics that describe global failure-bin statistics."""
        metrics["failures_max"][:] = self.global_failure_bin_counts.max()
        metrics["failures_mean"][:] = self.global_failure_bin_counts.mean()
        metrics["failures_min"][:] = self.global_failure_bin_counts.min()
        metrics["failures_max_over_uniform"][:] = (
            self.global_failure_bin_counts.max()
            * self.global_bin_count
            / (self.global_failure_bin_counts.sum() + 1e-8)
        )
    
    def _update_sampling_metrics(
        self,
        metrics: dict[str, torch.Tensor],
        sampling_probabilities: torch.Tensor,
        valid_failure_bin_mask: torch.Tensor | None = None,
    ) -> None:
        """Update sampling metrics without mutating sampler state."""
        sampling_entropy = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        valid_failure_bin_count = (
            int(valid_failure_bin_mask.sum().item()) if valid_failure_bin_mask is not None else self.global_bin_count
        )
        normalized_entropy = sampling_entropy / math.log(max(valid_failure_bin_count, 2))
        top_probability, top_probability_index = sampling_probabilities.max(dim=0)
        metrics["sampling_entropy"][:] = normalized_entropy
        metrics["sampling_top1_prob_max"][:] = top_probability
        metrics["prob_max_over_uniform"][:] = top_probability / (1 / max(float(valid_failure_bin_count), 1.0))
        metrics["prob_uniform"][:] = 1 / max(float(valid_failure_bin_count), 1.0)
        metrics["sampling_top1_prob_bin"][:] = top_probability_index.float() / max(
            float(sampling_probabilities.numel()),
            1.0,
        )
        metrics["sampling_top1_prob_mean"][:] = sampling_probabilities.mean()
        metrics["sampling_top1_prob_min"][:] = sampling_probabilities.min()
        metrics["num_concentrate_bins"][:] = (
            sampling_probabilities
            > self.cfg.failure_most_hard_cap_beta * 0.5 * sampling_probabilities.mean()
        ).sum()

