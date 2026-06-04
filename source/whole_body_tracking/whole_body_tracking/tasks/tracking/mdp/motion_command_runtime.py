from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, sample_uniform, yaw_quat


class MotionDataSource(Protocol):
    time_step_total: int
    time_step_start_idx: torch.Tensor
    time_step_end_idx: torch.Tensor

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Map global timestamps to motion ids."""


@dataclass
class MotionSelection:
    motion_ids: torch.Tensor
    local_time_steps: torch.Tensor
    frame_end: torch.Tensor


class MotionCommandTimeline:
    """Track per-environment motion ids and local frame cursors."""

    def __init__(self, num_envs: int, future_step_offsets: torch.Tensor, device: str):
        self.device = device
        self.local_time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.motion_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.frame_end_per_env = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.eval_cycle_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.future_step_offsets = future_step_offsets

    @property
    def num_future_frames(self) -> int:
        return int(self.future_step_offsets.numel())

    def global_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        return motion_source.time_step_start_idx[self.motion_ids] + self.local_time_steps

    def motion_start_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        return motion_source.time_step_start_idx[self.motion_ids]

    def motion_num_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        start = motion_source.time_step_start_idx[self.motion_ids]
        end = motion_source.time_step_end_idx[self.motion_ids]
        return end - start

    def future_motion_ids(self) -> torch.Tensor:
        return self.motion_ids[:, None].expand(-1, self.num_future_frames).reshape(-1)

    def future_time_steps(self, motion_source: MotionDataSource) -> torch.Tensor:
        start = self.motion_start_time_steps(motion_source)
        local_max = (self.motion_num_steps(motion_source) - 1).clamp(min=0)
        future_local = torch.clip(
            self.local_time_steps[:, None] + self.future_step_offsets[None, :],
            max=local_max[:, None],
        )
        return (start[:, None] + future_local).long()

    def selection_from_global_timestamps(
        self,
        motion_source: MotionDataSource,
        timestamps: torch.Tensor,
    ) -> MotionSelection:
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
        return torch.where(self.local_time_steps >= self.frame_end_per_env - max_future_step)[0]


class MotionReferenceCache:
    """Cache reference poses aligned into the robot's anchor XY-yaw frame."""

    def __init__(self, num_envs: int, body_count: int, device: str):
        self.body_pos_relative_w = torch.zeros(num_envs, body_count, 3, device=device)
        self.body_quat_relative_w = torch.zeros(num_envs, body_count, 4, device=device)
        self.body_quat_relative_w[:, :, 0] = 1.0

    def refresh(
        self,
        *,
        anchor_pos_w: torch.Tensor,
        anchor_quat_w: torch.Tensor,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> None:
        body_count = body_pos_w.shape[1]
        anchor_pos_w_repeat = anchor_pos_w[:, None, :].repeat(1, body_count, 1)
        anchor_quat_w_repeat = anchor_quat_w[:, None, :].repeat(1, body_count, 1)
        robot_anchor_pos_w_repeat = robot_anchor_pos_w[:, None, :].repeat(1, body_count, 1)
        robot_anchor_quat_w_repeat = robot_anchor_quat_w[:, None, :].repeat(1, body_count, 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w[:] = quat_mul(delta_ori_w, body_quat_w)
        self.body_pos_relative_w[:] = delta_pos_w + quat_apply(delta_ori_w, body_pos_w - anchor_pos_w_repeat)


class MotionCommandResetter:
    """Apply sampled reference motion states back to the simulator on reset/resample."""

    def __init__(self, cfg, robot: Articulation, device: str):
        self.cfg = cfg
        self.robot = robot
        self.device = device

    def apply(
        self,
        env_ids: Sequence[int],
        *,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> None:
        if len(env_ids) == 0:
            return

        root_pos = body_pos_w[:, 0].clone()
        root_ori = body_quat_w[:, 0].clone()
        root_lin_vel = body_lin_vel_w[:, 0].clone()
        root_ang_vel = body_ang_vel_w[:, 0].clone()

        pose_ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        pose_noise = sample_uniform(pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += pose_noise[:, 0:3]

        velocity_ranges = torch.tensor(
            [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=self.device,
        )
        velocity_noise = sample_uniform(
            velocity_ranges[:, 0], velocity_ranges[:, 1], (len(env_ids), 6), device=self.device
        )
        root_lin_vel[env_ids] += velocity_noise[:, :3]
        root_ang_vel[env_ids] += velocity_noise[:, 3:]

        joint_pos = joint_pos.clone()
        joint_vel = joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_vel_limits = self.robot.data.joint_vel_limits[env_ids]
        max_ang_vel_root = 20.0

        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids],
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )
        joint_vel[env_ids] = torch.clip(joint_vel[env_ids], -joint_vel_limits[:, :], joint_vel_limits[:, :])
        root_ang_vel[env_ids] = torch.clip(root_ang_vel[env_ids], -max_ang_vel_root, max_ang_vel_root)

        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )


class MotionCommandDebugVisualizer:
    """Own all debug-visualization markers and rendering details for MotionCommand."""

    def __init__(self, cfg, command, device: str):
        self.cfg = cfg
        self.command = command
        self.device = device

        self.current_anchor_visualizer: VisualizationMarkers | None = None
        self.goal_anchor_visualizer: VisualizationMarkers | None = None
        self.future_anchor_visualizer: VisualizationMarkers | None = None
        self.current_anchor_lin_vel_visualizer: VisualizationMarkers | None = None
        self.goal_anchor_lin_vel_visualizer: VisualizationMarkers | None = None
        self.current_body_visualizers: list[VisualizationMarkers] = []
        self.goal_body_visualizers: list[VisualizationMarkers] = []

    def set_enabled(self, debug_vis: bool) -> None:
        if debug_vis:
            self._ensure_initialized()
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.future_anchor_visualizer.set_visibility(True)
            if self.cfg.debug_anchor_speed:
                self.current_anchor_lin_vel_visualizer.set_visibility(True)
                self.goal_anchor_lin_vel_visualizer.set_visibility(True)
            for visualizer in self.current_body_visualizers:
                visualizer.set_visibility(True)
            for visualizer in self.goal_body_visualizers:
                visualizer.set_visibility(True)
            return

        if self.current_anchor_visualizer is None:
            return

        self.current_anchor_visualizer.set_visibility(False)
        self.goal_anchor_visualizer.set_visibility(False)
        self.future_anchor_visualizer.set_visibility(False)
        if self.cfg.debug_anchor_speed:
            self.current_anchor_lin_vel_visualizer.set_visibility(False)
            self.goal_anchor_lin_vel_visualizer.set_visibility(False)
        for visualizer in self.current_body_visualizers:
            visualizer.set_visibility(False)
        for visualizer in self.goal_body_visualizers:
            visualizer.set_visibility(False)

    def render(self) -> None:
        if not self.command.robot.is_initialized or self.future_anchor_visualizer is None:
            return

        self.current_anchor_visualizer.visualize(
            self.command.robot_anchor_pos_w,
            self.command.robot_anchor_quat_w,
        )
        self.goal_anchor_visualizer.visualize(
            self.command.anchor_pos_w,
            self.command.anchor_quat_w,
        )
        self.future_anchor_visualizer.visualize(
            self.command.anchor_pos_w_future.view(-1, 3),
            self.command.anchor_quat_w_future.view(-1, 4),
        )

        if self.cfg.debug_anchor_speed:
            current_lin_vel_scale, current_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.command.robot_anchor_lin_vel_w,
                self.current_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )
            goal_lin_vel_scale, goal_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.command.anchor_lin_vel_w,
                self.goal_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )

            self.current_anchor_lin_vel_visualizer.visualize(
                self.command.robot_anchor_pos_w,
                current_lin_vel_quat,
                current_lin_vel_scale,
            )
            self.goal_anchor_lin_vel_visualizer.visualize(
                self.command.anchor_pos_w,
                goal_lin_vel_quat,
                goal_lin_vel_scale,
            )

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.command.robot_body_pos_w[:, i],
                self.command.robot_body_quat_w[:, i],
            )
            self.goal_body_visualizers[i].visualize(
                self.command.body_pos_relative_w[:, i],
                self.command.body_quat_relative_w[:, i],
            )

    def _ensure_initialized(self) -> None:
        if self.current_anchor_visualizer is not None:
            return

        self.current_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
        )
        self.goal_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
        )
        self.future_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/future/anchor")
        )

        if self.cfg.debug_anchor_speed:
            self.current_anchor_lin_vel_visualizer = VisualizationMarkers(
                self.cfg.current_anchor_lin_vel_visualizer_cfg
            )
            self.goal_anchor_lin_vel_visualizer = VisualizationMarkers(
                self.cfg.goal_anchor_lin_vel_visualizer_cfg
            )

        self.current_body_visualizers = []
        self.goal_body_visualizers = []
        for name in self.cfg.body_names:
            self.current_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                )
            )
            self.goal_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                )
            )

    def _resolve_velocity_to_arrow(
        self,
        velocity_w: torch.Tensor,
        default_scale: tuple[float, float, float],
        speed_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 3D world velocity to arrow scale and world quaternion."""
        speed = torch.linalg.norm(velocity_w, dim=1)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= speed * speed_scale

        eps = 1.0e-8
        direction = velocity_w / speed.unsqueeze(-1).clamp(min=eps)
        x_axis = torch.zeros_like(direction)
        x_axis[:, 0] = 1.0

        cross = torch.cross(x_axis, direction, dim=1)
        dot = torch.sum(x_axis * direction, dim=1).clamp(-1.0, 1.0)

        w = torch.sqrt(((1.0 + dot).clamp(min=0.0)) * 0.5)
        xyz = cross / (2.0 * w.unsqueeze(-1).clamp(min=eps))
        arrow_quat_w = torch.cat([w.unsqueeze(-1), xyz], dim=1)

        opposite = dot < (-1.0 + 1.0e-6)
        if torch.any(opposite):
            arrow_quat_w[opposite] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

        stationary = speed < 1.0e-6
        if torch.any(stationary):
            arrow_quat_w[stationary] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        arrow_quat_w = torch.nn.functional.normalize(arrow_quat_w, dim=1)
        return arrow_scale, arrow_quat_w


class AdaptiveMotionSampler:
    """Maintain adaptive-sampling state and produce motion selections."""

    def __init__(self, cfg, device: str):
        self.cfg = cfg
        self.device = device
        self.bin_count = 0
        self.bin_failed_count = torch.zeros(0, dtype=torch.float, device=device)
        self._current_bin_failed = torch.zeros(0, dtype=torch.float, device=device)
        self.kernel = torch.zeros(0, dtype=torch.float, device=device)
        self.success_motion = torch.zeros(0, dtype=torch.float32, device=device)
        self._last_export_step = -1

    def reset_for_motion_source(self, motion_source: MotionDataSource, *, decimation: int, sim_dt: float) -> None:
        self.bin_count = int(motion_source.time_step_total // (1 / (decimation * sim_dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()
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
        if len(env_ids) == 0:
            raise ValueError("env_ids must not be empty for adaptive sampling")

        episode_failed = terminated[env_ids]
        if allow_failure_accounting and torch.any(episode_failed):
            global_ts = torch.clamp(timeline.global_time_steps(motion_source) - 1, min=0, max=int(motion_source.time_step_total) - 1)
            current_bin_index = torch.clamp(
                (global_ts * self.bin_count) // max(motion_source.time_step_total, 1),
                0,
                self.bin_count - 1,
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            # IsaacLab may invoke `_resample_command()` multiple times within one control step,
            # so we accumulate failures until `step_post_update()` applies the EMA update.
            self._current_bin_failed += torch.bincount(fail_bins, minlength=self.bin_count)

        sampling_probabilities = self._compute_sampling_probabilities(metrics)
        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)
        global_ts = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (motion_source.time_step_total - 1)
        ).long()

        selection = timeline.selection_from_global_timestamps(motion_source, global_ts)
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

class MotionSelectionPolicy(Protocol):
    fixed_eval_motion_ids: torch.Tensor | None

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        """Reset policy state after motion-source reload."""

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        """Produce a selection for the requested environments."""

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        """Update policy state after each command step."""

    @property
    def counts_eval_cycles(self) -> bool:
        """Whether the policy represents deterministic per-clip eval restarts."""


class FixedEvalMotionSelectionPolicy:
    """Deterministically bind each env to one motion id during evaluation."""

    def __init__(self, num_envs: int, device: str):
        self.num_envs = num_envs
        self.device = device
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return True

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        motion_num = int(getattr(motion_source, "motion_num", 0) or 0)
        if motion_num <= 0:
            raise RuntimeError("Motion source has no motions loaded; cannot set up fixed eval motion mapping.")
        if int(self.num_envs) != motion_num:
            raise ValueError(
                f"fixed_eval_motion_ids requires num_envs == motion_num, got num_envs={int(self.num_envs)} motion_num={motion_num}."
            )

        self.fixed_eval_motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        selection = timeline.selection_from_motion_ids(motion_source, self.fixed_eval_motion_ids)
        timeline.apply_selection(env_ids, selection)
        timeline.eval_cycle_count.zero_()

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        if self.fixed_eval_motion_ids is None:
            raise RuntimeError("Fixed eval policy is not bound. Call bind_motion_source() first.")
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        motion_ids = self.fixed_eval_motion_ids[env_ids_t]
        return timeline.selection_from_motion_ids(motion_source, motion_ids)

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        return


class AdaptiveMotionSelectionPolicy:
    """Policy adapter that exposes the adaptive sampler through a common selection interface."""

    def __init__(self, sampler: AdaptiveMotionSampler):
        self.sampler = sampler
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return False

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        self.fixed_eval_motion_ids = None
        self.sampler.reset_for_motion_source(motion_source, decimation=decimation, sim_dt=sim_dt)

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        return self.sampler.sample_selection(
            env_ids,
            motion_source=motion_source,
            timeline=timeline,
            terminated=terminated,
            metrics=metrics,
            allow_failure_accounting=allow_failure_accounting,
        )

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        self.sampler.step_post_update(
            motion_source=motion_source,
            command_step_count=command_step_count,
        )


class UnsupportedMotionSelectionPolicy:
    """Preserve the previous not-implemented behavior for unsupported configs."""

    def __init__(self):
        self.fixed_eval_motion_ids: torch.Tensor | None = None

    @property
    def counts_eval_cycles(self) -> bool:
        return False

    def bind_motion_source(
        self,
        motion_source: MotionDataSource,
        *,
        timeline: MotionCommandTimeline,
        decimation: int,
        sim_dt: float,
    ) -> None:
        return

    def select(
        self,
        env_ids: Sequence[int],
        *,
        motion_source: MotionDataSource,
        timeline: MotionCommandTimeline,
        terminated: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        allow_failure_accounting: bool = True,
    ) -> MotionSelection:
        raise NotImplementedError("Only fixed-eval and adaptive motion selection policies are implemented.")

    def step_post_update(
        self,
        *,
        motion_source: MotionDataSource,
        command_step_count: int,
    ) -> None:
        return


def create_motion_selection_policy(cfg, *, num_envs: int, device: str) -> MotionSelectionPolicy:
    if bool(getattr(cfg, "fixed_eval_motion_ids", False)) and bool(getattr(cfg, "eval_mode", False)):
        return FixedEvalMotionSelectionPolicy(num_envs=num_envs, device=device)
    if bool(getattr(cfg, "adaptive_sample", False)):
        return AdaptiveMotionSelectionPolicy(AdaptiveMotionSampler(cfg, device))
    return UnsupportedMotionSelectionPolicy()
