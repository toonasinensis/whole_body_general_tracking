from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.managers import TerminationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.domain_commands import domain_name_mask
from whole_body_tracking.tasks.tracking.mdp.heading_math import compute_height_filtered_contact_mask
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


class DelayedTerminationManager(TerminationManager):
    """Wrap ``TerminationManager`` and delay early terminations for a subset of envs."""

    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        self.__dict__.update(base.__dict__)
        self._delay_env_mask = delay_env_mask
        self.delayed_termination_env_mask = delay_env_mask
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self.delayed_termination_active_mask = torch.zeros_like(delay_env_mask, dtype=torch.bool)
        self._max_delay_steps = int(max_delay_steps)

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._delay_counters[env_ids] = 0
        self.delayed_termination_active_mask[env_ids] = False
        return extras

    def compute(self) -> torch.Tensor:
        dones = super().compute()
        self.delayed_termination_active_mask[:] = False
        if self._max_delay_steps <= 0:
            return dones

        # Delay only task failures. Time-outs should still reset immediately.
        delay_and_terminated = self._delay_env_mask & self._terminated_buf
        self._delay_counters[delay_and_terminated] += 1

        not_ready = delay_and_terminated & (self._delay_counters < self._max_delay_steps)
        # Expose only the currently suppressed termination window. Rewards use this
        # to switch from full tracking to recovery tracking on the same step.
        self.delayed_termination_active_mask[not_ready] = True
        self._terminated_buf[not_ready] = False

        ready = delay_and_terminated & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0

        self._delay_counters[self._delay_env_mask & ~self._terminated_buf & ~not_ready] = 0
        return self._truncated_buf | self._terminated_buf


def install_delayed_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
    use_motion_pose_range_mask: bool = True,
) -> None:
    """Startup event that installs delayed termination on the pose-range env subset."""
    del env_ids  # startup event applies globally

    if isinstance(env.termination_manager, DelayedTerminationManager):
        return

    if delay_reset_env_ratio <= 0.0 or max_delay_steps <= 0:
        return

    delay_mask = None
    if use_motion_pose_range_mask:
        delay_mask = getattr(env, "_motion_pose_range_env_mask", None)
    if delay_mask is not None:
        delay_mask = delay_mask.to(device=env.device, dtype=torch.bool).clone()
        if delay_reset_env_ratio > 0.0:
            max_count = int(env.num_envs * delay_reset_env_ratio)
            enabled = torch.where(delay_mask)[0]
            if enabled.numel() > max_count:
                delay_mask[enabled[max_count:]] = False
    else:
        ratio = delay_reset_env_ratio
        if use_motion_pose_range_mask:
            command_cfg = getattr(getattr(getattr(env, "cfg", None), "commands", None), "motion", None)
            pose_range_env_ratio = getattr(command_cfg, "pose_range_env_ratio", None)
            if pose_range_env_ratio is not None:
                ratio = min(float(delay_reset_env_ratio), float(pose_range_env_ratio))
        num_delay = int(env.num_envs * ratio)
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if num_delay > 0:
            delay_mask[:num_delay] = True

    num_delay = int(delay_mask.sum().item())
    if num_delay <= 0:
        return

    env.termination_manager = DelayedTerminationManager(
        base=env.termination_manager,
        delay_env_mask=delay_mask,
        max_delay_steps=max_delay_steps,
    )
    print(
        "[install_delayed_termination] DelayedTerminationManager installed: "
        f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
    )


def _apply_domain_termination_mask(
    env: ManagerBasedRLEnv,
    terminated: torch.Tensor,
    command_name: str,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    if not enabled_domain_names:
        return terminated
    return terminated & domain_name_mask(env, command_name, enabled_domain_names)


def _apply_min_episode_steps(
    env: ManagerBasedRLEnv,
    terminated: torch.Tensor,
    min_episode_steps: int = 0,
) -> torch.Tensor:
    if min_episode_steps <= 0:
        return terminated
    return terminated & (env.episode_length_buf >= int(min_episode_steps))


def bad_anchor_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    terminated = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def _disable_termination_on_delayed_envs(
    env: ManagerBasedRLEnv, terminated: torch.Tensor, disable_on_delayed_termination_envs: bool
) -> torch.Tensor:
    if not disable_on_delayed_termination_envs:
        return terminated

    mask = getattr(env.termination_manager, "delayed_termination_env_mask", None)
    if mask is None:
        return terminated
    return terminated & ~mask.to(device=terminated.device, dtype=torch.bool)


def bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    enabled_domain_names: tuple[str, ...] = (),
    curriculum_metric_threshold: float | None = None,
    min_threshold: float | None = None,
    threshold_update_rate: float = 0.001,
    ema_alpha: float = 0.99,
    metric_name: str = "wbc_tracking_anchor_pos",
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    threshold_value = _curriculum_anchor_z_threshold(
        env,
        command,
        base_threshold=float(threshold),
        curriculum_metric_threshold=curriculum_metric_threshold,
        min_threshold=min_threshold,
        threshold_update_rate=threshold_update_rate,
        ema_alpha=ema_alpha,
        metric_name=metric_name,
    )
    terminated = (command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold_value  # 不能低
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def _curriculum_anchor_z_threshold(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    base_threshold: float,
    curriculum_metric_threshold: float | None,
    min_threshold: float | None,
    threshold_update_rate: float,
    ema_alpha: float,
    metric_name: str,
) -> float:
    if curriculum_metric_threshold is None or min_threshold is None:
        return base_threshold

    state = getattr(env, "_wbt_anchor_z_threshold_curriculum", None)
    if state is None:
        state = {
            "threshold": float(base_threshold),
            "ema": None,
        }
        setattr(env, "_wbt_anchor_z_threshold_curriculum", state)

    metric = _read_scalar_metric(env, metric_name)
    if metric is None:
        metric = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1).mean()

    metric_value = float(metric.detach().mean().item() if isinstance(metric, torch.Tensor) else metric)
    alpha = float(max(0.0, min(1.0, ema_alpha)))
    previous_ema = state["ema"]
    ema = metric_value if previous_ema is None else alpha * float(previous_ema) + (1.0 - alpha) * metric_value
    state["ema"] = ema

    if ema < float(curriculum_metric_threshold):
        current = float(state["threshold"])
        target = float(min_threshold)
        rate = float(max(0.0, min(1.0, threshold_update_rate)))
        state["threshold"] = max(target, current + rate * (target - current))
    metric_template = command.metrics.get(
        "error_anchor_pos",
        torch.zeros(command.anchor_pos_w.shape[0], device=command.anchor_pos_w.device),
    )
    command.metrics["anchor_z_threshold_curriculum"] = torch.full_like(metric_template, float(state["threshold"]))
    command.metrics["anchor_z_threshold_metric_ema"] = torch.full_like(metric_template, float(ema))
    return float(state["threshold"])


def _read_scalar_metric(env: ManagerBasedRLEnv, metric_name: str) -> torch.Tensor | None:
    reward_manager = getattr(env, "reward_manager", None)
    episode_sums = getattr(reward_manager, "_episode_sums", None)
    if isinstance(episode_sums, dict):
        for name in (metric_name, f"{metric_name}_raw", f"{metric_name}_mean"):
            value = episode_sums.get(name)
            if value is not None:
                return value

    extras = getattr(env, "extras", None)
    if isinstance(extras, dict):
        value = extras.get(metric_name)
        if value is not None:
            return value
    return None


def bad_anchor_pos_xyz(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    terminated = torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def _compute_parkour_reach_timeout(
    *,
    motion_ids: torch.Tensor,
    local_time_steps: torch.Tensor,
    frame_end_per_env: torch.Tensor,
    max_future_step: int,
    motion_anchor_pos_w: torch.Tensor,
    time_step_end_idx: torch.Tensor,
    env_origins: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    distance_threshold: float,
    end_margin_steps: int,
) -> torch.Tensor:
    target_time_steps = (time_step_end_idx[motion_ids] - 1).clamp_min(0)
    target_anchor_pos_w = motion_anchor_pos_w[target_time_steps] + env_origins
    distance = torch.norm(robot_anchor_pos_w - target_anchor_pos_w, dim=-1)
    remaining_frames = frame_end_per_env - local_time_steps
    near_motion_end = remaining_frames <= int(max_future_step) + int(end_margin_steps)
    return near_motion_end & (distance <= float(distance_threshold))


def parkour_reach_timeout(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_threshold: float = 0.30,
    end_margin_steps: int = 1,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _compute_parkour_reach_timeout(
        motion_ids=command.motion_ids,
        local_time_steps=command.local_time_steps,
        frame_end_per_env=command.frame_end_per_env,
        max_future_step=int(command.cfg.max_future_step),
        motion_anchor_pos_w=command.motion.anchor_pos_w,
        time_step_end_idx=command.motion.time_step_end_idx,
        env_origins=env.scene.env_origins,
        robot_anchor_pos_w=command.robot_anchor_pos_w,
        distance_threshold=distance_threshold,
        end_margin_steps=end_margin_steps,
    )


def bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    terminated = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination_envs: bool = False,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    terminated = torch.any(error > threshold, dim=-1)
    terminated = _disable_termination_on_delayed_envs(env, terminated, disable_on_delayed_termination_envs)
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination_envs: bool = False,
    enabled_domain_names: tuple[str, ...] = (),
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    terminated = torch.any(error > threshold, dim=-1)
    terminated = _disable_termination_on_delayed_envs(env, terminated, disable_on_delayed_termination_envs)
    return _apply_domain_termination_mask(env, terminated, command_name, enabled_domain_names)


def illegal_contact_on_domains(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    domain_command_name: str = "motion",
    enabled_domain_names: tuple[str, ...] = ("flat_velocity",),
    min_episode_steps: int = 0,
) -> torch.Tensor:
    terminated = velocity_mdp.illegal_contact(env, threshold, sensor_cfg)
    terminated = _apply_min_episode_steps(env, terminated, min_episode_steps)
    return _apply_domain_termination_mask(env, terminated, domain_command_name, enabled_domain_names)


def root_height_below_on_domains(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.45,
    domain_command_name: str = "motion",
    enabled_domain_names: tuple[str, ...] = ("flat_velocity",),
    min_episode_steps: int = 0,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terminated = asset.data.root_pos_w[:, 2] < float(threshold)
    terminated = _apply_min_episode_steps(env, terminated, min_episode_steps)
    return _apply_domain_termination_mask(env, terminated, domain_command_name, enabled_domain_names)


def root_height_below_desired(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.3,
    target_height: float | None = None,
    min_episode_steps: int = 0,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    desired_height = (
        asset.data.default_root_state[:, 2]
        if target_height is None
        else torch.full_like(asset.data.root_pos_w[:, 2], float(target_height))
    )
    terminated = asset.data.root_pos_w[:, 2] < desired_height - float(margin)
    return _apply_min_episode_steps(env, terminated, min_episode_steps)


def projected_gravity_xy_on_domains(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.85,
    domain_command_name: str = "motion",
    enabled_domain_names: tuple[str, ...] = ("flat_velocity",),
    min_episode_steps: int = 0,
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terminated = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1) > float(threshold)
    terminated = _apply_min_episode_steps(env, terminated, min_episode_steps)
    return _apply_domain_termination_mask(env, terminated, domain_command_name, enabled_domain_names)


def height_filtered_illegal_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    max_body_height: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    body_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    is_contact = compute_height_filtered_contact_mask(force_norm, body_z, threshold, max_body_height)
    return torch.any(is_contact, dim=1)
