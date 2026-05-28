from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import TerminationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
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
        self._delay_counters = torch.zeros_like(delay_env_mask, dtype=torch.long)
        self._max_delay_steps = int(max_delay_steps)

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        extras = super().reset(env_ids=env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._delay_counters[env_ids] = 0
        return extras

    def compute(self) -> torch.Tensor:
        dones = super().compute()
        if self._max_delay_steps <= 0:
            return dones

        # Delay only task failures. Time-outs should still reset immediately.
        delay_and_terminated = self._delay_env_mask & self._terminated_buf
        self._delay_counters[delay_and_terminated] += 1

        not_ready = delay_and_terminated & (self._delay_counters < self._max_delay_steps)
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


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
