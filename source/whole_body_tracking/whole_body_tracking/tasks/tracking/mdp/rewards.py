from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _zero_delayed_termination_rewards(
    env: ManagerBasedRLEnv, reward: torch.Tensor, disable_on_delayed_termination: bool
) -> torch.Tensor:
    if not disable_on_delayed_termination:
        return reward

    mask = getattr(env.termination_manager, "delayed_termination_active_mask", None)
    if mask is None:
        return reward
    return torch.where(mask.to(device=reward.device, dtype=torch.bool), torch.zeros_like(reward), reward)


def _zero_non_delayed_termination_rewards(
    env: ManagerBasedRLEnv, reward: torch.Tensor, enable_on_delayed_termination: bool
) -> torch.Tensor:
    if not enable_on_delayed_termination:
        return reward

    mask = getattr(env.termination_manager, "delayed_termination_active_mask", None)
    if mask is None:
        return torch.zeros_like(reward)
    return torch.where(mask.to(device=reward.device, dtype=torch.bool), reward, torch.zeros_like(reward))


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, disable_on_delayed_termination: bool = False
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    reward = torch.exp(-error / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def motion_global_anchor_position_z_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2]  - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)

def motion_global_torso_position_z_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    enable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    left_arm_index = command.cfg.body_names.index("zarm_l2_link")
    right_arm_index = command.cfg.body_names.index("zarm_r2_link")
    robot_arm_height = 0.5 * (
        command.robot_body_pos_w[:, left_arm_index, 2] + command.robot_body_pos_w[:, right_arm_index, 2]
    )
    motion_arm_height = 0.5 * (
        command.body_pos_w[:, left_arm_index, 2] + command.body_pos_w[:, right_arm_index, 2]
    )
    motion_center_height = 0.5 * (command.anchor_pos_w[:, 2] + motion_arm_height)
    robot_center_height = 0.5 * (command.robot_anchor_pos_w[:, 2] + robot_arm_height)
    error = torch.square(motion_center_height - robot_center_height)
    reward = torch.exp(-error / std**2)
    return _zero_non_delayed_termination_rewards(env, reward, enable_on_delayed_termination)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
