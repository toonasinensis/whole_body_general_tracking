from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_yaw

from whole_body_tracking.tasks.fsqtrack.mdp.commands import MultiMotionCommand
from isaaclab.envs.mdp.rewards import (
    action_rate_l2 as _action_rate_l2,
    joint_pos_limits as _joint_pos_limits,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


####################
# tracking rewards #
####################
def _get_body_indexes(
    command: MultiMotionCommand,
    body_names: list[str] | None,
) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


### root body
def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Global anchor position error reward.

    NOTE this reward is not used in the SON task by default.
    """
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Global anchor orientation error reward."""
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


### body links
def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Relative body position error reward."""
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Relative body orientation error reward."""
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(
            command.body_quat_relative_w[:, body_indexes],
            command.robot_body_quat_w[:, body_indexes],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Global body linear velocity error reward."""
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Global body angular velocity error reward."""
    command: MultiMotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


##########################
# penalty rewards        #############
# action penalty: |a_t - a_t-1|^2    #
# joint limits  : |q_j - q_jlimit|^2 #
# undesired contacts     #############
##########################
def sonic_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action rate penalty: |a_t - a_{t-1}|^2 summed over action dims."""
    return _action_rate_l2(env)


def hand_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize large hand contact force × velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    velocity_norm = torch.norm(body_vel, dim=-1)

    force_velocity_product = contact_force_norm * velocity_norm
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)
    return reward


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """Penalize large foot contact force × velocity."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)

    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    velocity_norm = torch.norm(body_vel, dim=-1)

    force_velocity_product = contact_force_norm * velocity_norm
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)
    return reward

