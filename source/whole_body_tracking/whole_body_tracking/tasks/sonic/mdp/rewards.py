from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_yaw
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
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
    command: MotionCommand, 
    body_names: list[str] | None
) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

### root body
def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, std: float
) -> torch.Tensor:
    """ Global anchor position error reward. 
    NOTE this reward is not used in the SONIC task.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, std: float
) -> torch.Tensor:
    """ Global anchor orientation error reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


### body links
def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float, 
    body_names: list[str] | None = None
) -> torch.Tensor:
    """ Relative body position error reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float, 
    body_names: list[str] | None = None
) -> torch.Tensor:
    """ Relative body orientation error reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float, 
    body_names: list[str] | None = None
) -> torch.Tensor:
    """ Global body linear velocity error reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    std: float, 
    body_names: list[str] | None = None
) -> torch.Tensor:
    """ Global body angular velocity error reward.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


##########################
# penalty rewards        #############
# action penalty: |a_t - a_t-1|^2    #
# joint limits  : |q_j - q_jlimit|^2 #
# undesired contacts     #############
##########################
def sonic_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ action rate penalty: |a_t - a_{t-1}|^2 summed over action dims
    """
    return _action_rate_l2(env)
# joint limits have been realised in isaaclab's joint pos limits

def hand_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """
    当手与物体接触力x速度大于阈值时，如果速度过大就惩罚。
    """
    # 获取接触传感器对象
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触力历史数据，形状为 (num_envs, history_frames, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 使用最新的一帧数据，形状变化: [num_envs, num_feet, 3] -> [num_envs, num_feet]
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    # 获取脚的水平速度（x和y分量）
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    # 计算水平速度的模长
    velocity_norm = torch.norm(body_vel, dim=-1)  # (num_envs, num_feet)
    # 添加接触力与速度乘积的惩罚项
    force_velocity_product = contact_force_norm * velocity_norm  # (num_envs, num_feet)
    # 对超过阈值的部分进行惩罚，然后对所有脚求和
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  # (num_envs, num_feet)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)  # (num_envs,)
    return reward

def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_velocity_threshold: float = 100.0,
) -> torch.Tensor:
    """
    当脚与地面接触力x速度大于阈值时，如果速度过大就惩罚。
    """
    # 获取接触传感器对象
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触力历史数据，形状为 (num_envs, history_frames, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 使用最新的一帧数据，形状变化: [num_envs, num_feet, 3] -> [num_envs, num_feet]
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    # 获取脚的水平速度（x和y分量）
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    # 计算水平速度的模长
    velocity_norm = torch.norm(body_vel, dim=-1)  # (num_envs, num_feet)
    # 添加接触力与速度乘积的惩罚项
    force_velocity_product = contact_force_norm * velocity_norm  # (num_envs, num_feet)
    # 对超过阈值的部分进行惩罚，然后对所有脚求和
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  # (num_envs, num_feet)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)  # (num_envs,)
    return reward
