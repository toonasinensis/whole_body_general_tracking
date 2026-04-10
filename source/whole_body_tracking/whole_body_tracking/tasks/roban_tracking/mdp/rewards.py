from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_rotate_inverse
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_yaw
from whole_body_tracking.tasks.roban_tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def motion_feet_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    # return torch.exp(-error.mean(-1) / std**2)

    command_err = \
    command.body_pos_relative_w[:, body_indexes] \
    - command.robot_body_pos_w[:, body_indexes]
    square = torch.square(
        command.body_pos_relative_w[:, body_indexes] 
        - command.robot_body_pos_w[:, body_indexes])
    sum = torch.sum(square, dim=-1)
    reward = torch.exp(-error.mean(-1) / std**2)
    return reward

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel

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
    当脚接触地面且垂直速度向下（vz < 0）时，惩罚落地冲击力过大的行为。
    抬脚时（vz > 0）不惩罚，避免误惩罚离地动作。
    """
    # 获取接触传感器对象
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触力历史数据，形状为 (num_envs, history_frames, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 使用最新的一帧数据，形状变化: [num_envs, num_feet, 3] -> [num_envs, num_feet]
    contact_force_norm = torch.norm(net_contact_forces[:, -1, sensor_cfg.body_ids], dim=-1)
    # 获取脚的垂直速度（z分量），落地时为负值
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    # 只取向下的速度（vz < 0），抬脚时（vz > 0）置为0不惩罚
    downward_vel = torch.clamp(-body_vel_z, min=0.0)  # (num_envs, num_feet)
    # 添加接触力与向下速度乘积的惩罚项
    force_velocity_product = contact_force_norm * downward_vel  # (num_envs, num_feet)
    # 对超过阈值的部分进行惩罚，然后对所有脚求和
    violation = torch.clamp(force_velocity_product - force_velocity_threshold, min=0.0)  # (num_envs, num_feet)
    reward = torch.sum(violation / force_velocity_threshold, dim=1)  # (num_envs,)
    return reward

def motion_default_pose(
    env: ManagerBasedRLEnv,
    command_name: str,  # 必需参数，必须在前
    sigma = float,  # σ: 敏感度参数 (sensitivity)
    delta = float,   # δ: 容差参数 (tolerance)
    start_frames = int,  # 前x帧
    end_frames = int,    # 后x帧
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # 可选参数，必须在后
) -> torch.Tensor:
    r"""奖励在轨迹开始和结束时保持默认姿态
    使用指数核函数 (Exponential Kernel Function) 计算奖励:
    .. math:: K(x, \sigma, \delta) = \exp\left(-\left(\frac{\max(0, \|x\| - \delta)}{\sigma}\right)^2\right)
    其中:
    - :math:`x` 是关节位置误差向量 (current_joint_pos - default_joint_pos)
    - :math:`\|x\|` 是误差向量的 L2 范数 (欧几里得距离)
    - :math:`\sigma` (sigma) 是敏感度参数，控制奖励衰减的速度
    - :math:`\delta` (delta) 是容差参数，允许的误差阈值
    """
    joint_pos = env.command_manager.get_command(command_name)
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.asset_name]
    asset: Articulation = env.scene[asset_cfg.name]
    # 获取当前时间步和总时间步
    time_steps = command.time_steps
    total_steps = command.motion.time_step_total
    # 判断是否在起始或结束区域
    in_start = time_steps < start_frames
    in_end = time_steps > (total_steps - 1 - end_frames)
    in_boundary = in_start | in_end
    # 计算关节位置与默认位置的误差向量
    default_joint_pos = asset.data.default_joint_pos_nominal
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error_vec = current_joint_pos - default_joint_pos
    # 计算误差向量的 L2 范数: ||x||
    error_norm = torch.linalg.vector_norm(error_vec, dim=-1)
    # 应用容差: max(0, ||x|| - δ)
    clipped_error = torch.clamp(error_norm - delta, min=0.0)
    # 应用敏感度并平方: ((max(0, ||x|| - δ)) / σ)^2
    scaled_error_squared = torch.square(clipped_error / sigma)
    # 指数核函数: exp(-((max(0, ||x|| - δ)) / σ)^2)
    reward = torch.exp(-scaled_error_squared)
    # 只在边界区域给奖励
    reward = reward * in_boundary.float()
    return reward

def com_balance_when_stand(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    std: float = 0.05,
) -> torch.Tensor:
    """根据支撑情况奖励质心平衡
    
    - 单脚支撑：质心应该在支撑脚上方
    - 双脚支撑：质心应该在双脚中间
    
    Args:
        env: 环境
        sensor_cfg: 接触传感器配置
        command_name: 命令名称
        asset_cfg: 资产配置
        std: 标准差（基准值）
        contact_threshold: 接触力阈值（N）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 计算系统质心
    body_com_pos_w = asset.data.body_com_pos_w
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True)
    system_com = (body_com_pos_w * body_masses.unsqueeze(-1)).sum(dim=1) / total_mass  # (N, 3)
    
    # 获取接触信息
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]  # (N, history, 2, 3)
    
    # 计算左右脚的最大接触力
    # contact_forces[:, :, :, :].norm(dim=-1) → (N, history, 2) - 每只脚在各时间步的接触力大小
    # .max(dim=1)[0] → (N, 2) - 每只脚的历史最大接触力
    feet_max_forces = contact_forces.norm(dim=-1).max(dim=1)[0]  # (N, 2)
    
    # 判断单脚接触（使用接触力阈值）
    left_foot_contact = (feet_max_forces[:, 0] > total_mass.squeeze(-1) * 9.8 * 0.6)   # (N,)
    right_foot_contact = (feet_max_forces[:, 1] > total_mass.squeeze(-1) * 9.8 * 0.6)  # (N,)
    
    # 计算双脚总接触力
    total_contact_force = feet_max_forces[:, 0] + feet_max_forces[:, 1]  # (N,)
    # 判断是否为双脚支撑：双脚总接触力 > 0.8 * 全身重力
    both_feet_support = (total_contact_force > total_mass.squeeze(-1) * 9.8 * 0.8)  # (N,)
    
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # (N, 2, 3)
    waist_quat = asset.data.body_quat_w[:, 0, :]
    left_foot_pos = feet_pos[:, 0, :]  # (N, 3)
    right_foot_pos = feet_pos[:, 1, :]  # (N, 3)
    
    # 在 body frame 中的脚掌前向偏移
    foot_offset_local = torch.tensor([0.02, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    # 根据腰部姿态旋转到世界系
    foot_offset_world = quat_apply_yaw(waist_quat, foot_offset_local)
    left_foot_center = left_foot_pos + foot_offset_world  # (N, 3)
    right_foot_center = right_foot_pos + foot_offset_world 

    # 判断支撑情况（基于力的判断）
    only_left_contact = left_foot_contact & (~right_foot_contact) & (~both_feet_support)  # (N,)
    only_right_contact = (~left_foot_contact) & right_foot_contact & (~both_feet_support)  # (N,)
    no_contact = (~left_foot_contact) & (~right_foot_contact) & (~both_feet_support)  # (N,)
    has_contact = ~no_contact  # 至少有一只脚接触地面
    
    # 计算不同支撑情况下的目标质心位置（向量化）
    # 双脚中间目标位置
    both_feet_target = (left_foot_center + right_foot_center) / 2.0  # (N, 3)
    
    # 根据支撑情况选择目标位置（优先级：双脚支撑 > 单脚 > 默认）
    # 使用 where 进行向量化条件选择，避免 if 分支
    target_com = torch.where(
        both_feet_support.unsqueeze(-1),  # (N, 1)
        both_feet_target,  # 双脚支撑 → 双脚中间
        torch.where(
            only_left_contact.unsqueeze(-1),  # (N, 1)
            left_foot_center,  # 单左脚支撑 → 左脚中心
            torch.where(
                only_right_contact.unsqueeze(-1),  # (N, 1)
                right_foot_center,  # 单右脚支撑 → 右脚中心
                both_feet_target  # 默认（无接触） → 双脚中间
            )
        )
    )
    
    # 计算质心与目标位置的水平偏移（只考虑 x 和 y）
    com_offset = torch.norm(system_com[:, :2] - target_com[:, :2], dim=1)  # (N,)
    
    # 计算高斯奖励
    reward = torch.exp(-torch.square(1.5 * com_offset / std))
    
    # 只在站立阶段给奖励（进入后期 & 有接触）
    reward = torch.where(has_contact, reward, torch.zeros_like(reward))
    
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    # # 获取接触信息
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # (N, history, 2, 3)
    
    # left_foot_contact = contact_forces[0] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # right_foot_contact = contact_forces[1] > torch.sum(contact_forces, dim=1) / 2 * 0.9
    # both_feet_support = left_foot_contact & right_foot_contact
    
    # left_foot_air_time = torch.where(left_foot_contact & (~right_foot_contact), left_foot_air_time + env.step_dt, 0.0)
    
    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    # 计算总接触力
    total_contact_force = torch.sum(contact_forces, dim=1)
    body_weight = total_mass * 9.8
    # 判断总接触力是否在合理范围内（0.8~1.2倍体重）
    stand_mask = (total_contact_force > body_weight * 0.8) & (total_contact_force < body_weight * 1.2)
    # left_foot_contact = contact_forces[0] > contact_forces[1] * 2.0
    # right_foot_contact = contact_forces[1] > contact_forces[0] * 2.0
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    command: MotionCommand = env.command_manager.get_term(command_name)
    # mask = torch.where(command.start_time < 50, 0, 1)
    # mask = torch.where(command.out_time > 0, 1, mask)
    reward *= stand_mask # * mask
    return reward

def body_slide_vel(
        env, sensor_cfg: SceneEntityCfg, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        contact_threshold: float = 110.0
    ) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > contact_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward_vel = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward_vel

def penalize_feet_dragging(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,  # 判断是否接触地面的阈值（N）
    drag_threshold: float = 5.0,      # 判断是否磨蹭的阈值（N）
    threshold: float = 0.2,
) -> torch.Tensor:
    """惩罚单脚支撑时摆动脚磨蹭地面的行为。
    
    在单脚支撑阶段，摆动脚应该完全离地。如果摆动脚仍有小的接触力，
    说明在拖地/磨蹭，这会导致：
    1. 能量浪费
    2. 增加摩擦阻力
    3. 不自然的步态
    
    Args:
        env: 强化学习环境对象
        sensor_cfg: 接触传感器配置（监控双脚）
        asset_cfg: 机器人资产配置
        contact_threshold: 判断脚是否接触地面的力阈值（N），高于此值认为是支撑脚
        drag_threshold: 判断是否磨蹭的力阈值（N），低于contact_threshold但高于此值认为是磨蹭
        
    Returns:
        每个环境的惩罚值 (num_envs,)，值越大惩罚越重
        
    逻辑流程:
        1. 判断是否处于单脚支撑状态（只有一只脚接触力 > contact_threshold）
        2. 识别支撑脚和摆动脚
        3. 检查摆动脚的接触力是否大于 drag_threshold
        4. 如果摆动脚磨蹭地面，给予惩罚
    """
    # 获取接触力数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 获取最新一帧的接触力 Z 分量（垂直方向）
    # 形状: (num_envs, num_feet) - 假设 sensor_cfg.body_ids 包含 [左脚, 右脚]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]
    
    # 获取机器人总质量（用于判断是否真正支撑）
    asset: Articulation = env.scene[asset_cfg.name]
    total_mass = asset.root_physx_view.get_masses().to(env.device).sum(dim=1, keepdim=True).squeeze(-1)
    body_weight = total_mass * 9.8
    
    # 判断每只脚是否接触地面（支撑脚）
    # 形状: (num_envs, 2) - [左脚是否接触, 右脚是否接触]
    foot_in_contact = contact_forces_z > contact_threshold
    
    # 判断是否处于单脚支撑状态（恰好一只脚接触）
    # 形状: (num_envs,)
    num_feet_in_contact = torch.sum(foot_in_contact.int(), dim=1)
    single_foot_support = num_feet_in_contact == 1
    
    # 分离左右脚的接触力
    left_foot_force = contact_forces_z[:, 0]
    right_foot_force = contact_forces_z[:, 1]
    
    # 判断哪只脚是支撑脚
    left_is_support = contact_forces_z[:, 0] >  contact_forces_z[:, 1] * 2 # (num_envs,)
    # right_is_support = contact_forces_z[:, 1] > contact_forces_z[:, 0] * 2
    
    # 获取摆动脚的接触力
    # 如果左脚支撑，则检查右脚力；如果右脚支撑，则检查左脚力
    swing_foot_force = torch.where(
        left_is_support,
        right_foot_force,  # 左脚支撑，右脚是摆动脚
        left_foot_force    # 右脚支撑，左脚是摆动脚
    )
    
    # 计算磨蹭惩罚：摆动脚的接触力超过 drag_threshold 的部分
    # 如果摆动脚力 > drag_threshold，说明在磨蹭地面
    drag_penalty = torch.clamp(swing_foot_force - drag_threshold, min=0.0)
    
    # 判断是否双脚都不接触（腾空状态）
    no_foot_contact = num_feet_in_contact == 0
    
    # 腾空时两只脚的任何接触力都应该被惩罚
    both_feet_drag_penalty = (
        torch.clamp(left_foot_force - drag_threshold, min=0.0) + 
        torch.clamp(right_foot_force - drag_threshold, min=0.0)
    )
    
    # 单脚支撑时惩罚摆动脚拖地，腾空时惩罚两只脚的接触力
    reward = torch.where(
        single_foot_support,
        drag_penalty,  # 单脚支撑时惩罚摆动脚
        torch.where(
            no_foot_contact,
            both_feet_drag_penalty,  # 腾空时惩罚两只脚的接触力
            torch.zeros_like(drag_penalty)  # 双脚支撑时不惩罚
        )
    )
    reward = torch.clamp(reward, max=threshold)

    return reward

def penalize_feet_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    swing_force_ratio: float = 0.15,  # 摆动脚接触力占体重的比例阈值
    support_force_ratio: float = 0.5,  # 支撑脚接触力占体重的比例阈值
    single_support_target: float = 0.11,  # 摆动脚期望抬离高度（m）
    single_support_scale: float = 5.0,  # 惩罚斜率
) -> torch.Tensor:
    """惩罚单脚支撑时摆动脚高度不足。
    
    只使用接触力判断单脚支撑状态，不依赖动捕数据。
    当一只脚力量很小（< 体重 * swing_force_ratio）且另一只脚力量足够
    （> 体重 * support_force_ratio）时，认为是单脚支撑，惩罚摆动脚高度不足。
    
    对于29kg机器人（体重约284N）：
    - swing_force_ratio=0.15 → 摆动脚阈值约43N
    - support_force_ratio=0.5 → 支撑脚阈值约142N
    
    Args:
        env: 强化学习环境对象
        sensor_cfg: 接触传感器配置（监控双脚）
        asset_cfg: 机器人资产配置
        swing_force_ratio: 摆动脚接触力占体重的比例阈值，低于此值认为是摆动脚
        support_force_ratio: 支撑脚接触力占体重的比例阈值，高于此值认为是支撑脚
        single_support_target: 单脚支撑时摆动脚期望抬离高度（m）
        single_support_scale: 单脚支撑高度不足的惩罚斜率
        
    Returns:
        每个环境的高度惩罚值 (num_envs,)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_masses = asset.root_physx_view.get_masses().to(env.device)
    total_mass = body_masses.sum(dim=1, keepdim=True).squeeze(-1)  # (num_envs,)
    body_weight = total_mass * 9.8  # 体重（N）
    
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # (num_envs, 2)
    
    # 计算接触力阈值
    swing_threshold = body_weight * swing_force_ratio  # 摆动脚阈值
    support_threshold = body_weight * support_force_ratio  # 支撑脚阈值
    
    left_force = contact_forces_z[:, 0]
    right_force = contact_forces_z[:, 1]
    
    # 判断单脚支撑：一只脚力量小（摆动脚）且另一只力量大（支撑脚）
    left_is_swing = (left_force < swing_threshold) & (right_force > support_threshold)
    right_is_swing = (right_force < swing_threshold) & (left_force > support_threshold)
    single_foot_support = left_is_swing | right_is_swing
    
    # 获取实际脚高度
    feet_pos_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, 2)
    
    # 摆动脚的高度：左脚是摆动脚取左脚高度，右脚是摆动脚取右脚高度
    swing_foot_height = torch.where(
        left_is_swing,
        feet_pos_z[:, 0],  # 左脚是摆动脚
        torch.where(
            right_is_swing,
            feet_pos_z[:, 1],  # 右脚是摆动脚
            torch.zeros_like(feet_pos_z[:, 0]),  # 双脚支撑，无摆动脚
        )
    )
    
    # 计算高度惩罚：摆动脚高度不足时惩罚
    height_penalty = torch.clamp((single_support_target - swing_foot_height) * single_support_scale, min=0.0)

    # 只在单脚支撑时惩罚
    height_reward = torch.where(
        single_foot_support, 
        height_penalty,
        torch.zeros_like(height_penalty),
    )

    return height_reward

def twisted_feet(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    shank_cfg: SceneEntityCfg,  # 小腿配置，用于判断是否接近站立位置
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,  # 判断是否接触地面的阈值
    horizontal_threshold: float = 0.3,  # 局部坐标系 x/y 分量占比阈值
    shank_upright_threshold: float = 0.7,  # 小腿接近垂直的阈值（cos值，0.7≈45度）
) -> torch.Tensor:
    """惩罚脚掌歪着着地（接触力不垂直于脚掌表面）。
    
    在平地上，世界坐标系的接触力主要是竖直向上的（z轴）。
    要判断脚是否"歪着着地"，需要将接触力转换到脚的局部坐标系。
    如果脚正着落地，力应该主要沿着脚的局部 z 轴；
    如果脚歪着，力会有较大的局部 x 或 y 分量。
    
    只有当小腿接近垂直（站立位置）时才应用惩罚，
    避免在躺下、跪地或摆动腿时误惩罚。
    
    Args:
        env: 强化学习环境对象
        sensor_cfg: 接触传感器配置（监控双脚）
        shank_cfg: 小腿配置，body_ids 应包含左右小腿的索引
        asset_cfg: 机器人资产配置
        contact_threshold: 判断是否接触地面的力阈值（N）
        horizontal_threshold: 局部坐标系水平分量占比阈值（0-1），超过此比例会被惩罚
        shank_upright_threshold: 小腿接近垂直的阈值（cos值），默认0.7约等于45度倾斜
        
    Returns:
        每个环境的惩罚值 (num_envs,)，值越大惩罚越重
    """
    # 获取接触传感器和机器人
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取世界坐标系下的接触力（最新一帧）
    # 形状: (num_envs, num_feet, 3)
    contact_forces_w = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, :]
    
    # 获取小腿的四元数，用于判断小腿是否接近垂直
    # 形状: (num_envs, num_shanks, 4)
    shank_quat_w = asset.data.body_quat_w[:, shank_cfg.body_ids, :]
    
    # 计算小腿的 Z 轴在世界坐标系中的方向
    # 世界 Z 轴向量: [0, 0, 1]
    num_envs = contact_forces_w.shape[0]
    num_shanks = shank_quat_w.shape[1]
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_envs * num_shanks, 3)
    
    # 将世界 Z 轴转换到小腿局部坐标系
    shank_quat_flat = shank_quat_w.reshape(-1, 4)
    local_z = quat_rotate_inverse(shank_quat_flat, world_z)  # (num_envs * num_shanks, 3)
    local_z = local_z.reshape(num_envs, num_shanks, 3)
    
    # 小腿局部 Z 轴的世界 Z 分量表示小腿的垂直程度
    # 值接近 1 表示小腿垂直，值接近 0 表示小腿水平
    shank_uprightness = local_z[:, :, 2]  # (num_envs, num_shanks)
    
    # 判断小腿是否接近垂直（站立位置）
    is_shank_upright = shank_uprightness > shank_upright_threshold  # (num_envs, num_shanks)
    
    # 获取脚的四元数
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]  # (num_envs, num_feet, 4)
    num_feet = contact_forces_w.shape[1]
    
    # 展平: (num_envs * num_feet, 3) 和 (num_envs * num_feet, 4)
    contact_forces_w_flat = contact_forces_w.reshape(-1, 3)
    foot_quat_w_flat = foot_quat_w.reshape(-1, 4)
    
    # 转换到局部坐标系: (num_envs * num_feet, 3)
    contact_forces_local_flat = quat_rotate_inverse(foot_quat_w_flat, contact_forces_w_flat)
    
    # 恢复形状: (num_envs, num_feet, 3)
    contact_forces_local = contact_forces_local_flat.reshape(num_envs, num_feet, 3)
    
    # 分离局部坐标系的各个分量
    force_local_x = contact_forces_local[:, :, 0]  # (num_envs, num_feet)
    force_local_y = contact_forces_local[:, :, 1]
    force_local_z = contact_forces_local[:, :, 2]  # 应该是主要分量（垂直于脚掌）
    
    # 计算水平分量（局部 x 和 y）的模长
    force_horizontal = torch.sqrt(force_local_x**2 + force_local_y**2)
    
    # 计算总力的模长
    force_total = torch.norm(contact_forces_local, dim=-1)  # (num_envs, num_feet)
    
    # 计算水平分量占总力的比例
    # 添加小的 epsilon 避免除零
    horizontal_ratio = force_horizontal / (force_total + 1e-6)
    
    # 只有在脚接触地面时才计算惩罚
    contact_mag = torch.norm(contact_forces_w, dim=-1)
    is_in_contact = contact_mag > contact_threshold
    
    # 计算惩罚：超过阈值的水平分量比例
    # 如果 horizontal_ratio > horizontal_threshold，说明脚歪着着地
    penalty_per_foot = torch.clamp(horizontal_ratio - horizontal_threshold, min=0.0)
    
    # 只在接触且对应小腿接近垂直时应用惩罚
    # 假设 sensor_cfg.body_ids 和 shank_cfg.body_ids 的顺序一致（左脚-左小腿，右脚-右小腿）
    penalty_per_foot = torch.where(
        is_in_contact & is_shank_upright,
        penalty_per_foot,
        torch.zeros_like(penalty_per_foot)
    )
    
    # 对所有脚求和
    total_penalty = torch.sum(penalty_per_foot, dim=1)
    
    return total_penalty

def penalize_feet_contact_when_should_lift(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: list[str] | None = None,
    lift_height_threshold: float = 0.05,  # 动捕数据中脚高度超过此值认为应该离地（米）
    contact_threshold: float = 10.0,  # 判断机器人脚是否接触地面的力阈值（N）
    penalty_scale: float = 1.0,  # 惩罚缩放因子
) -> torch.Tensor:
    """惩罚在应该抬腿时脚还接触地面的行为。
    
    当动捕数据中脚的高度超过阈值（说明应该离地）时，如果机器人脚还接触地面，
    就给予惩罚。这有助于训练机器人在跳舞等需要大抬腿的动作中正确抬腿。
    """
    # 获取动捕命令
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    
    # 获取动捕数据中脚的位置（绝对世界坐标）
    # body_pos_w 形状: (num_envs, num_bodies, 3)，已经包含了env_origins，所以Z坐标就是相对于地面的高度
    motion_feet_pos_w = command.body_pos_w[:, body_indexes, :]  # (num_envs, num_feet, 3)
    # 获取环境原点（地面高度）
    env_origins = env.scene.env_origins  # (num_envs, 3)
    # 计算脚相对于地面的高度（Z坐标，body_pos_w已经包含env_origins，所以需要减去）
    motion_feet_height = motion_feet_pos_w[:, :, 2] - env_origins[:, None, 2]  # (num_envs, num_feet)
    
    # 判断动捕数据中脚是否应该离地（高度超过阈值）
    should_lift = motion_feet_height > lift_height_threshold  # (num_envs, num_feet)
    
    # 获取接触传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 获取接触力历史数据，形状为 (num_envs, history_frames, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # 计算接触力大小，取历史最大值
    contact_force_norm = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids, :], dim=-1), 
        dim=1
    )[0]  # (num_envs, num_feet)
    
    # 判断机器人脚是否接触地面
    is_in_contact = contact_force_norm > contact_threshold  # (num_envs, num_feet)
    
    # 计算惩罚：应该离地但还在接触的情况
    # 只有当 should_lift=True 且 is_in_contact=True 时才惩罚
    violation = should_lift & is_in_contact  # (num_envs, num_feet)
    
    # 惩罚值：根据接触力大小和高度差计算
    # 接触力越大，惩罚越大；高度差越大（应该抬得更高），惩罚越大
    height_error = motion_feet_height - lift_height_threshold  # 高度差
    contact_penalty = contact_force_norm / contact_threshold  # 归一化的接触力
    penalty_per_foot = violation.float() * height_error * contact_penalty * penalty_scale
    
    # 对所有脚求和
    total_penalty = torch.sum(penalty_per_foot, dim=1)  # (num_envs,)
    
    return total_penalty

def feet_flat_orientation_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 10.0,  # 判断是否接触地面的力阈值
    std: float = 0.3,  # 标准差，控制奖励衰减速度（弧度）
    only_when_contact: bool = True,  # 是否只在脚接触地面时给奖励
) -> torch.Tensor:
    """奖励脚底保持水平（平坦）的姿态。
    
    这个函数鼓励机器人在训练时保持脚底水平，即使原始动捕数据中脚底不是平的。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取脚底的姿态四元数
    # 形状: (num_envs, num_feet, 4)
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    
    # 世界坐标系的z轴（向上）
    num_envs = foot_quat_w.shape[0]
    num_feet = foot_quat_w.shape[1]
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_envs * num_feet, 3)
    
    # 将世界z轴转换到脚底的局部坐标系
    foot_quat_w_flat = foot_quat_w.reshape(-1, 4)
    local_z = quat_rotate_inverse(foot_quat_w_flat, world_z)  # (num_envs * num_feet, 3)
    local_z = local_z.reshape(num_envs, num_feet, 3)
    
    # 如果脚底水平，局部z轴应该接近[0, 0, 1]
    # 计算局部z轴与[0, 0, 1]的偏差
    target_local_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(num_envs, num_feet, 3)
    
    # 计算偏差（只考虑x和y分量，因为z分量在水平时应该接近1）
    # 脚底倾斜时，局部z轴的x和y分量会偏离0
    local_z_xy = local_z[:, :, :2]  # (num_envs, num_feet, 2) - 只取x和y分量
    deviation_xy = torch.norm(local_z_xy, dim=-1)  # (num_envs, num_feet) - 计算xy平面的偏差
    
    # 也可以考虑z分量的偏差（理想情况下应该接近1）
    local_z_z = local_z[:, :, 2]  # (num_envs, num_feet) - z分量
    deviation_z = torch.abs(local_z_z - 1.0)  # (num_envs, num_feet) - z分量与1的偏差
    
    # 综合偏差（可以加权组合）
    total_deviation = deviation_xy + 0.5 * deviation_z  # (num_envs, num_feet)
    
    # 使用指数函数计算奖励：exp(-deviation^2 / std^2)
    reward_per_foot = torch.exp(-torch.square(total_deviation) / (std ** 2))  # (num_envs, num_feet)
    
    # 如果设置了只在接触时给奖励，需要应用掩码
    if only_when_contact:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact_forces_z = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids, 2]  # (num_envs, num_feet)
        foot_in_contact = contact_forces_z > contact_threshold  # (num_envs, num_feet)
        
        # 只在接触时给奖励
        reward_per_foot = reward_per_foot * foot_in_contact.float()
    
    # 对所有脚求平均
    reward = reward_per_foot.mean(dim=-1)  # (num_envs,)
    
    return reward

