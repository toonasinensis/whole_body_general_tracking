from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude
from isaaclab.utils.string import resolve_matching_names_values

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.heading_math import compute_height_filtered_contact_mask

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


def _scale_to_delayed_termination_active(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    active_only: bool,
    active_scale: float,
) -> torch.Tensor:
    if not active_only:
        return reward

    mask = getattr(env.termination_manager, "delayed_termination_active_mask", None)
    if mask is None:
        return torch.zeros_like(reward)

    mask = mask.to(device=reward.device, dtype=torch.bool)
    return torch.where(mask, reward * active_scale, torch.zeros_like(reward))


def root_height_recovery_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float | None = None,
    active_only_on_delayed_termination: bool = True,
    active_scale: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    desired_height = (
        asset.data.default_root_state[:, 2]
        if target_height is None
        else torch.full_like(asset.data.root_pos_w[:, 2], float(target_height))
    )
    error = torch.square(desired_height - asset.data.root_pos_w[:, 2])
    reward = torch.exp(-error / std**2)
    return _scale_to_delayed_termination_active(env, reward, active_only_on_delayed_termination, active_scale)


def root_height_below_target_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize only the portion of root height below a target height."""
    asset = env.scene[asset_cfg.name]
    error = torch.clamp(float(target_height) - asset.data.root_pos_w[:, 2], min=0.0)
    return torch.square(error)


def root_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical root velocity so heading speed cannot be earned by falling."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def body_xy_ang_vel_stability_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        body_quat_w = asset.data.root_quat_w
        body_ang_vel_w = asset.data.root_ang_vel_w
    else:
        body_id = asset_cfg.body_ids[0]
        body_quat_w = asset.data.body_quat_w[:, body_id]
        body_ang_vel_w = asset.data.body_ang_vel_w[:, body_id]

    body_ang_vel_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
    error = torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=-1)
    reward = torch.exp(-error / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def upright_orientation_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        projected_gravity_b = asset.data.projected_gravity_b
    else:
        body_id = asset_cfg.body_ids[0]
        projected_gravity_b = quat_apply_inverse(asset.data.body_quat_w[:, body_id], asset.data.GRAVITY_VEC_W)
    error = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)
    reward = torch.exp(-error / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def projected_gravity_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch tilt using projected gravity in the base frame."""
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        projected_gravity_b = asset.data.projected_gravity_b
    else:
        body_id = asset_cfg.body_ids[0]
        projected_gravity_b = quat_apply_inverse(asset.data.body_quat_w[:, body_id], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


def body_projected_gravity_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch tilt for a selected body."""
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        projected_gravity_b = asset.data.projected_gravity_b
    else:
        body_id = asset_cfg.body_ids[0]
        projected_gravity_b = quat_apply_inverse(asset.data.body_quat_w[:, body_id], asset.data.GRAVITY_VEC_W)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)


def body_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch tilt for the root or selected bodies.

    If ``asset_cfg.body_names`` resolves to multiple links, their projected-gravity xy errors
    are averaged so the reward scale stays comparable when changing the link set.
    """
    asset = env.scene[asset_cfg.name]
    if asset_cfg.body_ids == slice(None):
        return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)

    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    if body_quat_w.ndim == 2:
        projected_gravity_b = quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)
        return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=-1)

    gravity_w = asset.data.GRAVITY_VEC_W
    if gravity_w.ndim == 2:
        gravity_w = gravity_w.unsqueeze(1).expand(-1, body_quat_w.shape[1], -1)
    else:
        gravity_w = gravity_w.view(1, 1, 3).expand(body_quat_w.shape[0], body_quat_w.shape[1], -1)
    projected_gravity_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), gravity_w.reshape(-1, 3)).reshape(
        body_quat_w.shape[0], body_quat_w.shape[1], 3
    )
    return torch.mean(torch.sum(torch.square(projected_gravity_b[..., :2]), dim=-1), dim=-1)


def default_joint_position_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    error = torch.mean(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=-1,
    )
    reward = torch.exp(-error / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


class heading_variable_joint_posture(ManagerTermBase):
    """Speed-conditioned posture reward for the standalone heading walk task."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos.clone()
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names, preserve_order=True)
        _, _, std_standing = resolve_matching_names_values(
            cfg.params["std_standing"], joint_names, preserve_order=False, strict=True
        )
        _, _, std_walking = resolve_matching_names_values(
            cfg.params["std_walking"], joint_names, preserve_order=False, strict=True
        )
        self.std_standing = torch.as_tensor(std_standing, device=env.device, dtype=torch.float32)
        self.std_walking = torch.as_tensor(std_walking, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        walking_threshold: float = 0.05,
    ) -> torch.Tensor:
        del std_standing, std_walking
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_term(command_name)
        target_speed = getattr(command, "target_speed", None)
        if target_speed is None:
            speed = torch.zeros(env.num_envs, device=env.device)
        else:
            speed = target_speed[:, 0]
        walking_mask = (speed >= float(walking_threshold)).float().unsqueeze(-1)
        std = self.std_standing * (1.0 - walking_mask) + self.std_walking * walking_mask
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / torch.square(std), dim=-1))


class variable_posture(ManagerTermBase):
    """Speed-conditioned default joint posture reward for velocity commands."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos.clone()
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names, preserve_order=True)
        _, _, std_standing = resolve_matching_names_values(
            cfg.params["std_standing"], joint_names, preserve_order=False, strict=True
        )
        _, _, std_walking = resolve_matching_names_values(
            cfg.params["std_walking"], joint_names, preserve_order=False, strict=True
        )
        _, _, std_running = resolve_matching_names_values(
            cfg.params["std_running"], joint_names, preserve_order=False, strict=True
        )
        self.std_standing = torch.as_tensor(std_standing, device=env.device, dtype=torch.float32)
        self.std_walking = torch.as_tensor(std_walking, device=env.device, dtype=torch.float32)
        self.std_running = torch.as_tensor(std_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.1,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < float(walking_threshold)).float().unsqueeze(-1)
        walking_mask = (
            ((total_speed >= float(walking_threshold)) & (total_speed < float(running_threshold))).float().unsqueeze(-1)
        )
        running_mask = (total_speed >= float(running_threshold)).float().unsqueeze(-1)
        std = self.std_standing * standing_mask + self.std_walking * walking_mask + self.std_running * running_mask
        std = torch.clamp(std, min=1.0e-6)

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        return torch.exp(-torch.mean(error_squared / torch.square(std), dim=-1))


def heading_base_linear_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    z_weight: float = 2.0,
    target_speed: float | None = None,
    min_root_height: float | None = None,
    target_root_height: float | None = None,
    height_gate_std: float = 0.08,
    upright_gate_std: float | None = None,
) -> torch.Tensor:
    """Track heading velocity in the robot frame and suppress falling shortcuts."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    speed = (
        command.target_speed[:, 0]
        if target_speed is None
        else torch.full_like(command.target_speed[:, 0], target_speed)
    )
    command_b = torch.cat((command.heading_dir_b * speed.unsqueeze(-1), torch.zeros_like(speed.unsqueeze(-1))), dim=-1)
    lin_vel_error = torch.sum(torch.square(command_b[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=-1)
    lin_vel_error = lin_vel_error + float(z_weight) * torch.square(asset.data.root_lin_vel_b[:, 2])
    reward = torch.exp(-lin_vel_error / float(std) ** 2)

    if min_root_height is not None:
        reward = reward * (asset.data.root_pos_w[:, 2] >= float(min_root_height)).float()
    if target_root_height is not None:
        height_error = torch.square(asset.data.root_pos_w[:, 2] - float(target_root_height))
        reward = reward * torch.exp(-height_error / height_gate_std**2)
    if upright_gate_std is not None:
        upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
        reward = reward * torch.exp(-upright_error / upright_gate_std**2)
    return reward


def motion_global_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, disable_on_delayed_termination: bool = False
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    reward = torch.exp(-error / std**2)
    return _zero_delayed_termination_rewards(env, reward, disable_on_delayed_termination)


def motion_global_anchor_position_z_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    return torch.exp(-error / std**2)


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


def height_filtered_undesired_contacts(
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
    return torch.sum(is_contact, dim=1)


def heading_feet_gait(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    period: float,
    offset: list[float],
    stance_threshold: float,
    force_threshold: float = 1.0,
    command_name: str = "heading",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward an alternating left/right foot contact pattern for the heading task."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    is_contact = force_norm > float(force_threshold)

    global_phase = ((env.episode_length_buf * env.step_dt) / float(period)).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    is_stance = ((global_phase + offsets) % 1.0) < float(stance_threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)

    command = env.command_manager.get_term(command_name)
    target_speed = getattr(command, "target_speed", None)
    if target_speed is not None:
        reward = reward * (target_speed[:, 0] > float(command_threshold)).float()
    return reward


def heading_feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
    command_name: str = "heading",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize horizontal foot velocity while a foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0]
    is_contact = (force_norm > float(force_threshold)).float()
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * is_contact, dim=1)

    command = env.command_manager.get_term(command_name)
    target_speed = getattr(command, "target_speed", None)
    if target_speed is not None:
        slip = slip * (target_speed[:, 0] > float(command_threshold)).float()
    if "log" in env.extras:
        contacts = torch.clamp(is_contact.sum(), min=1.0)
        env.extras["log"]["Metrics/heading/slip_velocity"] = (
            torch.sum(torch.norm(foot_vel_xy, dim=-1) * is_contact) / contacts
        )
    return slip
