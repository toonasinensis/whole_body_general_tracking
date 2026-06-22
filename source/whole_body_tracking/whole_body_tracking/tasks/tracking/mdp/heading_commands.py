from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .heading_math import compute_heading_reward

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _unit_xy_from_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def _unit_xy_from_quat_yaw(quat_w: torch.Tensor) -> torch.Tensor:
    yaw = math_utils.euler_xyz_from_quat(quat_w)[2]
    return _unit_xy_from_angle(yaw)


def resolve_motion_reset_heading_dir(
    root_lin_vel_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    min_speed: float,
) -> torch.Tensor:
    vel_xy = root_lin_vel_w[:, :2]
    speed = torch.norm(vel_xy, dim=-1, keepdim=True)
    vel_dir = vel_xy / torch.clamp(speed, min=1.0e-6)
    yaw_dir = _unit_xy_from_quat_yaw(root_quat_w)
    use_velocity = speed[:, 0] >= float(min_speed)
    return torch.where(use_velocity.unsqueeze(-1), vel_dir, yaw_dir)


class HeadingCommand(CommandTerm):
    """Goal-conditioned heading/facing command for the HIL heading task."""

    cfg: HeadingCommandCfg

    def __init__(self, cfg: HeadingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.heading_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.facing_dir_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_dir_b = torch.zeros_like(self.heading_dir_w)
        self.facing_dir_b = torch.zeros_like(self.facing_dir_w)
        self.target_speed = torch.full((self.num_envs, 1), float(cfg.target_speed), device=self.device)

        self.metrics["velocity_along_heading"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["facing_alignment"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat((self.heading_dir_b, self.facing_dir_b, self.target_speed), dim=-1)

    @property
    def heading_dir_w3(self) -> torch.Tensor:
        return torch.cat((self.heading_dir_w, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)

    @property
    def facing_dir_w3(self) -> torch.Tensor:
        return torch.cat((self.facing_dir_w, torch.zeros(self.num_envs, 1, device=self.device)), dim=-1)

    @property
    def robot_facing_dir_w(self) -> torch.Tensor:
        yaw = self.robot.data.heading_w
        return _unit_xy_from_angle(yaw)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        if self.cfg.provider == "random":
            angle = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.heading_range)
            self.heading_dir_w[env_ids] = _unit_xy_from_angle(angle)
        elif self.cfg.provider == "motion_reset_velocity":
            self._resample_from_last_motion_reset(env_ids)
        else:
            raise NotImplementedError(
                f"HeadingCommand provider '{self.cfg.provider}' is reserved for future trajectory-conditioned goals."
            )

        if self.cfg.facing_mode == "same_as_heading":
            self.facing_dir_w[env_ids] = self.heading_dir_w[env_ids]
        elif self.cfg.facing_mode == "independent":
            facing_angle = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.facing_range)
            self.facing_dir_w[env_ids] = _unit_xy_from_angle(facing_angle)
        else:
            raise ValueError(f"Unsupported facing_mode: {self.cfg.facing_mode}")

        self.target_speed[env_ids, 0] = float(self.cfg.target_speed)

    def _resample_from_last_motion_reset(self, env_ids: torch.Tensor) -> None:
        root_lin_vel = getattr(self._env, "_last_motion_reset_root_lin_vel_w", None)
        root_quat = getattr(self._env, "_last_motion_reset_root_quat_w", None)
        if root_lin_vel is None or root_quat is None:
            angle = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.heading_range)
            self.heading_dir_w[env_ids] = _unit_xy_from_angle(angle)
            return

        self.heading_dir_w[env_ids] = resolve_motion_reset_heading_dir(
            root_lin_vel_w=root_lin_vel[env_ids],
            root_quat_w=root_quat[env_ids],
            min_speed=self.cfg.motion_reset_min_speed,
        )

    def _update_command(self):
        yaw = self.robot.data.heading_w
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        self.heading_dir_b[:, 0] = cos_yaw * self.heading_dir_w[:, 0] + sin_yaw * self.heading_dir_w[:, 1]
        self.heading_dir_b[:, 1] = -sin_yaw * self.heading_dir_w[:, 0] + cos_yaw * self.heading_dir_w[:, 1]
        self.facing_dir_b[:, 0] = cos_yaw * self.facing_dir_w[:, 0] + sin_yaw * self.facing_dir_w[:, 1]
        self.facing_dir_b[:, 1] = -sin_yaw * self.facing_dir_w[:, 0] + cos_yaw * self.facing_dir_w[:, 1]

    def _update_metrics(self):
        root_vel_xy = self.robot.data.root_lin_vel_w[:, :2]
        self.metrics["velocity_along_heading"] = torch.sum(self.heading_dir_w * root_vel_xy, dim=-1)
        self.metrics["facing_alignment"] = torch.sum(self.facing_dir_w * self.robot_facing_dir_w, dim=-1)
        self.metrics["root_height"] = self.robot.data.root_pos_w[:, 2]
        self.metrics["upright_xy"] = torch.norm(self.robot.data.projected_gravity_b[:, :2], dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "heading_visualizer"):
                self.heading_visualizer = VisualizationMarkers(self.cfg.heading_visualizer_cfg)
            if not hasattr(self, "facing_visualizer"):
                self.facing_visualizer = VisualizationMarkers(self.cfg.facing_visualizer_cfg)
            self.heading_visualizer.set_visibility(True)
            self.facing_visualizer.set_visibility(True)
        else:
            if hasattr(self, "heading_visualizer"):
                self.heading_visualizer.set_visibility(False)
            if hasattr(self, "facing_visualizer"):
                self.facing_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        del event
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.6
        heading_scale, heading_quat = self._resolve_direction_to_arrow(self.heading_dir_w)
        facing_scale, facing_quat = self._resolve_direction_to_arrow(self.facing_dir_w)
        facing_pos_w = base_pos_w.clone()
        facing_pos_w[:, 2] += 0.15
        self.heading_visualizer.visualize(base_pos_w, heading_quat, heading_scale)
        self.facing_visualizer.visualize(facing_pos_w, facing_quat, facing_scale)

    def _resolve_direction_to_arrow(self, direction_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.cfg.heading_visualizer_cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(direction_xy.shape[0], 1)
        arrow_scale[:, 0] *= float(self.cfg.debug_arrow_length)
        angle = torch.atan2(direction_xy[:, 1], direction_xy[:, 0])
        zeros = torch.zeros_like(angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, angle)
        return arrow_scale, arrow_quat


@configclass
class HeadingCommandCfg(CommandTermCfg):
    class_type: type = HeadingCommand

    asset_name: str = MISSING
    target_speed: float = 1.2
    provider: str = "random"
    facing_mode: str = "same_as_heading"
    heading_range: tuple[float, float] = (-math.pi, math.pi)
    facing_range: tuple[float, float] = (-math.pi, math.pi)
    motion_reset_min_speed: float = 0.2
    debug_arrow_length: float = 1.0

    heading_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading"
    )
    facing_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/facing"
    )

    heading_visualizer_cfg.markers["arrow"].scale = (0.4, 0.05, 0.05)
    facing_visualizer_cfg.markers["arrow"].scale = (0.4, 0.05, 0.05)


def heading_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HeadingCommand = env.command_manager.get_term(command_name)
    return command.command


def heading_direction_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HeadingCommand = env.command_manager.get_term(command_name)
    return command.heading_dir_b


def facing_direction_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HeadingCommand = env.command_manager.get_term(command_name)
    return command.facing_dir_b


def heading_target_speed(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: HeadingCommand = env.command_manager.get_term(command_name)
    return command.target_speed


def heading_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_speed: float | None = None,
    alpha: float = 0.25,
    velocity_weight: float = 0.7,
    facing_weight: float = 0.3,
    min_root_height: float | None = None,
    target_root_height: float | None = None,
    height_gate_std: float = 0.12,
    upright_gate_std: float | None = None,
    disable_on_delayed_termination: bool = False,
) -> torch.Tensor:
    command: HeadingCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    speed = (
        command.target_speed[:, 0]
        if target_speed is None
        else torch.full_like(command.target_speed[:, 0], target_speed)
    )
    reward = compute_heading_reward(
        heading_dir_w=command.heading_dir_w,
        facing_dir_w=command.facing_dir_w,
        root_lin_vel_w=asset.data.root_lin_vel_w[:, :2],
        robot_facing_dir_w=command.robot_facing_dir_w,
        target_speed=speed,
        alpha=alpha,
        velocity_weight=velocity_weight,
        facing_weight=facing_weight,
    )
    if min_root_height is not None:
        reward = reward * (asset.data.root_pos_w[:, 2] >= float(min_root_height)).float()
    if target_root_height is not None:
        height_error = torch.square(asset.data.root_pos_w[:, 2] - float(target_root_height))
        reward = reward * torch.exp(-height_error / height_gate_std**2)
    if upright_gate_std is not None:
        upright_error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)
        reward = reward * torch.exp(-upright_error / upright_gate_std**2)
    if not disable_on_delayed_termination:
        return reward

    mask = getattr(env.termination_manager, "delayed_termination_active_mask", None)
    if mask is None:
        return reward
    return torch.where(mask.to(device=reward.device, dtype=torch.bool), torch.zeros_like(reward), reward)
