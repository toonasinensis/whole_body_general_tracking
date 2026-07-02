from __future__ import annotations

import os
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.domain_commands import domain_name_float_mask, domain_name_mask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_AMP_BODY_ID_CACHE: dict[tuple[int, tuple[str, ...], bool], list[int]] = {}


def _resolve_amp_body_ids(asset, body_names: str | Sequence[str], preserve_order: bool = True) -> list[int]:
    if isinstance(body_names, str):
        body_names_key = (body_names,)
    else:
        body_names_key = tuple(body_names)
    cache_key = (id(asset), body_names_key, preserve_order)
    body_ids = _AMP_BODY_ID_CACHE.get(cache_key)
    if body_ids is None:
        body_ids, _ = asset.find_bodies(list(body_names_key), preserve_order=preserve_order)
        _AMP_BODY_ID_CACHE[cache_key] = body_ids
    return body_ids


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def amp_robot_body_pos_b(
    env: ManagerBasedEnv,
    asset_name: str = "robot",
    anchor_body_name: str = "",
    body_names: tuple[str, ...] = (),
) -> torch.Tensor:
    asset = env.scene[asset_name]
    anchor_id = _resolve_amp_body_ids(asset, anchor_body_name)[0]
    body_ids = _resolve_amp_body_ids(asset, body_names)
    anchor_pos_w = asset.data.body_pos_w[:, anchor_id]
    anchor_quat_w = asset.data.body_quat_w[:, anchor_id]
    body_pos_w = asset.data.body_pos_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    num_bodies = len(body_ids)
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def amp_robot_body_ori_b(
    env: ManagerBasedEnv,
    asset_name: str = "robot",
    anchor_body_name: str = "",
    body_names: tuple[str, ...] = (),
) -> torch.Tensor:
    asset = env.scene[asset_name]
    anchor_id = _resolve_amp_body_ids(asset, anchor_body_name)[0]
    body_ids = _resolve_amp_body_ids(asset, body_names)
    anchor_pos_w = asset.data.body_pos_w[:, anchor_id]
    anchor_quat_w = asset.data.body_quat_w[:, anchor_id]
    body_pos_w = asset.data.body_pos_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    num_bodies = len(body_ids)
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def amp_robot_body_lin_vel_b(
    env: ManagerBasedEnv,
    asset_name: str = "robot",
    anchor_body_name: str = "",
    body_names: tuple[str, ...] = (),
) -> torch.Tensor:
    del anchor_body_name
    asset = env.scene[asset_name]
    body_ids = _resolve_amp_body_ids(asset, body_names)
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    num_bodies = len(body_ids)
    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return body_lin_vel_b.reshape(env.num_envs, -1)


def amp_robot_body_ang_vel_b(
    env: ManagerBasedEnv,
    asset_name: str = "robot",
    anchor_body_name: str = "",
    body_names: tuple[str, ...] = (),
) -> torch.Tensor:
    del anchor_body_name
    asset = env.scene[asset_name]
    body_ids = _resolve_amp_body_ids(asset, body_names)
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids]
    body_quat_w = asset.data.body_quat_w[:, body_ids]
    num_bodies = len(body_ids)
    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)
    return body_ang_vel_b.reshape(env.num_envs, -1)


def heading_phase(
    env: ManagerBasedEnv, period: float, command_name: str, command_threshold: float = 0.05
) -> torch.Tensor:
    """Sin/cos gait phase used by the standalone heading walk task."""
    phase_t = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.stack((torch.sin(phase_t * torch.pi * 2.0), torch.cos(phase_t * torch.pi * 2.0)), dim=-1)
    command = env.command_manager.get_term(command_name)
    target_speed = getattr(command, "target_speed", None)
    if target_speed is None:
        return phase
    active = target_speed[:, 0] > float(command_threshold)
    return torch.where(active.unsqueeze(-1), phase, torch.zeros_like(phase))


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.cat(
        [
            command.joint_pos,
            command.joint_vel,
            command.anchor_lin_vel_b,
            command.anchor_ang_vel_b,
            command.anchor_project_gravity,
            command.anchor_pos_z,
        ],
        dim=1,
    )


def motion_command_linvel_xy_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_yaw_quat = yaw_quat(command.anchor_quat_w)
    return quat_apply_inverse(anchor_yaw_quat, command.anchor_lin_vel_w)


def motion_velocity_command_yaw_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_yaw_quat = yaw_quat(command.anchor_quat_w)
    lin_vel_yaw_b = quat_apply_inverse(anchor_yaw_quat, command.anchor_lin_vel_w)
    ang_vel_yaw_b = quat_apply_inverse(anchor_yaw_quat, command.anchor_ang_vel_w)
    yaw_rate = ang_vel_yaw_b[:, 2:3]
    return torch.cat((lin_vel_yaw_b[:, :2], yaw_rate), dim=-1)


def mixed_velocity_command(
    env: ManagerBasedEnv,
    motion_command_name: str = "motion",
    velocity_command_name: str = "base_velocity",
    velocity_domain_names: tuple[str, ...] = ("flat_velocity",),
) -> torch.Tensor:
    motion_cmd = motion_velocity_command_yaw_b(env, motion_command_name)
    if not velocity_domain_names:
        return motion_cmd
    velocity_cmd = velocity_mdp.generated_commands(env, velocity_command_name)
    velocity_mask = domain_name_mask(env, motion_command_name, velocity_domain_names).unsqueeze(-1)
    out = torch.where(velocity_mask, velocity_cmd, motion_cmd)
    if os.getenv("WBT_DEBUG_VELCOMMAND", "0") not in ("", "0", "false", "False"):
        _debug_mixed_velocity_command_once(
            env, motion_command_name, velocity_domain_names, velocity_mask, velocity_cmd, motion_cmd, out
        )
    return out


def _debug_mixed_velocity_command_once(
    env: ManagerBasedEnv,
    motion_command_name: str,
    velocity_domain_names: tuple[str, ...],
    velocity_mask: torch.Tensor,
    velocity_cmd: torch.Tensor,
    motion_cmd: torch.Tensor,
    out: torch.Tensor,
) -> None:
    if getattr(env, "_wbt_debug_velcommand_printed", False):
        return
    setattr(env, "_wbt_debug_velcommand_printed", True)
    command = env.command_manager.get_term(motion_command_name)
    domain_layout = getattr(command, "domain_layout", None)
    domain_names = tuple(
        getattr(domain, "name", f"domain_{i}") for i, domain in enumerate(getattr(domain_layout, "domains", ()))
    )
    env_domain_ids = getattr(command, "env_domain_ids", None)
    sample_count = min(int(getattr(env, "num_envs", 0)), 8)
    print("[WBT_DEBUG_VELCOMMAND] velocity_domain_names:", tuple(velocity_domain_names))
    print("[WBT_DEBUG_VELCOMMAND] available_domains:", domain_names)
    for env_id in range(sample_count):
        domain_id = int(env_domain_ids[env_id].item()) if env_domain_ids is not None else -1
        domain_name = domain_names[domain_id] if 0 <= domain_id < len(domain_names) else "unknown"
        print(
            "[WBT_DEBUG_VELCOMMAND] "
            f"env={env_id} domain={domain_name} vel_task_mask={float(velocity_mask[env_id, 0].item()):.1f} "
            f"base_velocity={velocity_cmd[env_id].detach().cpu().tolist()} "
            f"motion_cmd={motion_cmd[env_id].detach().cpu().tolist()} "
            f"velcommand_obs={out[env_id].detach().cpu().tolist()}"
        )


def domain_aux_mask(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    aux_domain_names: tuple[str, ...] = ("flat_wbc", "omniretarget_g1_terrain"),
) -> torch.Tensor:
    # aux_domain_names comes from profile obs_route_cfg. This is where the
    # profile-level boolean is converted to the numeric policy input:
    # enabled domains -> 1.0, all other domains -> 0.0.
    if not aux_domain_names:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return domain_name_float_mask(env, command_name, aux_domain_names)


def motion_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel


def motion_anchor_lin_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_lin_vel_b


def motion_anchor_ang_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_ang_vel_b


def motion_anchor_project_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_project_gravity


def motion_anchor_pos_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_pos_z


def motion_joint_pos_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos_future.view(env.num_envs, -1)  # zero out for ablation


def motion_joint_vel_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel_future.view(env.num_envs, -1)  # zero out for ablation


def motion_anchor_ori_b_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w_future
    root_rot_dif = quat_mul(
        quat_inv(command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, command.num_future_frames, 1)),
        ref_root_quat.view(env.num_envs, command.num_future_frames, 4),
    )
    mat = matrix_from_quat(root_rot_dif)
    root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)  # zero out for ablation
    return root_rot_dif_l_mat


def motion_anchor_z_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_posz = command.anchor_pos_w_future[:, :, 2]
    return ref_root_posz.view(env.num_envs, -1)


def smpl_joints_local_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_smpl_data:
        return torch.zeros(env.num_envs, command.num_future_frames * 24 * 3, device=command.device)
    future_local = command.smpl_joints_local_multi_future.view(env.num_envs, -1)
    return future_local


def smpl_root_quat_w_dif_l_multi_future(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.has_smpl_data:
        return torch.zeros(env.num_envs, command.num_future_frames * 6, device=command.device)
    future_local = command.smpl_root_quat_w_dif_l_multi_future.view(env.num_envs, -1)
    return future_local
