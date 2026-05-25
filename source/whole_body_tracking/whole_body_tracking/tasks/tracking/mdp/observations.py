from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

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
    return command.joint_pos_future.view(env.num_envs, -1)


def motion_joint_vel_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel_future.view(env.num_envs, -1)


def motion_anchor_ori_b_mf(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w_future
    root_rot_dif = quat_mul(
        quat_inv(command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, command.num_future_frames, 1)),
        ref_root_quat.view(env.num_envs, command.num_future_frames, 4),
    )
    mat = matrix_from_quat(root_rot_dif)
    root_rot_dif_l_mat = mat[..., :2].reshape(mat.shape[0], -1)
    return root_rot_dif_l_mat


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
