from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from whole_body_tracking.tasks.sonic.mdp.commands import MotionCommandSonic
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from isaaclab.envs.mdp.observations import (
    joint_pos_rel as _joint_pos_rel,
    joint_vel_rel as _joint_vel_rel,
    base_ang_vel as _base_ang_vel,
    projected_gravity as _projected_gravity,
    last_action as _last_action,
    generated_commands as _generated_commands,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


##########################
# Proprioceptive sensing #########
# 1. Joint position and velocity ########################
# 2. Root angular velocity in local heading frame       #
# 3. Gravity vector in root frame                       #
# 4. Previous action                                    #
#########################################################
def sonic_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _joint_pos_rel(env, asset_cfg)

def sonic_joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _joint_vel_rel(env, asset_cfg)

def sonic_base_ang_vel(env: ManagerBasedEnv) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    # ω in base frame
    omega_b = asset.data.root_ang_vel_b  # (N, 3)
    # Base orientation and yaw-only heading orientation (world frame)
    quat_base_w = asset.data.root_quat_w          # (N, 4)
    quat_heading_w = yaw_quat(quat_base_w)        # (N, 4), roll/pitch removed
    # Rotation from base frame to heading frame: q_hb = q_heading_w * inv(q_base_w)
    quat_hb = quat_mul(quat_heading_w, quat_inv(quat_base_w))
    # Express ω in heading frame
    omega_h = quat_apply(quat_hb, omega_b)        # (N, 3)
    return omega_h

def sonic_projected_gravity(env: ManagerBasedEnv) -> torch.Tensor:
    return _projected_gravity(env, SceneEntityCfg("robot"))

def sonic_last_action(env: ManagerBasedEnv) -> torch.Tensor:
    return _last_action(env)


###########
# Command #
###########
def sonic_commands(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    return _generated_commands(env, command_name)


#######################
# Special for critics ###################################
# Relative body position and orientation to anchor body #
#########################################################
def sonic_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body position in the anchor body frame (for critics).

    Returns shape: (num_envs, num_bodies * 3)
    """
    command: MotionCommandSonic = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)

    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def sonic_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body orientation in the anchor body frame (for critics).

    Encodes each body orientation by the first 2 columns of its rotation matrix.
    Returns shape: (num_envs, num_bodies * 6)
    """
    command: MotionCommandSonic = env.command_manager.get_term(command_name)
    num_bodies = len(command.cfg.body_names)

    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    mat = matrix_from_quat(ori_b)          # (num_envs, num_bodies, 3, 3)
    return mat[..., :2].reshape(mat.shape[0], -1)
