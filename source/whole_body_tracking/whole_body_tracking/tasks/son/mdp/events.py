from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


#######################
# physical parameters #
# NOTE friction       #
# NOTE restitution    #
#######################
def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the joint default positions (calibration errors)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos,
            pos_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, :] = pos.unsqueeze(1)


def set_joint_soft_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    margin: float | dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set soft joint limits using absolute margins from the URDF hard limits."""
    import re

    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if isinstance(asset_cfg.joint_ids, slice):
        if asset_cfg.joint_ids == slice(None):
            joint_ids = list(range(asset.num_joints))
        else:
            joint_ids = list(range(*asset_cfg.joint_ids.indices(asset.num_joints)))
    elif isinstance(asset_cfg.joint_ids, list):
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = list(asset_cfg.joint_ids)

    joint_names = [asset.joint_names[i] for i in joint_ids]

    # build margin tensor
    if isinstance(margin, dict):
        margin_list = []
        for joint_name in joint_names:
            matched_margin = 0.0
            for pattern, margin_value in margin.items():
                regex_pattern = pattern.replace("[l,r]", "[lr]")
                if re.match(f"^{regex_pattern}$", joint_name):
                    matched_margin = margin_value
                    break
            margin_list.append(matched_margin)
        margin_tensor = torch.tensor(margin_list, device=asset.device, dtype=torch.float32)
    else:
        margin_tensor = torch.full((len(joint_ids),), margin, device=asset.device, dtype=torch.float32)

    joint_limits = asset.data.joint_pos_limits[:, joint_ids, :]

    for env_id in env_ids:
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 0] = joint_limits[env_id, :, 0] + margin_tensor
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 1] = joint_limits[env_id, :, 1] - margin_tensor

    print(f"[set_joint_soft_limits] Set custom soft limits for {len(env_ids)} envs")
    print(f"[set_joint_soft_limits] Affected joints: {joint_names}")


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    coms = asset.root_physx_view.get_coms().clone()
    coms[:, body_ids, :3] += rand_samples
    asset.root_physx_view.set_coms(coms, env_ids)


#############################
# External force and torque #
#############################
def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize external forces and torques on robot bodies."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range_t = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(
        force_range_t[:, 0],
        force_range_t[:, 1],
        size,
        asset.device,
    )
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range_t = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(
        torque_range_t[:, 0],
        torque_range_t[:, 1],
        size,
        asset.device,
    )

    asset.set_external_force_and_torque(
        forces,
        torques,
        env_ids=masked_env_ids,
        body_ids=asset_cfg.body_ids,
    )

