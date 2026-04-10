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
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        # env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos
        env.action_manager.get_term("joint_pos")._offset[env_ids, :] = pos.unsqueeze(1)


def set_joint_soft_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    margin: float | dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """设置关节软限位，使用绝对边界值而不是百分比。
    
    该函数在环境初始化时调用，用于设置每个关节的软限位。软限位基于硬限位（URDF中定义的关节限位）
    减去指定的绝对边界值（margin）。
    
    Args:
        env: 环境实例
        env_ids: 环境索引（startup事件中通常为None，表示所有环境）
        margin: 边界值，可以是:
            - float: 对所有关节使用相同的边界值
            - dict[str, float]: 为不同关节指定不同的边界值，支持正则表达式
                例如: {"leg_[l,r]1_joint": 0.1, "waist_yaw_joint": 0.2}
        asset_cfg: 资产配置，默认为 "robot"
        
    Example:
        在 tracking_env_cfg.py 中使用:
        ```python
        @configclass
        class EventCfg:
            startup = EventTermCfg(
                func=mdp.set_joint_soft_limits,
                params={
                    "margin": {
                        "waist_yaw_joint": 0.1,
                        "leg_[l,r]1_joint": 0.1,
                        # ...
                    },
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                },
                mode="startup",
            )
        ```
    """
    import re
    
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # resolve joint indices
    if isinstance(asset_cfg.joint_ids, slice):
        # 处理 slice 类型（包括 slice(None)）
        if asset_cfg.joint_ids == slice(None):
            joint_ids = list(range(asset.num_joints))
        else:
            # 处理其他 slice 类型，如 slice(0, 10)
            joint_ids = list(range(*asset_cfg.joint_ids.indices(asset.num_joints)))
    elif isinstance(asset_cfg.joint_ids, list):
        joint_ids = asset_cfg.joint_ids
    else:
        # 如果是其他可迭代类型（如 tuple），转换为 list
        joint_ids = list(asset_cfg.joint_ids)
    
    # 获取关节名称
    joint_names = [asset.joint_names[i] for i in joint_ids]
    
    # 构建每个关节的 margin tensor
    if isinstance(margin, dict):
        margin_list = []
        for joint_name in joint_names:
            matched_margin = 0.0  # 默认值
            for pattern, margin_value in margin.items():
                # 支持 [l,r] 简写为 [lr]
                regex_pattern = pattern.replace("[l,r]", "[lr]")
                if re.match(f"^{regex_pattern}$", joint_name):
                    matched_margin = margin_value
                    break
            margin_list.append(matched_margin)
        margin_tensor = torch.tensor(margin_list, device=asset.device, dtype=torch.float32)
    else:
        # 所有关节使用相同的 margin
        margin_tensor = torch.full((len(joint_ids),), margin, device=asset.device, dtype=torch.float32)
    
    # 获取硬限位（URDF 中定义的关节限位）
    joint_limits = asset.data.joint_pos_limits[:, joint_ids, :]  # (num_envs, num_joints, 2)
    
    # 计算软限位: [lower + margin, upper - margin]
    for env_id in env_ids:
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 0] = joint_limits[env_id, :, 0] + margin_tensor
        asset.data.soft_joint_pos_limits[env_id, joint_ids, 1] = joint_limits[env_id, :, 1] - margin_tensor
    
    print(f"[set_joint_soft_limits] 已为 {len(env_ids)} 个环境设置自定义软限位")
    print(f"[set_joint_soft_limits] 影响的关节: {joint_names}")


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
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
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = (
        len(asset_cfg.body_ids)
        if isinstance(asset_cfg.body_ids, list)
        else asset.num_bodies
    )

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(
        force_range[:, 0], force_range[:, 1], size, asset.device
    )
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(
        torque_range[:, 0], torque_range[:, 1], size, asset.device
    )
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(
        forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids
    )


#######################
# command disturbance ###################################
# Joint comman jitter is applied to the joint positions #
# #######################################################
