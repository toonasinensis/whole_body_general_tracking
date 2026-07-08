from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ratio: float = 5.0,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet whose horizontal contact force dominates vertical support force."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    horizontal_force = torch.linalg.norm(net_contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(net_contact_forces[..., 2])
    stumble = (horizontal_force > float(force_threshold)) & (
        horizontal_force > float(ratio) * vertical_force.clamp_min(1.0e-6)
    )
    return torch.any(stumble, dim=(1, 2)).float()


def feet_flat(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ratio: float = 5.0,
    force_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize contacted feet whose sole frame is not flat with the world ground plane.

    Only feet with contact force above ``force_threshold`` are penalized.  The
    penalty is the foot-frame gravity xy error, which is zero when the foot's
    local z axis is aligned with world up.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force = torch.linalg.norm(net_contact_forces, dim=-1)
    is_contact = torch.any(contact_force > float(force_threshold), dim=1)

    asset = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, sensor_cfg.body_ids]
    gravity_w = asset.data.GRAVITY_VEC_W
    if gravity_w.ndim == 2:
        gravity_w = gravity_w.unsqueeze(1).expand(-1, foot_quat_w.shape[1], -1)
    else:
        gravity_w = gravity_w.view(1, 1, 3).expand(foot_quat_w.shape[0], foot_quat_w.shape[1], -1)

    projected_gravity_b = quat_apply_inverse(
        foot_quat_w.reshape(-1, 4),
        gravity_w.reshape(-1, 3),
    ).reshape(foot_quat_w.shape[0], foot_quat_w.shape[1], 3)
    flat_error = torch.sum(torch.square(projected_gravity_b[..., :2]), dim=-1)
    contact_mask = is_contact.to(dtype=flat_error.dtype)
    contact_count = contact_mask.sum(dim=1).clamp_min(1.0)
    return torch.sum(flat_error * contact_mask, dim=1) / contact_count
