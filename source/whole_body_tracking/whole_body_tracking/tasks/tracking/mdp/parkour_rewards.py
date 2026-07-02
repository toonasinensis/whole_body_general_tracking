from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

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
