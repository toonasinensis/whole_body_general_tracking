from __future__ import annotations

import torch


def compute_heading_reward(
    heading_dir_w: torch.Tensor,
    facing_dir_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    robot_facing_dir_w: torch.Tensor,
    target_speed: torch.Tensor | float,
    alpha: float = 0.25,
    velocity_weight: float = 0.7,
    facing_weight: float = 0.3,
) -> torch.Tensor:
    if not torch.is_tensor(target_speed):
        target_speed = torch.full_like(root_lin_vel_w[:, 0], float(target_speed))
    velocity_along_heading = torch.sum(heading_dir_w * root_lin_vel_w, dim=-1)
    velocity_reward = torch.exp(-alpha * torch.square(target_speed - velocity_along_heading))
    facing_alignment = torch.sum(facing_dir_w * robot_facing_dir_w, dim=-1)
    return velocity_weight * velocity_reward + facing_weight * facing_alignment


def compute_height_filtered_contact_mask(
    force_norm: torch.Tensor,
    body_z: torch.Tensor,
    threshold: float,
    max_body_height: float,
    ground_height: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    if not torch.is_tensor(ground_height):
        ground_height = torch.full_like(body_z[:, :1], float(ground_height))
    elif ground_height.ndim == 1:
        ground_height = ground_height.unsqueeze(-1)
    else:
        ground_height = ground_height.to(device=body_z.device, dtype=body_z.dtype)
    return (force_norm > threshold) & ((body_z - ground_height) < max_body_height)
