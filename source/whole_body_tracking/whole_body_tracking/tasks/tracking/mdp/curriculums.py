from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityMetricStage(TypedDict, total=False):
    lin_vel_x: tuple[float, float]
    lin_vel_y: tuple[float, float]
    ang_vel_z: tuple[float, float]


def commands_vel_metric(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    velocity_stages: list[VelocityMetricStage],
    metric_targets: dict[str, float] | None = None,
    metric_thresholds: dict[str, float] | None = None,
    ema_alpha: float = 0.995,
    stage_tolerance: float = 1.0e-3,
    min_update_calls: int = 200,
) -> dict[str, torch.Tensor]:
    """EMA-update velocity command ranges when tracking metrics are below thresholds."""
    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command term {command_name!r} not found.")
    if not velocity_stages:
        return {}
    metric_env_ids = slice(None) if env_ids is None else env_ids
    if isinstance(metric_env_ids, slice):
        metric_env_count = int(torch.arange(env.num_envs, device=env.device)[metric_env_ids].numel())
    else:
        metric_env_count = int(torch.as_tensor(metric_env_ids, device=env.device).numel())

    state_name = f"_wbt_commands_vel_metric_{command_name}"
    state = getattr(env, state_name, None)
    if state is None:
        state = {"stage": 0, "call_count": 0}
        setattr(env, state_name, state)
        _apply_velocity_stage(command_term.cfg, velocity_stages[0])
    state["call_count"] = int(state.get("call_count", 0)) + 1

    # Keep metric_thresholds as a backwards-compatible alias.
    metric_targets = metric_targets or metric_thresholds or {"error_vel_xy": 0.35, "error_vel_yaw": 0.35}
    ema_alpha = min(max(float(ema_alpha), 0.0), 0.9999999999999999999)
    metric_values = {}
    metric_ok_per_name = {}
    metrics = getattr(command_term, "metrics", {})
    for metric_name, target in metric_targets.items():
        target = max(float(target), 1.0e-6)
        value = metrics.get(metric_name)
        if value is None:
            metric_tensor = torch.full((metric_env_count,), target * 10.0, device=env.device)
        else:
            metric_tensor = value[metric_env_ids].detach()
            if metric_tensor.ndim == 0:
                metric_tensor = metric_tensor.reshape(1)
        metric_value = metric_tensor.mean()
        metric_values[metric_name] = metric_value
        metric_ok_per_name[metric_name] = metric_tensor <= target

    metric_ok = torch.stack([value.to(dtype=torch.bool) for value in metric_ok_per_name.values()], dim=0).all(dim=0)
    metric_ok_float = metric_ok.to(dtype=torch.float32)
    metric_ok_mean = metric_ok_float.mean()
    current_stage = min(max(int(state["stage"]), 0), len(velocity_stages) - 1)
    target_stage = min(current_stage + 1, len(velocity_stages) - 1)
    update_enabled_scalar = bool(metric_ok_mean.item() > 0.1) and int(state["call_count"]) > int(min_update_calls)
    update_enabled = torch.full_like(metric_ok_float, float(metric_ok_mean.item() if update_enabled_scalar else 0.0))
    if update_enabled_scalar and target_stage > current_stage:
        target_reached = _ema_update_velocity_stage(
            command_term.cfg,
            velocity_stages[target_stage],
            ema_alpha=ema_alpha,
            tolerance=float(stage_tolerance),
        )
        if target_reached:
            state["stage"] = target_stage
            _apply_velocity_stage(command_term.cfg, velocity_stages[target_stage])
            current_stage = target_stage
            target_stage = min(current_stage + 1, len(velocity_stages) - 1)

    out = {
        "stage_index": torch.tensor(float(current_stage), device=env.device),
        "target_stage_index": torch.tensor(float(target_stage), device=env.device),
        "metric_update_enabled": update_enabled.mean(),
        "metric_ok": metric_ok_mean,
        "update_call_count": torch.tensor(float(state["call_count"]), device=env.device),
    }
    for metric_name, value in metric_values.items():
        out[f"metric/{metric_name}"] = value
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        value = getattr(command_term.cfg.ranges, key)
        out[f"{key}_min"] = torch.tensor(float(value[0]), device=env.device)
        out[f"{key}_max"] = torch.tensor(float(value[1]), device=env.device)
    return out


def _ema_update_velocity_stage(cfg, target_stage: VelocityMetricStage, ema_alpha: float, tolerance: float) -> bool:
    reached = True
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        target_value = target_stage.get(key)
        if target_value is None:
            continue
        current_value = getattr(cfg.ranges, key)
        next_value = (
            ema_alpha * float(current_value[0]) + (1.0 - ema_alpha) * float(target_value[0]),
            ema_alpha * float(current_value[1]) + (1.0 - ema_alpha) * float(target_value[1]),
        )
        setattr(cfg.ranges, key, next_value)
        reached = reached and abs(next_value[0] - float(target_value[0])) <= tolerance
        reached = reached and abs(next_value[1] - float(target_value[1])) <= tolerance
    return reached


def _apply_velocity_stage(cfg, stage: VelocityMetricStage) -> None:
    if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
    if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
    if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
