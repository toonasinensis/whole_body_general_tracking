from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import torch

from . import tmt_functions
from .tmt_manager import DelayedTerminationManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def install_delayed_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    delay_reset_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
    use_motion_pose_range_mask: bool = True,
) -> None:
    """Install a delayed termination manager on the selected environment subset."""
    del env_ids

    if isinstance(env.termination_manager, DelayedTerminationManager):
        return

    if delay_reset_env_ratio <= 0.0 or max_delay_steps <= 0:
        return

    if use_motion_pose_range_mask:
        command_cfg = getattr(getattr(getattr(env, "cfg", None), "commands", None), "motion", None)
        ratios = getattr(command_cfg, "pose_init_method_ratios", None)
        if not ratios:
            return
        total_ratio = sum(max(0.0, float(ratio)) for ratio in ratios.values())
        if total_ratio <= 0.0:
            return

        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        start = 0
        items = list(ratios.items())
        for index, (method, ratio) in enumerate(items):
            ratio = max(0.0, float(ratio)) / total_ratio
            end = env.num_envs if index == len(items) - 1 else start + int(env.num_envs * ratio)
            if str(method) != "range":
                delay_mask[start:end] = True
            start = end
    else:
        num_delay = int(env.num_envs * delay_reset_env_ratio)
        delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        delay_mask[:num_delay] = True

    max_count = int(env.num_envs * delay_reset_env_ratio)
    enabled = torch.where(delay_mask)[0]
    if enabled.numel() > max_count:
        delay_mask[enabled[max_count:]] = False

    num_delay = int(delay_mask.sum().item())
    if num_delay <= 0:
        return

    env.termination_manager = DelayedTerminationManager(
        base=env.termination_manager,
        delay_env_mask=delay_mask,
        max_delay_steps=max_delay_steps,
    )
    print(
        "[install_delayed_termination] DelayedTerminationManager installed: "
        f"{num_delay}/{env.num_envs} envs, max_delay_steps={max_delay_steps}"
    )


def tmt_cnd_disable_wrapper(func: Callable) -> Callable:
    """Add delayed-env masking to a normal termination function."""

    def wrapped(
        env: ManagerBasedRLEnv,
        *args,
        disable_on_delayed_termination_envs: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        terminated = func(env, *args, **kwargs)
        if not disable_on_delayed_termination_envs:
            return terminated
        # 
        mask = getattr(env.termination_manager, "delayed_termination_env_mask", None)
        if mask is None:
            return terminated
        return terminated & ~mask.to(device=terminated.device, dtype=torch.bool)

    wrapped.__name__ = func.__name__
    wrapped.__qualname__ = func.__qualname__
    wrapped.__doc__ = func.__doc__

    signature = inspect.signature(func)
    if "disable_on_delayed_termination_envs" not in signature.parameters:
        parameters = list(signature.parameters.values())
        parameters.append(
            inspect.Parameter(
                "disable_on_delayed_termination_envs",
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            )
        )
        wrapped.__signature__ = signature.replace(parameters=parameters)
    
    return wrapped


bad_anchor_pos = tmt_cnd_disable_wrapper(tmt_functions.bad_anchor_pos)
bad_anchor_pos_z_only = tmt_cnd_disable_wrapper(tmt_functions.bad_anchor_pos_z_only)
bad_anchor_ori = tmt_cnd_disable_wrapper(tmt_functions.bad_anchor_ori)
bad_motion_body_pos = tmt_cnd_disable_wrapper(tmt_functions.bad_motion_body_pos)
bad_motion_body_pos_z_only = tmt_cnd_disable_wrapper(tmt_functions.bad_motion_body_pos_z_only)


__all__ = [
    "DelayedTerminationManager",
    "install_delayed_termination",

    "tmt_cnd_disable_wrapper",
    "bad_anchor_pos",
    "bad_anchor_pos_z_only",
    "bad_anchor_ori",
    "bad_motion_body_pos",
    "bad_motion_body_pos_z_only",
]
