from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

import torch

from . import rwd_functions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rwd_zero_delayed_wrapper(func: Callable) -> Callable:
    """Zero rewards for envs whose delayed termination is currently active."""

    def wrapped(
        env: ManagerBasedRLEnv,
        *args,
        disable_on_delayed_termination: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        reward = func(env, *args, **kwargs)
        
        if not disable_on_delayed_termination: return reward
        mask = getattr(env.termination_manager, "delayed_termination_active_mask", None)
        if mask is None: return reward
        # mask rewards
        reward = torch.where(
            mask.to(device=reward.device, dtype=torch.bool), 
            torch.zeros_like(reward), 
            reward
        )
        
        return reward

    wrapped.__name__ = func.__name__
    wrapped.__qualname__ = func.__qualname__
    wrapped.__doc__ = func.__doc__

    signature = inspect.signature(func)
    if "disable_on_delayed_termination" not in signature.parameters:
        parameters = list(signature.parameters.values())
        parameters.append(
            inspect.Parameter(
                "disable_on_delayed_termination",
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            )
        )
        wrapped.__signature__ = signature.replace(parameters=parameters)

    return wrapped


motion_global_anchor_position_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_global_anchor_position_error_exp
)
motion_global_anchor_position_z_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_global_anchor_position_z_error_exp
)
motion_global_anchor_orientation_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_global_anchor_orientation_error_exp
)
motion_relative_body_position_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_relative_body_position_error_exp
)
motion_relative_body_orientation_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_relative_body_orientation_error_exp
)
motion_global_body_linear_velocity_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_global_body_linear_velocity_error_exp
)
motion_global_body_angular_velocity_error_exp = rwd_zero_delayed_wrapper(
    rwd_functions.motion_global_body_angular_velocity_error_exp
)
feet_contact_time = rwd_zero_delayed_wrapper(
    rwd_functions.feet_contact_time
)


__all__ = [
    "rwd_zero_delayed_wrapper",
    "motion_global_anchor_position_error_exp",
    "motion_global_anchor_position_z_error_exp",
    "motion_global_anchor_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "feet_contact_time",
]
