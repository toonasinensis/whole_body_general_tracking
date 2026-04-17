# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch
from dataclasses import MISSING

# from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.utils import configclass

# from .actuator_pd import DelayedImplicitActuator
from .actuator_pd import DelayedPDActuator_KuavoS52, DelayedPDActuator_RobanS22


@configclass
class DelayedPDActuatorCfg_RobanS22(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    effort_limit_rated: float | dict[str, float] = MISSING
    class_type: type = DelayedPDActuator_RobanS22
    friction_static: float | dict[str, float] = 0
    activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0


@configclass
class DelayedPDActuatorCfg_KuavoS52(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedPDActuator_KuavoS52
    friction_static: float | dict[str, float] = 0
    activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0
