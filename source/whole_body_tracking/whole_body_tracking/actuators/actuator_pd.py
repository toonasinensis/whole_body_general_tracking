# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DelayedPDActuator
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_cfg import DelayedPDActuatorCfg_RobanS22
    from .actuator_cfg import DelayedPDActuatorCfg_KuavoS52

import re


class DelayedPDActuator_RobanS22(DelayedPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg_RobanS22

    def __init__(self, cfg: DelayedPDActuatorCfg_RobanS22, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # build saturation effort limit tensor
        self._saturation_effort = self._build_effort_limit_tensor()

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._joint_vel[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        # calculate the desired joint torques with friction compensation
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
            - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)
        )

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _build_effort_limit_tensor(self):
        effort_cfg = self.cfg.effort_limit_sim
        limits = []

        for name in self.joint_names:
            matched = False
            for pattern, value in effort_cfg.items():
                if re.fullmatch(pattern.replace("[l,r]", "[lr]"), name):
                    limits.append(value)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No effort limit for joint: {name}")

        limits_tensor = torch.ones(
            (1, len(limits)),  # 直接创建 (1, num_dof) 形状
            device=self.computed_effort.device,
            dtype=self.computed_effort.dtype,
            requires_grad=False,  # 显式禁用梯度
        )
        for i, limit in enumerate(limits):
            limits_tensor[0, i] = limit
        return limits_tensor

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # safe_vel_ratio = torch.where(
        #     self.velocity_limit > 1e-6,  # 避免除以接近零的值
        #     self._joint_vel / self.velocity_limit,
        #     torch.zeros_like(self._joint_vel),
        # )
        # -- max limit
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)
        # -- min limit
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        # clip the torques based on the motor limits
        return torch.clip(effort, min=min_effort, max=max_effort)


class DelayedPDActuator_KuavoS52(DelayedPDActuator):
    """Ideal PD actuator with delayed command application.

    This class extends the :class:`IdealPDActuator` class by adding a delay to the actuator commands. The delay
    is implemented using a circular buffer that stores the actuator commands for a certain number of physics steps.
    The most recent actuation value is pushed to the buffer at every physics step, but the final actuation value
    applied to the simulation is lagged by a certain number of physics steps.

    The amount of time lag is configurable and can be set to a random value between the minimum and maximum time
    lag bounds at every reset. The minimum and maximum time lag values are set in the configuration instance passed
    to the class.
    """

    cfg: DelayedPDActuatorCfg_KuavoS52

    def __init__(self, cfg: DelayedPDActuatorCfg_KuavoS52, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.friction_static = self._parse_joint_parameter(self.cfg.friction_static, 0.0)
        self.activation_vel = self._parse_joint_parameter(self.cfg.activation_vel, torch.inf)
        self.friction_dynamic = self._parse_joint_parameter(self.cfg.friction_dynamic, 0.0)
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # build saturation effort limit tensor
        self._saturation_effort = self._build_effort_limit_tensor()

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._joint_vel[env_ids] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # apply delay based on the delay the model for all the setpoints
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # compute errors
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel

        # calculate the desired joint torques with friction compensation
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + control_action.joint_efforts
            - (self.friction_static * torch.tanh(joint_vel / self.activation_vel) + self.friction_dynamic * joint_vel)
        )

        # clip the torques based on the motor limits
        self.applied_effort = self._clip_effort(self.computed_effort)

        # set the computed actions back into the control action
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def _build_effort_limit_tensor(self):
        effort_cfg = self.cfg.effort_limit_sim
        limits = []

        for name in self.joint_names:
            matched = False
            for pattern, value in effort_cfg.items():
                if re.fullmatch(pattern.replace("[l,r]", "[lr]"), name):
                    limits.append(value)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"No effort limit for joint: {name}")

        limits_tensor = torch.ones(
            (1, len(limits)),  # 直接创建 (1, num_dof) 形状
            device=self.computed_effort.device,
            dtype=self.computed_effort.dtype,
            requires_grad=False,  # 显式禁用梯度
        )
        for i, limit in enumerate(limits):
            limits_tensor[0, i] = limit
        return limits_tensor

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # 安全除法，避免除以零
        # safe_vel_ratio = torch.where(
        #     self.velocity_limit > 1e-6,  # 避免除以接近零的值
        #     self._joint_vel / self.velocity_limit,
        #     torch.zeros_like(self._joint_vel),
        # )
        # -- max limit
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        max_effort = torch.clip(max_effort, min=self._zeros_effort, max=self.effort_limit)
        # -- min limit
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        min_effort = torch.clip(min_effort, min=-self.effort_limit, max=self._zeros_effort)

        # clip the torques based on the motor limits
        return torch.clip(effort, min=min_effort, max=max_effort)
