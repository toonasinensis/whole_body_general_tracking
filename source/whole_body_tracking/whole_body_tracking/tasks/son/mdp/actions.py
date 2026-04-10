from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs.mdp.actions import JointAction, JointPositionAction

from whole_body_tracking.tasks.son.mdp.commands import MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from . import actions_cfg


class JointPositionResidualsAction(JointPositionAction):
    """Joint position action defined as residuals around a motion-tracking target."""

    cfg: "actions_cfg.JointPositionResidualsActionCfg"
    _asset: Articulation
    _scale: torch.Tensor | float
    _clip: torch.Tensor
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: "actions_cfg.JointPositionResidualsActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
        cmd: MultiMotionCommand = self._env.command_manager.get_term("motion")
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations around motion target joint positions
        self._processed_actions = self._raw_actions * self._scale + cmd.joint_pos
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
