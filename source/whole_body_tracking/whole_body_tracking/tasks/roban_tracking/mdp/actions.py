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

from whole_body_tracking.tasks.roban_tracking.mdp import MotionCommand
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from. import actions_cfg


class JointPositionResidualsAction(JointPositionAction):

    cfg: actions_cfg.JointPositionResidualsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: actions_cfg.JointPositionResidualsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

    def process_actions(self, actions: torch.Tensor):
            cmd: MotionCommand = self._env.command_manager.get_term("motion")
            # store the raw actions
            self._raw_actions[:] = actions
            # apply the affine transformations
            self._processed_actions = self._raw_actions * self._scale + cmd.joint_pos
            # clip actions
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )
