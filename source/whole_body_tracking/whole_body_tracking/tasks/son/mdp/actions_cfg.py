from isaaclab.envs.mdp.actions.actions_cfg import *

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .actions import JointPositionResidualsAction


@configclass
class JointPositionResidualsActionCfg(JointPositionActionCfg):
    """Config for joint position residual actions around motion-tracking targets."""

    class_type: type[ActionTerm] = JointPositionResidualsAction

