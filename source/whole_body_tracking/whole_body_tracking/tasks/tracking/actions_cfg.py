from __future__ import annotations

import whole_body_tracking.tasks.tracking.mdp as mdp
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


__all__ = ["ActionsCfg"]
