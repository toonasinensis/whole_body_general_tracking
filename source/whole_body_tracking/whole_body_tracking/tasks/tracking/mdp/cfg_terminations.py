from __future__ import annotations

import whole_body_tracking.tasks.tracking.mdp as mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from . import tmt_modules as tmt


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=tmt.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.4},
    )
    anchor_ori = DoneTerm(
        func=tmt.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.0},
    )
    ee_body_pos = DoneTerm(
        func=tmt.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.4,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
            "disable_on_delayed_termination_envs": True,
        },
    )


__all__ = ["TerminationsCfg"]
