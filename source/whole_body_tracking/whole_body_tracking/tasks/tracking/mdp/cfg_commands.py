from __future__ import annotations

from isaaclab.utils import configclass

from .cmd_modules import MotionCommandCfg


VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    motion = MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        max_motion_num=999999,
        motion_file=None,
        pose_range={
            "x": (-0.0, 0.0),
            "y": (-0.0, 0.0),
            "z": (0.05, 0.1),
            "roll": (-1.0, 1.0),
            "pitch": (-1.0, 1.0),
            "yaw": (-0.0, 0.0),
        },
        pose_range_env_ratio=0.3,
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.0, 0.0),
    )


__all__ = ["CommandsCfg", "VELOCITY_RANGE"]
