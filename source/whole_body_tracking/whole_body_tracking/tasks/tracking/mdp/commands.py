"""Compatibility exports for motion command classes.

The implementation lives in ``mdp.cmd_modules.motion_command``.
"""

from .cmd_modules.motion_command import MotionCommand, MotionCommandCfg, MotionLoader

__all__ = ["MotionCommand", "MotionCommandCfg", "MotionLoader"]
