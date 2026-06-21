"""This sub-module contains the functions that are specific to the locomotion environments."""

import isaaclab.utils.math as _isaac_math

# Isaac Lab 2.1 / Isaac Sim 4.5 renamed quat_apply_inverse to quat_rotate_inverse.
if not hasattr(_isaac_math, "quat_apply_inverse"):
    _isaac_math.quat_apply_inverse = _isaac_math.quat_rotate_inverse

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .cmd_modules import *  # noqa: F401, F403
from .evt_modules import *  # noqa: F401, F403
from .obs_modules import *  # noqa: F401, F403
from .rwd_modules import *  # noqa: F401, F403
from .tmt_modules import *  # noqa: F401, F403
