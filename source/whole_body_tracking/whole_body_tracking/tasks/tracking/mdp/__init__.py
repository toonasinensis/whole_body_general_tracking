"""This sub-module contains the functions that are specific to the locomotion environments."""

import isaaclab.utils.math as _isaac_math

# Isaac Lab 2.1 / Isaac Sim 4.5 renamed quat_apply_inverse to quat_rotate_inverse.
if not hasattr(_isaac_math, "quat_apply_inverse"):
    _isaac_math.quat_apply_inverse = _isaac_math.quat_rotate_inverse

from isaaclab.envs.mdp import *  # noqa: F401, F403

from whole_body_tracking.tasks.tracking.mdp import *  # noqa: F401, F403

from .commands import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
