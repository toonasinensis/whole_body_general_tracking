from typing import TYPE_CHECKING

from .loader import MotionData, load_motion_file
from .motion_lib import SmplMotionLib
from .resample import resample_30hz_to_50hz_and_save, resample_and_save_motion_file
from .unified_motion_lib import UnifiedMotionLib

if TYPE_CHECKING:
    from .visualization import visualize_motion, visualize_motion_batch, visualize_motion_directory, visualize_motion_file

_VISUALIZATION_EXPORTS = {
    "visualize_motion",
    "visualize_motion_batch",
    "visualize_motion_directory",
    "visualize_motion_file",
}

__all__ = [
    "MotionData",
    "SmplMotionLib",
    "UnifiedMotionLib",
    "load_motion_file",
    "resample_30hz_to_50hz_and_save",
    "resample_and_save_motion_file",
    "visualize_motion",
    "visualize_motion_batch",
    "visualize_motion_directory",
    "visualize_motion_file",
]


def __getattr__(name: str):
    if name in _VISUALIZATION_EXPORTS:
        from . import visualization

        value = getattr(visualization, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
