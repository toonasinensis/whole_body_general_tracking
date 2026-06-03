from .config import DiscoveryOptions, LoadOptions, SmplLoadConfig, UnifiedLoadConfig
from .types import LoadReport, MotionData, MotionIndex, PairedMotionData, PairedPath, RobotMotionData

__all__ = [
    "DiscoveryOptions",
    "LoadOptions",
    "SmplLoadConfig",
    "UnifiedLoadConfig",
    "LoadReport",
    "MotionData",
    "MotionIndex",
    "PairedMotionData",
    "PairedPath",
    "RobotMotionData",
    "SmplMotionLib",
    "UnifiedMotionLib",
    "discover_npz_files",
    "load_motion_file",
    "load_pkl",
    "load_robot_motion_file",
    "resample_30hz_to_50hz_and_save",
    "resample_and_save_motion_file",
]


def __getattr__(name: str):
    if name in {
        "discover_npz_files",
        "load_motion_file",
        "load_pkl",
        "load_robot_motion_file",
        "discover_pkl_files",
        "resample_30hz_to_50hz_and_save",
        "resample_and_save_motion_file",
    }:
        from .api import (
            discover_pkl_files,
            discover_npz_files,
            load_motion_file,
            load_pkl,
            load_robot_motion_file,
            resample_30hz_to_50hz_and_save,
            resample_and_save_motion_file,
        )

        mapping = {
            "discover_pkl_files": discover_pkl_files,
            "discover_npz_files": discover_npz_files,
            "load_motion_file": load_motion_file,
            "load_pkl": load_pkl,
            "load_robot_motion_file": load_robot_motion_file,
            "resample_30hz_to_50hz_and_save": resample_30hz_to_50hz_and_save,
            "resample_and_save_motion_file": resample_and_save_motion_file,
        }
        return mapping[name]
    if name in {"SmplMotionLib", "UnifiedMotionLib"}:
        from .facade import SmplMotionLib, UnifiedMotionLib

        mapping = {
            "SmplMotionLib": SmplMotionLib,
            "UnifiedMotionLib": UnifiedMotionLib,
        }
        return mapping[name]
    raise AttributeError(name)
