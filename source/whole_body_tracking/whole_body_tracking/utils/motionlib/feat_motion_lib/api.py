from __future__ import annotations

import joblib

from .config import DiscoveryOptions
from .facade import SmplMotionLib, UnifiedMotionLib
from .io import discover_npz_files as _discover_npz_files
from .io import discover_pkl_files
from .io import load_motion_file, load_pkl, load_robot_motion_file


def discover_npz_files(options: DiscoveryOptions) -> tuple[list[str], int]:
    return _discover_npz_files(options)


def resample_and_save_motion_file(input_path: str, output_path: str, target_fps: float) -> None:
    motion = load_motion_file(input_path, target_fps=target_fps)
    data = {
        "pose_aa": motion.pose_aa.cpu().numpy(),
        "smpl_joints": motion.smpl_joints.cpu().numpy(),
        "transl": motion.transl.cpu().numpy(),
        "fps": float(motion.fps),
        "source_fps": float(motion.source_fps),
    }
    joblib.dump(data, output_path)


def resample_30hz_to_50hz_and_save(input_path: str, output_path: str) -> None:
    resample_and_save_motion_file(input_path, output_path, target_fps=50.0)
