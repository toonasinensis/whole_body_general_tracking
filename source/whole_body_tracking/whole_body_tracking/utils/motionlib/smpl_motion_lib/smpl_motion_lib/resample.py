"""Motion file resampling utilities."""

from __future__ import annotations

import joblib

from .loader import load_motion_file


def resample_and_save_motion_file(
    input_path: str,
    output_path: str,
    target_fps: float,
) -> None:
    """Resample one motion pkl to target_fps and save a new pkl.

    The output pkl keeps keys compatible with the rest of this repo:
    `pose_aa`, `smpl_joints`, `transl`, `fps`.
    """
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
    """Convenience helper for 30Hz -> 50Hz resampling and saving."""
    resample_and_save_motion_file(input_path, output_path, target_fps=50.0)
