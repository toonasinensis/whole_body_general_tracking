from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

from ..io import load_smpl_motion_file, load_robot_motion_file
from ..transform import pair_motion_paths, trim_paired_clips
from ..types import LoadReport, MotionData, PairedLoadResult, RobotLoadResult

"""
加载 motion data
"""

def load_smpl_clips(
    files: Sequence[str | MotionData],
    *,
    target_fps: float | None,
) -> list[MotionData]:
    if not files:
        raise ValueError("files list is empty")
    clips: list[MotionData] = []
    for entry in files:
        if isinstance(entry, MotionData):
            clips.append(entry)
        else:
            clips.append(load_smpl_motion_file(entry, target_fps=target_fps))
    return clips


def load_robot_clips(
    npz_files: list[str],
    *,
    target_fps: float,
    joint_names: Sequence[str] | None,
    motion_body_names: Sequence[str] | None,
    all_body_names: Sequence[str] | None,
    body_indexes: Sequence[int] | None,
    fallback_used: bool = False,
) -> RobotLoadResult:
    robot_clips = []
    loaded_files: list[str] = []
    skipped_files: list[str] = []
    warning_msgs: list[str] = []

    for path in npz_files:
        try:
            clip = load_robot_motion_file(
                path,
                target_fps=target_fps,
                joint_names=joint_names,
                motion_body_names=motion_body_names,
                all_body_names=all_body_names,
                body_indexes=body_indexes,
            )
            robot_clips.append(clip)
            loaded_files.append(path)
        except Exception as exc:
            msg = f"Skip invalid npz '{path}': {exc}"
            warnings.warn(msg)
            warning_msgs.append(msg)
            skipped_files.append(path)

    if not robot_clips:
        raise RuntimeError("No valid robot npz motions were loaded.")

    mode = "paired_fallback" if fallback_used else "robot_only"
    report = LoadReport(
        loaded_files=loaded_files,
        skipped_files=skipped_files,
        warnings=warning_msgs,
        fallback_used=fallback_used,
        mode=mode,
    )
    return RobotLoadResult(clips=robot_clips, report=report)


def load_paired_clips(
    npz_files: list[str | Path],
    smpl_dir: str,
    *,
    target_fps: float,
    joint_names: Sequence[str] | None,
    motion_body_names: Sequence[str] | None,
    all_body_names: Sequence[str] | None,
    body_indexes: Sequence[int] | None,
    max_frame_diff: int = 2,
) -> PairedLoadResult | None:
    path_pairs = pair_motion_paths(npz_files, smpl_dir)
    if not path_pairs:
        warnings.warn("No matching stems between NPZ files and SMPL PKL files. Falling back to robot-only loading.")
        return None

    paired_clips = []
    loaded_files: list[str] = []
    skipped_files: list[str] = []
    warning_msgs: list[str] = []

    for pair in path_pairs:
        try:
            robot = load_robot_motion_file(
                str(pair.robot_path),
                target_fps=target_fps,
                joint_names=joint_names,
                motion_body_names=motion_body_names,
                all_body_names=all_body_names,
                body_indexes=body_indexes,
            )
            smpl = load_smpl_motion_file(str(pair.smpl_path), target_fps=target_fps)
            trimmed = trim_paired_clips(pair.stem, robot, smpl, max_frame_diff=max_frame_diff)
            if trimmed is None:
                diff = abs(robot.num_frames - smpl.num_frames)
                msg = (
                    f"Skip '{pair.stem}': frame mismatch "
                    f"robot={robot.num_frames} smpl={smpl.num_frames} diff={diff}"
                )
                warnings.warn(msg)
                warning_msgs.append(msg)
                skipped_files.append(pair.stem)
                continue

            paired_clips.append(trimmed)
            loaded_files.append(str(pair.robot_path))
        except Exception as exc:
            msg = f"Skip '{pair.stem}': {exc}"
            warnings.warn(msg)
            warning_msgs.append(msg)
            skipped_files.append(pair.stem)

    if not paired_clips:
        warnings.warn("No paired motions loaded successfully after filtering. Falling back to robot-only loading.")
        return None

    report = LoadReport(
        loaded_files=loaded_files,
        skipped_files=skipped_files,
        warnings=warning_msgs,
        fallback_used=False,
        mode="paired",
        requested_pairs=len(path_pairs),
        loaded_pairs=len(paired_clips),
    )
    return PairedLoadResult(clips=paired_clips, report=report)
