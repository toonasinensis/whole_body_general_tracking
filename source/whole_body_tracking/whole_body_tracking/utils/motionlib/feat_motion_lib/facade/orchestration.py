from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..config import SmplLoadConfig, UnifiedLoadConfig
from ..types import LoadReport, UnifiedMotionState

from ..io import discover_npz_files
from ..store import PairedMotionStore, RobotMotionStore, SmplMotionStore

from .loading import load_paired_clips, load_robot_clips, load_smpl_clips

if TYPE_CHECKING:
    from .smpl_motion_lib import SmplMotionLib


@dataclass
class UnifiedLoadOutcome:
    state: UnifiedMotionState
    smpl_lib: SmplMotionLib | None
    report: LoadReport
    next_sample_counter: int


@dataclass
class SmplLoadOutcome:
    store: SmplMotionStore
    report: LoadReport


def execute_smpl_load(load_config: SmplLoadConfig) -> SmplLoadOutcome:
    clips = load_smpl_clips(
        [str(path) for path in load_config.motion_files],
        target_fps=load_config.load.target_fps,
    )
    store = SmplMotionStore(
        clips,
        up_axis=load_config.load.up_axis,
        device=load_config.load.device,
    )
    report = LoadReport(
        loaded_files=[str(path) for path in load_config.motion_files],
        skipped_files=[],
        warnings=[],
        fallback_used=False,
        mode="smpl_only",
    )
    return SmplLoadOutcome(store=store, report=report)


def execute_unified_load(
    *,
    body_indexes: list[int],
    joint_names: list[str] | None,
    motion_body_names: list[str] | None,
    all_body_names: list[str] | None,
    load_config: UnifiedLoadConfig,
) -> UnifiedLoadOutcome:
    npz_files, next_counter = discover_npz_files(load_config.discovery)
    base_dir = str(load_config.discovery.motion_dir)
    smpl_dir = load_config.smpl_dir
    smpl_requested = bool(smpl_dir and smpl_dir.is_dir())

    if smpl_requested:
        paired_result = load_paired_clips(
            npz_files,
            str(smpl_dir),
            target_fps=load_config.load.target_fps,
            joint_names=joint_names,
            motion_body_names=motion_body_names,
            all_body_names=all_body_names,
            body_indexes=body_indexes,
            max_frame_diff=load_config.max_frame_diff,
        )
        if paired_result is not None:
            from .smpl_motion_lib import SmplMotionLib

            robot_clips = [pair.robot for pair in paired_result.clips]
            smpl_clips = [pair.smpl for pair in paired_result.clips]
            robot_store = RobotMotionStore(robot_clips, device=load_config.load.device)
            smpl_lib = SmplMotionLib(up_axis=load_config.load.up_axis, device=load_config.load.device)
            smpl_lib.load_motions(smpl_clips, target_fps=None)
            paired_store = PairedMotionStore(robot_store, smpl_lib._require_store())
            return UnifiedLoadOutcome(
                state=paired_store.robot_store.build_state(base_dir),
                smpl_lib=smpl_lib,
                report=paired_result.report,
                next_sample_counter=next_counter,
            )

    robot_result = load_robot_clips(
        npz_files,
        target_fps=load_config.load.target_fps,
        joint_names=joint_names,
        motion_body_names=motion_body_names,
        all_body_names=all_body_names,
        body_indexes=body_indexes,
        fallback_used=smpl_requested,
    )
    robot_store = RobotMotionStore(robot_result.clips, device=load_config.load.device)
    return UnifiedLoadOutcome(
        state=robot_store.build_state(base_dir),
        smpl_lib=None,
        report=robot_result.report,
        next_sample_counter=next_counter,
    )
