from __future__ import annotations

from ..errors import MotionValidationError
from .robot_store import RobotMotionStore
from .smpl_store import SmplMotionStore


class PairedMotionStore:
    def __init__(self, robot_store: RobotMotionStore, smpl_store: SmplMotionStore) -> None:
        self.robot_store = robot_store
        self.smpl_store = smpl_store
        self.validate_shared_index()

    def validate_shared_index(self) -> None:
        robot_index = self.robot_store.index
        smpl_index = self.smpl_store.index
        if robot_index.total_frames != smpl_index.total_frames:
            raise MotionValidationError(
                f"Paired stores total_frames mismatch: robot={robot_index.total_frames} smpl={smpl_index.total_frames}"
            )
        if robot_index.frame_counts.numel() != smpl_index.frame_counts.numel():
            raise MotionValidationError(
                "Paired stores motion count mismatch: "
                f"robot={robot_index.frame_counts.numel()} smpl={smpl_index.frame_counts.numel()}"
            )
        if not robot_index.frame_counts.equal(smpl_index.frame_counts):
            raise MotionValidationError("Paired stores frame_counts mismatch.")
        if not robot_index.start_idx.equal(smpl_index.start_idx):
            raise MotionValidationError("Paired stores start_idx mismatch.")
        if not robot_index.end_idx.equal(smpl_index.end_idx):
            raise MotionValidationError("Paired stores end_idx mismatch.")
