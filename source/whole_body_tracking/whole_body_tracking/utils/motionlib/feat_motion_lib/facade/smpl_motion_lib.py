from __future__ import annotations

from collections.abc import Sequence

import torch

from ..config import SmplLoadConfig
from ..errors import MotionNotLoadedError
from ..types import LoadReport, SmplMotionClip

from ..store import SmplMotionStore

from .config_adapter import cfg_to_smpl_load_config
from .loading import load_smpl_clips
from .orchestration import execute_smpl_load

class SmplMotionLib:
    def __init__(self, up_axis: str = "yup", device: str = "cpu") -> None:
        self.up_axis = up_axis
        self.device = device
        self._store: SmplMotionStore | None = None
        self._last_report: LoadReport | None = None

    def reset(self) -> None:
        self._store = None
        self._last_report = None

    def load_motions(self, files: Sequence[str | SmplMotionClip], target_fps: float | None = None) -> None:
        self.reset()
        clips = load_smpl_clips(files, target_fps=target_fps)
        self._store = SmplMotionStore(clips, up_axis=self.up_axis, device=self.device)
        self._last_report = LoadReport(
            loaded_files=[entry for entry in files if isinstance(entry, str)],
            skipped_files=[],
            warnings=[],
            fallback_used=False,
            mode="smpl_only",
        )

    def load(self, load_config: SmplLoadConfig) -> LoadReport:
        self.reset()
        outcome = execute_smpl_load(load_config)
        self.up_axis = load_config.load.up_axis
        self.device = load_config.load.device
        self._store = outcome.store
        self._last_report = outcome.report
        return outcome.report

    def load_from_cfg(self, cfg) -> LoadReport:
        load_config = cfg_to_smpl_load_config(cfg, device=self.device)
        return self.load(load_config)

    def _require_store(self) -> SmplMotionStore:
        if self._store is None:
            raise MotionNotLoadedError("No motions loaded. Call load_motions(), load(), or load_from_cfg() first.")
        return self._store

    def to_device(self, device: str | torch.device) -> None:
        self.device = str(device)
        self._require_store().to_device(device)

    @property
    def last_report(self) -> LoadReport | None:
        return self._last_report

    @property
    def poses_flat(self) -> torch.Tensor:
        return self._require_store().poses_flat

    @property
    def joints_flat(self) -> torch.Tensor:
        return self._require_store().joints_flat

    @property
    def transl_flat(self) -> torch.Tensor:
        return self._require_store().transl_flat

    def get_smpl_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_store().get_pose(motion_ids, motion_steps)

    def get_smpl_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_store().get_joints(motion_ids, motion_steps)

    def get_smpl_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_store().get_transl(motion_ids, motion_steps)

    def get_smpl_global_position(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_store().get_global_positions(motion_ids, motion_steps)

    def get_smpl_global_rotations(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self._require_store().get_global_rotations(motion_ids, motion_steps)

    def get_num_motions(self) -> int:
        return self._require_store().get_num_motions()

    def get_num_frames(self, motion_id: int) -> int:
        return self._require_store().get_num_frames(motion_id)

    def get_motion_fps(self, motion_id: int) -> float:
        return self._require_store().get_motion_fps(motion_id)

    def get_motion_duration(self, motion_id: int) -> float:
        return self._require_store().get_motion_duration(motion_id)

    def sample_random(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._require_store().sample_random(batch_size)
