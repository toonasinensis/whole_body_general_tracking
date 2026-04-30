"""UnifiedMotionLib: standalone motion database for robot NPZ + optional SMPL PKL data.

Handles file discovery (random sampling, eval-mode sequential slicing, multi-GPU
distributed splitting), frame-rate alignment, body-index filtering, and flat-
concatenated tensor storage with O(1) batch indexing.

SMPL data is loaded only when a PKL directory is provided.  When present, a
SmplMotionLib instance is composed in — SMPL loading logic stays in one place.
Without PKL files the class works as a pure robot motion loader.
"""

from __future__ import annotations

import numpy as np
import os
import random
import torch
import warnings
from collections.abc import Sequence
from pathlib import Path
from tqdm import tqdm

from smpl_math_utils import interpolate_linear

from .loader import MotionData, load_motion_file
from .motion_lib import SmplMotionLib

_ROBOT_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class UnifiedMotionLib:
    """Loads robot NPZ motion data with optional paired SMPL PKL data.

    SMPL data is managed by a composed ``SmplMotionLib`` instance that is only
    created when PKL files are available.  Robot and SMPL data share the same
    flat-index layout (``time_step_start_idx`` == ``SmplMotionLib._length_starts``),
    so a global frame index produced by MotionCommand is valid for both.

    After ``load_from_cfg()`` all attributes are populated and ready for O(1)
    tensor indexing.  The hot path contains zero conditional branches.

    Attributes exposed to ``MotionCommand`` (drop-in replacement for MotionLoader):
        joint_pos, joint_vel                   -- (total_frames, N_joints, 3)
        body_pos_w, body_quat_w, ...           -- properties, body-filtered
        anchor_pos_w, anchor_quat_w, ...       -- properties, single anchor body
        time_step_start_idx, time_step_end_idx -- (M,) torch.long
        frame_list, motion_num, fps, ...       -- scalar / 1-D metadata
        motion_ids_from_timestamps()           -- global-frame → motion-id lookup

    Optional SMPL attributes (None when no PKL dir supplied):
        smpl_joints  -- (total_frames, 24, 3)   via composed SmplMotionLib
        smpl_transl  -- (total_frames, 3)
        smpl_poses   -- (total_frames, 72)
    """

    def __init__(
        self,
        body_indexes: Sequence[int],
        motion_anchor_body_index: int,
        device: str = "cpu",
    ) -> None:
        self.body_indexes = list(body_indexes)
        self.motion_anchor_body_index = motion_anchor_body_index
        self.device = device

        # Robot data tensors
        self.joint_pos: torch.Tensor | None = None
        self.joint_vel: torch.Tensor | None = None
        self._body_pos_w_sel: torch.Tensor | None = None
        self._body_quat_w_sel: torch.Tensor | None = None
        self._body_lin_vel_w_sel: torch.Tensor | None = None
        self._body_ang_vel_w_sel: torch.Tensor | None = None

        # Frame-index metadata (mirrors MotionLoader interface exactly)
        self.fps: float | None = None
        self.time_step_total: int | None = None
        self.file_names: list[str] | None = None
        self.motion_num: int | None = None
        self.frame_list: torch.Tensor | None = None
        self.time_step_start_idx: torch.Tensor | None = None
        self.time_step_end_idx: torch.Tensor | None = None

        # Composed SMPL lib — only instantiated when PKL dir is given
        self._smpl_lib: SmplMotionLib | None = None

        # Sequential counter for eval_mode file slicing
        self._sample_counter: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def load_from_cfg(self, cfg) -> None:
        """One-shot loader.  Reads from a MotionCommandCfg (duck-typed).

        Fields consumed from cfg:
            motion_file, max_motion_num, dataset_txt
            eval_mode, distributed, local_rank, total_rank
            smpl_file_path  (optional — omit or set to None/empty for robot-only)
        """
        target_fps = 50.0

        npz_files = self._find_npz_files(
            cfg.motion_file,
            cfg.max_motion_num,
            cfg.dataset_txt,
            cfg.eval_mode,
            cfg.distributed,
            cfg.local_rank,
            cfg.total_rank,
        )

        smpl_dir = getattr(cfg, "smpl_file_path", None)
        if smpl_dir and Path(smpl_dir).is_dir():
            self._load_paired(npz_files, smpl_dir, cfg.motion_file, target_fps)
        else:
            self._load_robot(npz_files, cfg.motion_file, target_fps)

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _find_npz_files(
        self,
        dir_path: str,
        motion_num: int,
        dataset_txt: str | None,
        eval_mode: bool,
        distributed: bool,
        local_rank: int,
        total_rank: int,
    ) -> list[str]:
        """Discover and subset NPZ files according to sampling strategy.

        Priority order (matching original MotionLoader._find_npz_files):
          1. motion_num sampling (random or eval sequential)
          2. distributed split across ranks
          3. load all
        """
        if dataset_txt is not None:
            with open(dataset_txt) as f:
                relative_paths = [line.strip() for line in f if line.strip()]
            npz_files = [dir_path + "/" + p for p in relative_paths]
            random.seed(42)
            random.shuffle(npz_files)
        else:
            npz_files = sorted(str(p) for p in Path(dir_path).rglob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No .npz files found in {dir_path}")

        if len(npz_files) > motion_num != -1:
            if eval_mode:
                start = min(self._sample_counter * motion_num, len(npz_files) - 1)
                end = min(start + motion_num, len(npz_files) - 1)
                if start >= len(npz_files) - 1:
                    raise ValueError("eval_mode sampling exhausted all data.")
                sampled = npz_files[start:end]
                print("eval_mode process: ", float(start) / len(npz_files))
                self._sample_counter += 1
            else:
                sampled = sorted(random.sample(npz_files, motion_num))

        elif distributed:
            per_rank = len(npz_files) // total_rank
            start = per_rank * local_rank
            end = per_rank * (local_rank + 1)
            print(f"[GPU {local_rank}/{total_rank}] 数据 {start}→{end}，共 {per_rank} 个")
            sampled = npz_files[start:end]

        else:
            print(f"加载全部共 {len(npz_files)} 个 NPZ 动作数据")
            sampled = npz_files

        return sampled

    # ------------------------------------------------------------------
    # Loading paths
    # ------------------------------------------------------------------

    def _load_robot(self, npz_files: list[str], base_dir: str, target_fps: float) -> None:
        """Load robot NPZ files only.  _smpl_lib remains None."""
        robot_lists: dict[str, list[torch.Tensor]] = {k: [] for k in _ROBOT_KEYS}
        frame_counts: list[int] = []
        rel_names: list[str] = []

        for p in tqdm(npz_files, desc="Loading robot NPZ"):
            raw = np.load(str(p), allow_pickle=True)
            npz_fps = float(raw["fps"])
            tensors = {k: torch.from_numpy(np.asarray(raw[k], dtype=np.float32)) for k in _ROBOT_KEYS}
            if abs(npz_fps - target_fps) > 1e-3:
                tensors = {k: interpolate_linear(v, npz_fps, target_fps) for k, v in tensors.items()}
            n = tensors["joint_pos"].shape[0]
            for k in _ROBOT_KEYS:
                robot_lists[k].append(tensors[k][:n])
            frame_counts.append(n)
            rel_names.append(os.path.relpath(str(p), base_dir))

        self._set_robot_tensors(robot_lists, frame_counts, rel_names, target_fps)

    def _load_paired(
        self,
        npz_files: list[str],
        smpl_dir: str,
        base_dir: str,
        target_fps: float,
    ) -> None:
        """Load paired robot NPZ + SMPL PKL files, aligned to the same frame count.

        For each valid pair:
          1. Load robot NPZ (resample to target_fps if needed) → robot_n frames
          2. Load SMPL PKL via load_motion_file (resampled to target_fps) → smpl_n frames
          3. n = min(robot_n, smpl_n); skip if |robot_n - smpl_n| > 2
          4. Trim both to n frames

        Trimmed MotionData objects are passed to SmplMotionLib.load_motions() so that
        all SMPL loading logic stays inside SmplMotionLib.
        """
        smpl_map = {Path(p).stem: str(p) for p in Path(smpl_dir).glob("*.pkl")}
        robot_map = {Path(p).stem: str(p) for p in npz_files}
        common = sorted(set(robot_map) & set(smpl_map))

        if not common:
            raise ValueError(
                "No matching stems between NPZ dir and PKL dir.\n"
                f"  NPZ stems (first 5): {sorted(robot_map)[:5]}\n"
                f"  PKL stems (first 5): {sorted(smpl_map)[:5]}"
            )

        robot_lists: dict[str, list[torch.Tensor]] = {k: [] for k in _ROBOT_KEYS}
        valid_smpl_motions: list[MotionData] = []
        frame_counts: list[int] = []
        rel_names: list[str] = []
        skipped = 0

        for stem in tqdm(common, desc="Loading paired motions"):
            try:
                raw = np.load(robot_map[stem], allow_pickle=True)
                npz_fps = float(raw["fps"])
                tensors = {k: torch.from_numpy(np.asarray(raw[k], dtype=np.float32)) for k in _ROBOT_KEYS}
                if abs(npz_fps - target_fps) > 1e-3:
                    tensors = {k: interpolate_linear(v, npz_fps, target_fps) for k, v in tensors.items()}
                robot_n = tensors["joint_pos"].shape[0]

                smpl_data = load_motion_file(smpl_map[stem], target_fps=target_fps)
                smpl_n = smpl_data.num_frames

                diff = abs(robot_n - smpl_n)
                if diff > 2:
                    warnings.warn(f"Skip '{stem}': frame mismatch robot={robot_n} smpl={smpl_n} diff={diff}")
                    skipped += 1
                    continue
                n = min(robot_n, smpl_n)

                for k in _ROBOT_KEYS:
                    robot_lists[k].append(tensors[k][:n])

                # Trim MotionData to n frames; pass to SmplMotionLib below
                valid_smpl_motions.append(
                    MotionData(
                        pose_aa=smpl_data.pose_aa[:n],
                        smpl_joints=smpl_data.smpl_joints[:n],
                        transl=smpl_data.transl[:n],
                        fps=smpl_data.fps,
                        source_fps=smpl_data.source_fps,
                        num_frames=n,
                        duration=(n - 1) / smpl_data.fps if n > 1 else 0.0,
                    )
                )
                frame_counts.append(n)
                rel_names.append(os.path.relpath(robot_map[stem], base_dir))

            except Exception as exc:
                warnings.warn(f"Skip '{stem}': {exc}")
                skipped += 1

        if not frame_counts:
            raise RuntimeError("No paired motions loaded successfully.")

        print(f"[UnifiedMotionLib] Loaded {len(frame_counts)}/{len(common)} paired motions, skipped {skipped}")

        self._set_robot_tensors(robot_lists, frame_counts, rel_names, target_fps)

        # Delegate SMPL storage to SmplMotionLib; pass pre-trimmed MotionData objects
        # so SmplMotionLib.load_motions() skips re-loading from disk.
        self._smpl_lib = SmplMotionLib(up_axis="yup")
        self._smpl_lib.load_motions(valid_smpl_motions, target_fps=None)

        if self.device != "cpu":
            self._smpl_lib.to_device(self.device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_robot_tensors(
        self,
        robot_lists: dict[str, list[torch.Tensor]],
        frame_counts: list[int],
        rel_names: list[str],
        target_fps: float,
        batch_size: int = 1024,
    ) -> None:
        """Cat, upload to device in batches, filter body indexes, build frame index."""

        def cat_to_device(lst: list[torch.Tensor]) -> torch.Tensor:
            full = torch.cat(lst, dim=0)
            if self.device == "cpu":
                return full
            chunks = []
            for i in range(0, full.shape[0], batch_size):
                chunks.append(full[i : i + batch_size].to(self.device))
            return torch.cat(chunks, dim=0)

        self.joint_pos = cat_to_device(robot_lists["joint_pos"])
        self.joint_vel = cat_to_device(robot_lists["joint_vel"])
        body_pos = cat_to_device(robot_lists["body_pos_w"])
        body_quat = cat_to_device(robot_lists["body_quat_w"])
        body_lvel = cat_to_device(robot_lists["body_lin_vel_w"])
        body_avel = cat_to_device(robot_lists["body_ang_vel_w"])

        self._body_pos_w_sel = body_pos[:, self.body_indexes]
        self._body_quat_w_sel = body_quat[:, self.body_indexes]
        self._body_lin_vel_w_sel = body_lvel[:, self.body_indexes]
        self._body_ang_vel_w_sel = body_avel[:, self.body_indexes]

        fc = torch.tensor(frame_counts, dtype=torch.long, device=self.device)
        self.frame_list = fc
        self.motion_num = len(frame_counts)
        self.fps = float(target_fps)
        self.time_step_total = int(fc.sum())
        self.time_step_end_idx = torch.cumsum(fc, dim=0)
        self.time_step_start_idx = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=self.device), self.time_step_end_idx[:-1]]
        )
        self.file_names = rel_names

    # ------------------------------------------------------------------
    # Properties (body-level views)
    # ------------------------------------------------------------------

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w_sel

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w_sel

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w_sel

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w_sel

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._body_pos_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._body_quat_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w_sel[:, self.motion_anchor_body_index]

    @property
    def anchor_pos_z(self) -> torch.Tensor:
        return self.anchor_pos_w[:, 2:3]

    # ------------------------------------------------------------------
    # SMPL data — exposed via composed SmplMotionLib
    # ------------------------------------------------------------------

    @property
    def smpl_joints(self) -> torch.Tensor | None:
        """(total_frames, 24, 3).  None if no PKL dir was given."""
        return self._smpl_lib.joints_flat if self._smpl_lib is not None else None

    @property
    def smpl_transl(self) -> torch.Tensor | None:
        """(total_frames, 3).  None if no PKL dir was given."""
        return self._smpl_lib.transl_flat if self._smpl_lib is not None else None

    @property
    def smpl_poses(self) -> torch.Tensor | None:
        """(total_frames, 72).  None if no PKL dir was given."""
        return self._smpl_lib.poses_flat if self._smpl_lib is not None else None

    # ------------------------------------------------------------------
    # SMPL query methods — (motion_ids, motion_steps) → data
    # ------------------------------------------------------------------
    # time_step_start_idx == SmplMotionLib._length_starts (same roll-cumsum formula,
    # same frame_counts), so the flat index is valid for both robot and SMPL tensors.

    def _smpl_idx(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        return self.time_step_start_idx[motion_ids] + motion_steps

    def _require_smpl(self) -> SmplMotionLib:
        if self._smpl_lib is None:
            raise RuntimeError("SMPL data not loaded (no smpl_file_path provided).")
        return self._smpl_lib

    def get_smpl_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 24, 3) joint positions."""
        return self._require_smpl().get_smpl_joints(motion_ids, motion_steps)

    def get_smpl_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 3) root translation."""
        return self._require_smpl().get_smpl_transl(motion_ids, motion_steps)

    def get_smpl_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 72) axis-angle pose."""
        return self._require_smpl().get_smpl_pose(motion_ids, motion_steps)
        # return self._require_smpl().poses_flat[self._smpl_idx(motion_ids, motion_steps)]

    def get_smpl_global_position(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 24, 3) global joint positions."""
        return self._require_smpl().get_smpl_global_position(motion_ids, motion_steps)

    # ------------------------------------------------------------------
    # Frame-index → motion-id conversion
    # ------------------------------------------------------------------

    def motion_ids_from_timestamps(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Map global frame indices to motion ids.

        Args:
            timestamps: (num_envs,) global frame indices

        Returns:
            (num_envs,) motion ids
        """
        if self.time_step_end_idx is None:
            raise RuntimeError("UnifiedMotionLib not loaded. Call load_from_cfg() first.")
        if timestamps.dtype != torch.long:
            timestamps = timestamps.long()
        timestamps = torch.clamp(timestamps, min=0, max=int(self.time_step_total) - 1)
        motion_ids = torch.bucketize(timestamps, self.time_step_end_idx, right=True)
        return torch.clamp(motion_ids, min=0, max=self.motion_num - 1)
