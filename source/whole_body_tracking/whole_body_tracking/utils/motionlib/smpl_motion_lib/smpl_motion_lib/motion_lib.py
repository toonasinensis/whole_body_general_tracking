"""SmplMotionLib: flat-concatenated storage with O(1) batch indexing.

Design mirrors gear_sonic/utils/motion_lib/motion_lib_base.py lines 626-636 and 1524-1528.
"""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence

from smpl_math_utils import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

from .loader import MotionData, load_motion_file

# Y-up → Z-up: (x, y, z) → (x, −z, y)
_Y_UP_TO_Z_UP = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)

# SMPL 24-joint kinematic tree (parent index for each joint, -1 = root)
_SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21]


class SmplMotionLib:
    """Loads and queries SMPL motion sequences stored in pkl files.

    Internal storage uses flat-concatenated tensors with cumulative offset
    (`_length_starts`) for O(1) vectorised batch access — identical to the
    indexing pattern in motion_lib_base.py.

    All query methods accept and return ``torch.Tensor``.  Tensors are kept on
    CPU after loading; move them to a device via ``to_device()`` if needed.

    Args:
        up_axis: coordinate convention of the stored data.
            ``"yup"`` (default) — data is Y-up; ``get_*`` calls auto-convert to Z-up.
            ``"zup"`` — data is already Z-up; no conversion applied.
    """

    def __init__(self, up_axis: str = "yup") -> None:
        if up_axis not in ("yup", "zup"):
            raise ValueError(f"up_axis must be 'yup' or 'zup', got {up_axis!r}")
        self.up_axis = up_axis
        self._R: torch.Tensor | None = _Y_UP_TO_Z_UP if up_axis == "yup" else None

        self._motion_smpl_poses: torch.Tensor | None = None  # (total_frames, 72)
        self._motion_smpl_joints: torch.Tensor | None = None  # (total_frames, 24, 3)
        self._motion_smpl_transl: torch.Tensor | None = None  # (total_frames, 3)
        self._motion_num_frames: np.ndarray | None = None  # (M,) int64
        self._motion_fps: np.ndarray | None = None  # (M,) float32
        self._motion_source_fps: np.ndarray | None = None  # (M,) float32
        self._length_starts: torch.Tensor | None = None  # (M,) int64, on same device as tensors

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_motions(self, files: Sequence[str | MotionData], target_fps: float | None = None) -> None:
        """Load motion data from pkl paths or pre-loaded MotionData objects.

        Args:
            files: sequence of pkl file paths OR already-loaded MotionData objects.
                   Mixing both types is allowed.
            target_fps: resample to this fps when loading from a path.
                        Ignored for pre-loaded MotionData (caller is responsible
                        for resampling before passing in).
        """
        if not files:
            raise ValueError("files list is empty")

        motions: list[MotionData] = []
        for entry in files:
            if isinstance(entry, MotionData):
                motions.append(entry)
            else:
                motions.append(load_motion_file(entry, target_fps))

        self._motion_smpl_poses = torch.cat([m.pose_aa for m in motions], dim=0)
        self._motion_smpl_joints = torch.cat([m.smpl_joints for m in motions], dim=0)
        self._motion_smpl_transl = torch.cat([m.transl for m in motions], dim=0)
        self._motion_num_frames = np.array([m.num_frames for m in motions], dtype=np.int64)
        self._motion_fps = np.array([m.fps for m in motions], dtype=np.float32)
        self._motion_source_fps = np.array([m.source_fps for m in motions], dtype=np.float32)

        # Replicate length_starts from motion_lib_base.py lines 1524-1528:
        #   lengths_shifted = lengths.roll(1); lengths_shifted[0] = 0
        #   self.length_starts = lengths_shifted.cumsum(0)
        num_frames = torch.from_numpy(self._motion_num_frames.copy())
        ls = torch.roll(num_frames, 1)
        ls[0] = 0
        self._length_starts = torch.cumsum(ls, dim=0)

    def to_device(self, device: str | torch.device) -> None:
        """Move all tensors (including _length_starts) to ``device`` in-place."""
        self._check_loaded()
        self._motion_smpl_poses = self._motion_smpl_poses.to(device)  # type: ignore[union-attr]
        self._motion_smpl_joints = self._motion_smpl_joints.to(device)  # type: ignore[union-attr]
        self._motion_smpl_transl = self._motion_smpl_transl.to(device)  # type: ignore[union-attr]
        self._length_starts = self._length_starts.to(device)  # type: ignore[union-attr]

    def _check_loaded(self) -> None:
        if self._motion_smpl_poses is None:
            raise RuntimeError("No motions loaded. Call load_motions() first.")

    # ------------------------------------------------------------------
    # Public flat-tensor accessors
    # ------------------------------------------------------------------

    @property
    def poses_flat(self) -> torch.Tensor:
        """(total_frames, 72) flat axis-angle tensor."""
        self._check_loaded()
        return self._motion_smpl_poses  # type: ignore[return-value]

    @property
    def joints_flat(self) -> torch.Tensor:
        """(total_frames, 24, 3) flat joint-position tensor."""
        self._check_loaded()
        return self._motion_smpl_joints  # type: ignore[return-value]

    @property
    def transl_flat(self) -> torch.Tensor:
        """(total_frames, 3) flat root-translation tensor."""
        self._check_loaded()
        return self._motion_smpl_transl  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def _to_zup(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate (..., 3) points from Y-up to Z-up. No-op when up_axis='zup'."""
        if self._R is None:
            return x
        return x @ self._R.T.to(x.device)

    # ------------------------------------------------------------------
    # Query API  (motion_ids / motion_steps are torch.Tensor)
    # ------------------------------------------------------------------

    def _idx(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """Compute flat tensor indices from (motion_id, local_step) pairs."""
        starts = self._length_starts.to(motion_ids.device)  # type: ignore[union-attr]
        return motion_steps + starts[motion_ids]

    def get_smpl_pose(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 72) axis-angle. Root joint (first 3) converted to Z-up if up_axis='yup'.

        Uses left-multiply C @ M_yup (not sandwich C @ M @ C^T) — correct for SMPL root
        where the rotation maps local body frame into world frame.
        """
        self._check_loaded()
        pose = self._motion_smpl_poses[self._idx(motion_ids, motion_steps)]  # type: ignore[index]
        if self._R is not None:
            pose = pose.clone()
            R = self._R.to(pose.device)
            M_yup = angle_axis_to_rotation_matrix(pose[:, :3])  # (B, 3, 3)
            M_zup = R @ M_yup  # (B, 3, 3) — left multiply only
            pose[:, :3] = rotation_matrix_to_angle_axis(M_zup)  # (B, 3)
        return pose

    def get_smpl_joints(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 24, 3) joint positions, converted to Z-up if up_axis='yup'."""
        self._check_loaded()
        return self._motion_smpl_joints[self._idx(motion_ids, motion_steps)]  # type: ignore[index]

    def get_smpl_transl(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 3) root translation, converted to Z-up if up_axis='yup'."""
        self._check_loaded()
        return self._to_zup(self._motion_smpl_transl[self._idx(motion_ids, motion_steps)])  # type: ignore[index]

    def get_smpl_global_position(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 3) root translation, converted to Z-up if up_axis='yup'."""
        self._check_loaded()
        return self.get_smpl_joints(motion_ids, motion_steps) + self.get_smpl_transl(
            motion_ids, motion_steps
        ).unsqueeze(1)

    def get_smpl_global_rotations(self, motion_ids: torch.Tensor, motion_steps: torch.Tensor) -> torch.Tensor:
        """(B, 24, 3, 3) world-space rotation matrices via SMPL FK, Z-up if needed."""
        self._check_loaded()
        pose = self.get_smpl_pose(motion_ids, motion_steps)  # (B, 72)
        B = pose.shape[0]

        local_rot = angle_axis_to_rotation_matrix(pose.reshape(B * 24, 3)).reshape(B, 24, 3, 3)

        global_rot = torch.empty_like(local_rot)
        for j, p in enumerate(_SMPL_PARENTS):
            if p < 0:
                global_rot[:, j] = local_rot[:, j]
            else:
                global_rot[:, j] = global_rot[:, p] @ local_rot[:, j]

        if self._R is not None:
            R = self._R.to(global_rot.device)
            global_rot = (R @ global_rot.reshape(B * 24, 3, 3)).reshape(B, 24, 3, 3)

        return global_rot

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_num_motions(self) -> int:
        self._check_loaded()
        return len(self._motion_num_frames)  # type: ignore[arg-type]

    def get_num_frames(self, motion_id: int) -> int:
        self._check_loaded()
        if motion_id < 0 or motion_id >= len(self._motion_num_frames):  # type: ignore[arg-type]
            raise IndexError(f"motion_id {motion_id} out of range [0, {len(self._motion_num_frames)})")  # type: ignore[arg-type]
        return int(self._motion_num_frames[motion_id])  # type: ignore[index]

    def get_motion_fps(self, motion_id: int) -> float:
        self._check_loaded()
        return float(self._motion_fps[motion_id])  # type: ignore[index]

    def get_motion_duration(self, motion_id: int) -> float:
        """Return duration of motion in seconds."""
        self._check_loaded()
        return (self.get_num_frames(motion_id) - 1) / self.get_motion_fps(motion_id)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_random(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (motion_id, step) pairs.

        Returns:
            (motion_ids, motion_steps) each shape (batch_size,) as torch.long tensors.
        """
        self._check_loaded()
        n = self.get_num_motions()
        motion_ids = torch.randint(0, n, (batch_size,))
        frame_counts = torch.from_numpy(self._motion_num_frames[motion_ids.numpy()])  # type: ignore[index]
        motion_steps = (torch.rand(batch_size) * frame_counts).long()
        return motion_ids, motion_steps
