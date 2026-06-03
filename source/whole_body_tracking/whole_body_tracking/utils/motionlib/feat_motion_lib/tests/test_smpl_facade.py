from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from feat_motion_lib.config import LoadOptions, SmplLoadConfig
from feat_motion_lib.errors import MotionNotLoadedError
from feat_motion_lib.facade.smpl_motion_lib import SmplMotionLib


def _write_smpl_pkl(path, *, num_frames: int = 3, fps: float = 50.0) -> None:
    joblib.dump(
        {
            "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
            "transl": np.zeros((num_frames, 3), dtype=np.float32),
            "fps": fps,
        },
        path,
    )


def test_smpl_motion_lib_load_accepts_structured_config(tmp_path) -> None:
    path_a = tmp_path / "a.pkl"
    path_b = tmp_path / "b.pkl"
    _write_smpl_pkl(path_a, num_frames=3)
    _write_smpl_pkl(path_b, num_frames=2)

    lib = SmplMotionLib(device="cpu")
    report = lib.load(
        SmplLoadConfig(
            motion_files=[path_a, path_b],
            load=LoadOptions(target_fps=50.0, device="cpu", up_axis="yup"),
        )
    )

    assert report.mode == "smpl_only"
    assert lib.get_num_motions() == 2
    assert lib.poses_flat.shape[0] == 5
    assert lib.last_report is report


def test_smpl_motion_lib_load_from_cfg_supports_directory(tmp_path) -> None:
    _write_smpl_pkl(tmp_path / "a.pkl", num_frames=3)
    _write_smpl_pkl(tmp_path / "b.pkl", num_frames=2)
    cfg = SimpleNamespace(motion_file=str(tmp_path), target_fps=50.0, up_axis="yup")

    lib = SmplMotionLib(device="cpu")
    report = lib.load_from_cfg(cfg)

    assert report.mode == "smpl_only"
    assert lib.get_num_motions() == 2


def test_smpl_motion_lib_reset_restores_unloaded_state(tmp_path) -> None:
    path = tmp_path / "sample.pkl"
    _write_smpl_pkl(path, num_frames=3)
    lib = SmplMotionLib(device="cpu")
    lib.load(SmplLoadConfig(motion_files=[path], load=LoadOptions(device="cpu")))
    lib.reset()

    with pytest.raises(MotionNotLoadedError):
        _ = lib.poses_flat
    assert lib.last_report is None
