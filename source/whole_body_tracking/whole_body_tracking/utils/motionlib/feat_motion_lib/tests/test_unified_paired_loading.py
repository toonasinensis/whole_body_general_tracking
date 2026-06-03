from __future__ import annotations

from types import SimpleNamespace

import joblib
import numpy as np

from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib


def _cfg(motion_file: str, smpl_file_path: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        motion_file=motion_file,
        max_motion_num=-1,
        dataset_txt=None,
        eval_mode=False,
        distributed=False,
        local_rank=0,
        total_rank=1,
        smpl_file_path=smpl_file_path,
        target_fps=50.0,
    )


def _write_robot_npz(path, *, body_count: int, num_frames: int = 3) -> None:
    np.savez(
        path,
        fps=np.asarray([50.0], dtype=np.float32),
        joint_pos=np.zeros((num_frames, 2), dtype=np.float32),
        joint_vel=np.zeros((num_frames, 2), dtype=np.float32),
        body_pos_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
        body_quat_w=np.zeros((num_frames, body_count, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((num_frames, body_count, 3), dtype=np.float32),
    )


def _write_smpl_pkl(path, *, num_frames: int = 3) -> None:
    joblib.dump(
        {
            "pose_aa": np.zeros((num_frames, 72), dtype=np.float32),
            "smpl_joints": np.zeros((num_frames, 24, 3), dtype=np.float32),
            "transl": np.zeros((num_frames, 3), dtype=np.float32),
            "fps": 50.0,
        },
        path,
    )


def test_unified_motion_lib_loads_paired_and_reports_mode(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    motion_dir.mkdir()
    smpl_dir.mkdir()
    _write_robot_npz(motion_dir / "sample.npz", body_count=3, num_frames=3)
    _write_smpl_pkl(smpl_dir / "sample.pkl", num_frames=4)

    lib = UnifiedMotionLib(
        body_indexes=[0, 1],
        motion_anchor_body_index=0,
        joint_names=None,
        motion_body_names=None,
        all_body_names=None,
        device="cpu",
    )
    report = lib.load_from_cfg(_cfg(str(motion_dir), str(smpl_dir)))

    assert report.mode == "paired"
    assert report.requested_pairs == 1
    assert report.loaded_pairs == 1
    assert lib.motion_num == 1
    assert int(lib.time_step_total) == 3
    assert lib.smpl_poses is not None
    assert lib.smpl_poses.shape[0] == 3


def test_unified_motion_lib_falls_back_when_no_matching_pairs(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    motion_dir.mkdir()
    smpl_dir.mkdir()
    _write_robot_npz(motion_dir / "sample.npz", body_count=3, num_frames=3)
    _write_smpl_pkl(smpl_dir / "other.pkl", num_frames=3)

    lib = UnifiedMotionLib(
        body_indexes=[0, 1],
        motion_anchor_body_index=0,
        device="cpu",
    )
    report = lib.load_from_cfg(_cfg(str(motion_dir), str(smpl_dir)))

    assert report.mode == "paired_fallback"
    assert report.fallback_used is True
    assert lib.smpl_poses is None
