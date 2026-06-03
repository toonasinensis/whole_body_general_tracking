from __future__ import annotations

import joblib
import numpy as np

from feat_motion_lib.config import DiscoveryOptions, LoadOptions, UnifiedLoadConfig
from feat_motion_lib.facade.orchestration import execute_unified_load


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


def test_execute_unified_load_robot_only(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    motion_dir.mkdir()
    _write_robot_npz(motion_dir / "sample_a.npz", body_count=3, num_frames=3)
    _write_robot_npz(motion_dir / "sample_b.npz", body_count=3, num_frames=2)

    load_config = UnifiedLoadConfig(
        discovery=DiscoveryOptions(motion_dir=motion_dir),
        load=LoadOptions(device="cpu"),
    )
    outcome = execute_unified_load(
        body_indexes=[0, 1],
        joint_names=None,
        motion_body_names=None,
        all_body_names=None,
        load_config=load_config,
    )

    assert outcome.report.mode == "robot_only"
    assert outcome.smpl_lib is None
    assert outcome.state.file_names == ["sample_a.npz", "sample_b.npz"]
    assert outcome.state.time_step_total == 5
    assert outcome.state.frame_list.tolist() == [3, 2]


def test_execute_unified_load_paired(tmp_path) -> None:
    motion_dir = tmp_path / "robot"
    smpl_dir = tmp_path / "smpl"
    motion_dir.mkdir()
    smpl_dir.mkdir()
    _write_robot_npz(motion_dir / "sample.npz", body_count=3, num_frames=3)
    _write_smpl_pkl(smpl_dir / "sample.pkl", num_frames=4)

    load_config = UnifiedLoadConfig(
        discovery=DiscoveryOptions(motion_dir=motion_dir),
        load=LoadOptions(device="cpu"),
        smpl_dir=smpl_dir,
    )
    outcome = execute_unified_load(
        body_indexes=[0, 1],
        joint_names=None,
        motion_body_names=None,
        all_body_names=None,
        load_config=load_config,
    )

    assert outcome.report.mode == "paired"
    assert outcome.report.loaded_pairs == 1
    assert outcome.smpl_lib is not None
    assert outcome.state.time_step_total == 3
