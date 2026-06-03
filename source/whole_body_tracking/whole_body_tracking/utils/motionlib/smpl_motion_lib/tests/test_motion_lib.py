import numpy as np
import torch

import joblib
import pytest
from smpl_motion_lib import SmplMotionLib

TEST_PKL_DIR = "/home/thl/Downloads/data/TEST"


# ── Loading ────────────────────────────────────────────────────────────────────


def test_load_three_files(loaded_lib):
    assert loaded_lib.get_num_motions() == 3


def test_num_frames_positive(loaded_lib):
    for i in range(loaded_lib.get_num_motions()):
        assert loaded_lib.get_num_frames(i) > 0


def test_duration_positive(loaded_lib):
    for i in range(loaded_lib.get_num_motions()):
        assert loaded_lib.get_motion_duration(i) > 0.0


def test_fps_after_resampling(loaded_lib):
    for i in range(loaded_lib.get_num_motions()):
        assert loaded_lib.get_motion_fps(i) == pytest.approx(30.0)


def test_native_fps_preserved(loaded_lib_native, test_pkl_files):
    for i, path in enumerate(test_pkl_files):
        raw = joblib.load(path)
        expected_fps = float(raw.get("fps", 30))
        assert loaded_lib_native.get_motion_fps(i) == pytest.approx(expected_fps)


# ── Array shapes ──────────────────────────────────────────────────────────────


def test_get_smpl_pose_shape(loaded_lib):
    ids = np.array([0, 0, 1])
    steps = np.array([0, 1, 0])
    result = loaded_lib.get_smpl_pose(ids, steps)
    assert result.shape == (3, 72)
    assert result.dtype == torch.float32


def test_get_smpl_joints_shape(loaded_lib):
    ids = np.array([0])
    steps = np.array([0])
    result = loaded_lib.get_smpl_joints(ids, steps)
    assert result.shape == (1, 24, 3)
    assert result.dtype == torch.float32


def test_get_smpl_transl_shape(loaded_lib):
    ids = np.array([0])
    steps = np.array([0])
    result = loaded_lib.get_smpl_transl(ids, steps)
    assert result.shape == (1, 3)
    assert result.dtype == torch.float32


# ── Indexing correctness ──────────────────────────────────────────────────────


def test_length_starts_first_is_zero(loaded_lib):
    assert loaded_lib._length_starts[0] == 0


def test_length_starts_increasing(loaded_lib):
    ls = loaded_lib._length_starts
    assert np.all(np.diff(ls) > 0)


def test_indexing_first_frame_matches_raw(test_pkl_files, loaded_lib_native):
    """Frame 0 of motion 0 should match the raw pkl data exactly."""
    raw = joblib.load(test_pkl_files[0])
    expected_pose = torch.from_numpy(np.asarray(raw["pose_aa"][0], dtype=np.float32))
    got = loaded_lib_native.get_smpl_pose(np.array([0]), np.array([0]))[0]
    assert torch.allclose(got, expected_pose, atol=1e-6)


def test_indexing_last_frame_no_oob(loaded_lib):
    for i in range(loaded_lib.get_num_motions()):
        last = loaded_lib.get_num_frames(i) - 1
        ids = np.array([i])
        steps = np.array([last])
        pose = loaded_lib.get_smpl_pose(ids, steps)
        assert pose.shape == (1, 72)


# ── sample_random ─────────────────────────────────────────────────────────────


def test_sample_random_shape(loaded_lib):
    ids, steps = loaded_lib.sample_random(16)
    assert ids.shape == (16,)
    assert steps.shape == (16,)


def test_sample_random_valid_ids(loaded_lib):
    ids, steps = loaded_lib.sample_random(100)
    assert np.all(ids >= 0) and np.all(ids < loaded_lib.get_num_motions())


def test_sample_random_valid_steps(loaded_lib):
    ids, steps = loaded_lib.sample_random(100)
    for mid, step in zip(ids, steps):
        assert 0 <= step < loaded_lib.get_num_frames(int(mid))


def test_sample_random_retrieval(loaded_lib):
    ids, steps = loaded_lib.sample_random(32)
    pose = loaded_lib.get_smpl_pose(ids, steps)
    joints = loaded_lib.get_smpl_joints(ids, steps)
    transl = loaded_lib.get_smpl_transl(ids, steps)
    assert pose.shape == (32, 72)
    assert joints.shape == (32, 24, 3)
    assert transl.shape == (32, 3)


# ── Error handling ─────────────────────────────────────────────────────────────


def test_motion_id_out_of_range(loaded_lib):
    with pytest.raises(IndexError):
        loaded_lib.get_num_frames(999)


def test_no_motions_loaded():
    lib = SmplMotionLib()
    with pytest.raises(RuntimeError):
        lib.get_smpl_pose(np.array([0]), np.array([0]))


def test_empty_file_list():
    lib = SmplMotionLib()
    with pytest.raises(ValueError):
        lib.load_motions([])
