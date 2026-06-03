import glob
import os

import pytest
from smpl_motion_lib import SmplMotionLib

TEST_PKL_DIR = "/home/thl/Downloads/data/TEST"


@pytest.fixture(scope="session")
def test_pkl_files():
    files = sorted(glob.glob(os.path.join(TEST_PKL_DIR, "*.pkl")))
    if not files:
        pytest.skip(f"No pkl files found in {TEST_PKL_DIR}")
    return files


@pytest.fixture(scope="session")
def loaded_lib(test_pkl_files):
    lib = SmplMotionLib()
    lib.load_motions(test_pkl_files, target_fps=30.0)
    return lib


@pytest.fixture(scope="session")
def loaded_lib_native(test_pkl_files):
    """Library loaded without resampling (native FPS)."""
    lib = SmplMotionLib()
    lib.load_motions(test_pkl_files)
    return lib
