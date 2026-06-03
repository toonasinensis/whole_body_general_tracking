from __future__ import annotations

import sys
import types
from pathlib import Path


def _bootstrap_motionlib_test_imports() -> None:
    motionlib_dir = Path(__file__).resolve().parents[2]
    if str(motionlib_dir) not in sys.path:
        sys.path.insert(0, str(motionlib_dir))
    smpl_motion_lib_dir = motionlib_dir / "smpl_motion_lib"
    if str(smpl_motion_lib_dir) not in sys.path:
        sys.path.insert(0, str(smpl_motion_lib_dir))
    smpl_math_utils_dir = motionlib_dir / "smpl_math_utils"
    if str(smpl_math_utils_dir) not in sys.path:
        sys.path.insert(0, str(smpl_math_utils_dir))

    import smpl_math_utils as smpl_math_utils_module

    if "whole_body_tracking" not in sys.modules:
        sys.modules["whole_body_tracking"] = types.ModuleType("whole_body_tracking")
    if "whole_body_tracking.utils" not in sys.modules:
        sys.modules["whole_body_tracking.utils"] = types.ModuleType("whole_body_tracking.utils")
    if "whole_body_tracking.utils.motionlib" not in sys.modules:
        sys.modules["whole_body_tracking.utils.motionlib"] = types.ModuleType("whole_body_tracking.utils.motionlib")

    whole_body_tracking_module = sys.modules["whole_body_tracking"]
    utils_module = sys.modules["whole_body_tracking.utils"]
    motionlib_module = sys.modules["whole_body_tracking.utils.motionlib"]

    whole_body_tracking_module.utils = utils_module
    utils_module.motionlib = motionlib_module

    smpl_math_pkg_name = "whole_body_tracking.utils.motionlib.smpl_math_utils"
    if smpl_math_pkg_name not in sys.modules:
        sys.modules[smpl_math_pkg_name] = types.ModuleType(smpl_math_pkg_name)
    smpl_math_pkg = sys.modules[smpl_math_pkg_name]
    smpl_math_pkg.smpl_math_utils = smpl_math_utils_module
    motionlib_module.smpl_math_utils = smpl_math_pkg
    sys.modules["whole_body_tracking.utils.motionlib.smpl_math_utils.smpl_math_utils"] = smpl_math_utils_module


_bootstrap_motionlib_test_imports()


def pytest_sessionstart(session) -> None:
    _bootstrap_motionlib_test_imports()
