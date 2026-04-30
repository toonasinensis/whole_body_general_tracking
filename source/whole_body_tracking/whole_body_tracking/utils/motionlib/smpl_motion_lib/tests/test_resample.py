import numpy as np

import joblib
from smpl_motion_lib import resample_30hz_to_50hz_and_save


def test_resample_30_to_50_and_save(tmp_path):
    in_path = tmp_path / "in.pkl"
    out_path = tmp_path / "out_50hz.pkl"

    T = 61  # 2 seconds at 30Hz with inclusive endpoints
    pose_aa = np.zeros((T, 72), dtype=np.float32)
    joints = np.zeros((T, 24, 3), dtype=np.float32)
    transl = np.zeros((T, 3), dtype=np.float32)
    transl[:, 0] = np.linspace(0.0, 1.0, T, dtype=np.float32)

    joblib.dump(
        {
            "pose_aa": pose_aa,
            "smpl_joints": joints,
            "transl": transl,
            "fps": 30.0,
        },
        in_path,
    )

    resample_30hz_to_50hz_and_save(str(in_path), str(out_path))

    assert out_path.exists()
    out = joblib.load(out_path)
    assert float(out["fps"]) == 50.0

    # Duration remains 2s, so expected frame count is floor(2 * 50) + 1 = 101
    assert out["pose_aa"].shape == (101, 72)
    assert out["smpl_joints"].shape == (101, 24, 3)
    assert out["transl"].shape == (101, 3)
