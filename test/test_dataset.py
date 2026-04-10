""" Scripts to test the motion dataset class.
it shall return the correct motion bins and meta information.
"""
from whole_body_tracking.utils.motion_dataset import Motion_Bins_Dataset

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_split import visualize_motion_in_mujoco, check_segment_dimensions
dataset_dir = "assets/motion_bins_50"


if __name__ == "__main__":
    dataset = Motion_Bins_Dataset(dataset_dir=dataset_dir)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        # reform sample into inputs expected by the check_segment_dimensions function
        data = {
            "fps": [sample["fps"]],
            "joint_pos": sample["motion"]["joint_pos"],
            "joint_vel": sample["motion"]["joint_vel"],
            "body_pos_w": sample["motion"]["body_pos_w"],
            "body_quat_w": sample["motion"]["body_quat_w"],
            "body_lin_vel_w": sample["motion"]["body_lin_vel_w"],
            "body_ang_vel_w": sample["motion"]["body_ang_vel_w"],
        }
        # length is of data["body_pos_w"]
        assert sample["length"] == data["body_pos_w"].shape[0]
        qpos, qvel, body_pos_w, body_quat_w = check_segment_dimensions(data)
        if idx%20 == 0:
            visualize_motion_in_mujoco(qpos, qvel, body_pos_w, body_quat_w)
