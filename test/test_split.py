"""Tests for split_motion_npz: load motion bins, check dimensions, and optional visualization."""

import sys
import numpy as np
import time
import mujoco
import mujoco.viewer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MOTION_BINS_DIR = REPO_ROOT / "assets" / "roban_motions_bins_50" / "210531"


def load_motion_bins_dir(bins_dir: Path) -> list[Path]:
    """Load motion data: return list of NPZ paths in assets/motion_bins_50."""
    bins_dir = Path(bins_dir).expanduser().resolve()
    if not bins_dir.is_dir(): return []
    npz_paths = sorted(bins_dir.glob("*.npz"))
    return npz_paths


def load_single_motion(npz_path: Path) -> dict:
    """Load a single motion NPZ into a dict of numpy arrays (and fps)."""
    data = np.load(npz_path)
    out = {}
    for key in data.files:
        out[key] = np.asarray(data[key])
    return out


REQUIRED_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def check_segment_dimensions(data: dict, npz_path: Path | None = None):
    """ Check dimensions of a segment NPZ as produced by run_split.
    """
    prefix = f"NPZ {npz_path}: " if npz_path else ""

    for key in REQUIRED_KEYS:
        assert key in data, f"{prefix}missing key '{key}'"

    fps = data["fps"]
    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    body_pos_w = data["body_pos_w"]
    body_quat_w = data["body_quat_w"]
    body_lin_vel_w = data["body_lin_vel_w"]
    body_ang_vel_w = data["body_ang_vel_w"]
    one_sec_frames = int(round(fps[0] * 1.0))
    
    # joints and bodies num shall be the same
    assert joint_pos.shape[1] == joint_vel.shape[1]
    assert body_pos_w.shape[1] == body_quat_w.shape[1]
    assert body_pos_w.shape[1] == body_lin_vel_w.shape[1]
    assert body_pos_w.shape[1] == body_ang_vel_w.shape[1]
    # check frames num
    frame_num = body_pos_w.shape[0]
    assert frame_num == body_quat_w.shape[0] 
    assert frame_num == body_lin_vel_w.shape[0] 
    assert frame_num == body_ang_vel_w.shape[0]
    # joint pos and vel shall have one second of extra frames
    assert (frame_num+one_sec_frames) == joint_pos.shape[0]
    assert (frame_num+one_sec_frames) == joint_vel.shape[0]
    
    print(f"fps: {fps}")
    print(f"one_sec_frames: {one_sec_frames}")
    print(f"frame_num: {frame_num}")
    print("=========================================================")
    print(f"joint_pos: {joint_pos.shape}")
    print(f"joint_vel: {joint_vel.shape}")
    print(f"body_pos_w: {body_pos_w.shape}")
    print(f"body_quat_w: {body_quat_w.shape}")
    print(f"body_lin_vel_w: {body_lin_vel_w.shape}")
    print(f"body_ang_vel_w: {body_ang_vel_w.shape}")
    print("=========================================================")
    return joint_pos, joint_vel, body_pos_w, body_quat_w


def visualize_motion_in_mujoco(
    joint_pos: np.ndarray, 
    joint_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    xml_path: str = "source/whole_body_tracking/whole_body_tracking/assets/kuavo_s52/xml/biped_s52.xml",
    fps: float = 50.0,
    loop: bool = False,
    show_viewer_controls: bool = True,
) -> None:
    """Visualize the motion sequence in MuJoCo viewer.
    
    Args:
        joint_pos: [n_frames, n_qpos] joint positions (qpos)
        joint_vel: [n_frames, n_qvel] joint velocities (qvel) - optional
        xml_path: path to the MuJoCo XML model file
        fps: playback speed in frames per second
        loop: whether to loop the animation forever
        joint_names: if provided, only these named joints will be set
        show_viewer_controls: whether to show the GUI controls
    """

    root_pos  = np.asarray(body_pos_w[:, 0, :])
    root_quat = np.asarray(body_quat_w[:, 0, :])
    motion_len = root_pos.shape[0]
    joint_pos = np.asarray(joint_pos)[:motion_len, :]
    joint_vel = np.asarray(joint_vel)[:motion_len, :]
    qpos = np.concatenate([root_pos, root_quat, joint_pos], axis=1)

    # Load model
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Decide which DOFs to control
    qpos_indices = np.arange(model.nq-2)
    qvel_indices = np.arange(model.nv-2)

    # Prepare timing
    frame_time = 1.0 / fps
    paused = False

    def playback():
        nonlocal paused
        frame_idx = 0

        while True:
            if not paused:
                # Set state
                data.qpos[qpos_indices] = qpos[frame_idx]
                mujoco.mj_forward(model, data)
                frame_idx = (frame_idx + 1) % motion_len
                if not loop and frame_idx == 0:
                    break
            time.sleep(frame_time)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=show_viewer_controls,
        show_right_ui=show_viewer_controls,
    ) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

        viewer.cam.lookat[:] = [0, 0, 0.8]  # you may want to adjust this
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        viewer.sync()

        print("Viewer controls:")
        print("  Space: pause/resume")
        print("  Esc / Q: quit")
        print("  Double-click: re-center camera")

        last_time = time.time()
        frame_idx = 0

        while viewer.is_running():
            now = time.time()
            dt = now - last_time
            if dt >= frame_time:
                # Advance one frame
                data.qpos[qpos_indices] = qpos[frame_idx % motion_len]
                mujoco.mj_forward(model, data)
                frame_idx = (frame_idx + 1) % motion_len
                last_time = now
            viewer.sync()

    print("Visualization finished.")


if __name__ == "__main__":
    npz_paths = load_motion_bins_dir(MOTION_BINS_DIR)
    for idx, npz_path in enumerate(npz_paths):
        data = load_single_motion(npz_path)
        qpos, qvel, body_pos_w, body_quat_w = check_segment_dimensions(data, npz_path)
        if idx%20 == 0:
            visualize_motion_in_mujoco(qpos, qvel, body_pos_w, body_quat_w)
