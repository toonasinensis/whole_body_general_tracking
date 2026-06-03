"""VISER-based 3D visualization utilities for SMPL motions."""

from __future__ import annotations

import numpy as np
import time
from pathlib import Path

import viser

from .motion_lib import SmplMotionLib

# SMPL 24-joint kinematic tree (parent index for each joint)
SMPL_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)


def _build_bone_segments(frame_joints: np.ndarray) -> np.ndarray:
    """Create line segments with shape (num_bones, 2, 3) for one frame."""
    segments = []
    for child, parent in enumerate(SMPL_PARENTS):
        if parent < 0:
            continue
        segments.append(np.stack([frame_joints[parent], frame_joints[child]], axis=0))
    return np.stack(segments, axis=0).astype(np.float32)


def _build_ground_grid(
    center_xy: np.ndarray,
    ground_z: float,
    half_size: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a square ground grid as line segments and RGB colors."""
    lines = []
    x_vals = np.arange(-half_size, half_size + 1e-6, step, dtype=np.float32)
    y_vals = np.arange(-half_size, half_size + 1e-6, step, dtype=np.float32)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    z = float(ground_z)

    for x in x_vals:
        lines.append([[cx + x, cy - half_size, z], [cx + x, cy + half_size, z]])
    for y in y_vals:
        lines.append([[cx - half_size, cy + y, z], [cx + half_size, cy + y, z]])

    points = np.asarray(lines, dtype=np.float32)
    colors = np.tile(np.array([[[0.35, 0.35, 0.35], [0.35, 0.35, 0.35]]], dtype=np.float32), (points.shape[0], 1, 1))
    return points, colors


def _build_joint_frame_segments(
    frame_joints: np.ndarray,
    frame_rotations: np.ndarray,
    axis_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create RGB axis segments for all joints in one frame."""
    axis_colors = np.asarray(
        [
            [[1.0, 0.15, 0.15], [1.0, 0.15, 0.15]],
            [[0.15, 1.0, 0.15], [0.15, 1.0, 0.15]],
            [[0.15, 0.4, 1.0], [0.15, 0.4, 1.0]],
        ],
        dtype=np.float32,
    )

    points = []
    colors = []
    for joint_idx in range(frame_joints.shape[0]):
        origin = frame_joints[joint_idx]
        rotation = frame_rotations[joint_idx]
        for axis_idx in range(3):
            direction = rotation[:, axis_idx] * axis_length
            points.append(np.stack([origin, origin + direction], axis=0))
            colors.append(axis_colors[axis_idx])

    return np.asarray(points, dtype=np.float32), np.asarray(colors, dtype=np.float32)


def visualize_motion(
    motion_lib: SmplMotionLib,
    motion_id: int = 0,
    use_transl: bool = False,
    show_ground: bool = True,
    show_joint_frames: bool = False,
    frame_axis_length: float = 0.08,
    host: str = "127.0.0.1",
    port: int = 8080,
    loop: bool = True,
    blocking: bool = True,
):
    """Visualize one motion sequence as a 3D animated skeleton in VISER.

    Coordinate conversion (Y-up → Z-up) is handled automatically by
    ``motion_lib`` according to its ``up_axis`` setting.

    Args:
        motion_lib: loaded SmplMotionLib (set ``up_axis`` at construction time)
        motion_id: motion index to visualize
        use_transl: if True, adds root translation to all joint positions
        show_ground: if True, render a ground grid below the motion
        show_joint_frames: if True, render RGB local axes at each joint
        frame_axis_length: length of each axis line in meters
        host: VISER server host
        port: VISER server port
        loop: whether to loop playback
        blocking: if True, block current thread and stream frames continuously

    Returns:
        viser.ViserServer object
    """
    num_frames = motion_lib.get_num_frames(motion_id)
    fps = motion_lib.get_motion_fps(motion_id)

    motion_ids = np.full((num_frames,), motion_id, dtype=np.int64)
    steps = np.arange(num_frames, dtype=np.int64)

    joints = motion_lib.get_smpl_joints(motion_ids, steps).cpu().numpy()  # (T, 24, 3), Z-up
    if use_transl:
        transl = motion_lib.get_smpl_transl(motion_ids, steps).cpu().numpy()  # (T, 3), Z-up
        joints = joints + transl[:, None, :]

    joint_rotations = None
    if show_joint_frames:
        joint_rotations = motion_lib.get_smpl_global_rotations(motion_ids, steps).cpu().numpy()

    server = viser.ViserServer(host=host, port=port)

    if show_ground:
        xy_min = joints[..., :2].min(axis=(0, 1))
        xy_max = joints[..., :2].max(axis=(0, 1))
        center_xy = 0.5 * (xy_min + xy_max)
        half_size = float(max(np.max(xy_max - xy_min) * 0.7, 1.5))
        step = max(half_size / 10.0, 0.15)
        ground_z = float(np.min(joints[..., 2]))
        ground_points, ground_colors = _build_ground_grid(center_xy, ground_z, half_size, step)
        server.scene.add_line_segments(
            "/smpl/ground",
            points=ground_points,
            colors=ground_colors,
            line_width=1.0,
        )

    colors = np.tile(np.array([[0.20, 0.90, 0.85]], dtype=np.float32), (joints.shape[1], 1))
    joint_handle = server.scene.add_point_cloud(
        "/smpl/joints",
        points=joints[0],
        colors=colors,
        point_size=0.03,
    )

    bone_colors = np.tile(np.array([[[0.95, 0.65, 0.15], [0.95, 0.65, 0.15]]], dtype=np.float32), (23, 1, 1))
    bone_handle = server.scene.add_line_segments(
        "/smpl/bones",
        points=_build_bone_segments(joints[0]),
        colors=bone_colors,
        line_width=2.0,
    )

    frame_handle = None
    if show_joint_frames and joint_rotations is not None:
        frame_points, frame_colors = _build_joint_frame_segments(joints[0], joint_rotations[0], frame_axis_length)
        frame_handle = server.scene.add_line_segments(
            "/smpl/joint_frames",
            points=frame_points,
            colors=frame_colors,
            line_width=1.5,
        )

    idx = 0

    def step_once() -> None:
        nonlocal idx
        joint_handle.points = joints[idx]
        bone_handle.points = _build_bone_segments(joints[idx])
        if frame_handle is not None and joint_rotations is not None:
            frame_handle.points = _build_joint_frame_segments(joints[idx], joint_rotations[idx], frame_axis_length)[0]
        idx += 1
        if idx >= num_frames:
            idx = 0 if loop else num_frames - 1

    if not blocking:
        step_once()
        return server

    frame_dt = 1.0 / max(fps, 1e-6)
    while True:
        step_once()
        time.sleep(frame_dt)


def visualize_motion_batch(
    motion_lib: SmplMotionLib,
    motion_ids: list[int] | None = None,
    use_transl: bool = False,
    show_ground: bool = True,
    show_joint_frames: bool = False,
    frame_axis_length: float = 0.08,
    host: str = "127.0.0.1",
    port: int = 8080,
    loop: bool = True,
    blocking: bool = True,
):
    """Visualize multiple motions sequentially in VISER.

    Each motion is played once, then the next motion starts. If ``loop=True``,
    playback cycles over the selected motion list.
    """
    total_motions = motion_lib.get_num_motions()
    selected_ids = list(range(total_motions)) if motion_ids is None else motion_ids
    if not selected_ids:
        raise ValueError("motion_ids is empty")

    joints_per_motion: list[np.ndarray] = []
    rotations_per_motion: list[np.ndarray] = []
    fps_per_motion: list[float] = []

    for motion_id in selected_ids:
        num_frames = motion_lib.get_num_frames(motion_id)
        fps = motion_lib.get_motion_fps(motion_id)
        ids = np.full((num_frames,), motion_id, dtype=np.int64)
        steps = np.arange(num_frames, dtype=np.int64)

        joints = motion_lib.get_smpl_joints(ids, steps).cpu().numpy()  # Z-up
        if use_transl:
            transl = motion_lib.get_smpl_transl(ids, steps).cpu().numpy()  # Z-up
            joints = joints + transl[:, None, :]
        joints_per_motion.append(joints)

        if show_joint_frames:
            rotations_per_motion.append(motion_lib.get_smpl_global_rotations(ids, steps).cpu().numpy())
        fps_per_motion.append(float(fps))

    all_joints = np.concatenate(joints_per_motion, axis=0)

    server = viser.ViserServer(host=host, port=port)

    if show_ground:
        xy_min = all_joints[..., :2].min(axis=(0, 1))
        xy_max = all_joints[..., :2].max(axis=(0, 1))
        center_xy = 0.5 * (xy_min + xy_max)
        half_size = float(max(np.max(xy_max - xy_min) * 0.7, 1.5))
        step = max(half_size / 10.0, 0.15)
        ground_z = float(np.min(all_joints[..., 2]))
        ground_points, ground_colors = _build_ground_grid(center_xy, ground_z, half_size, step)
        server.scene.add_line_segments(
            "/smpl/ground",
            points=ground_points,
            colors=ground_colors,
            line_width=1.0,
        )

    colors = np.tile(np.array([[0.20, 0.90, 0.85]], dtype=np.float32), (all_joints.shape[1], 1))
    joint_handle = server.scene.add_point_cloud(
        "/smpl/joints",
        points=joints_per_motion[0][0],
        colors=colors,
        point_size=0.03,
    )
    bone_colors = np.tile(np.array([[[0.95, 0.65, 0.15], [0.95, 0.65, 0.15]]], dtype=np.float32), (23, 1, 1))
    bone_handle = server.scene.add_line_segments(
        "/smpl/bones",
        points=_build_bone_segments(joints_per_motion[0][0]),
        colors=bone_colors,
        line_width=2.0,
    )

    frame_handle = None
    if show_joint_frames and rotations_per_motion:
        frame_points, frame_colors = _build_joint_frame_segments(
            joints_per_motion[0][0],
            rotations_per_motion[0][0],
            frame_axis_length,
        )
        frame_handle = server.scene.add_line_segments(
            "/smpl/joint_frames",
            points=frame_points,
            colors=frame_colors,
            line_width=1.5,
        )

    motion_ptr = 0
    frame_idx = 0

    def step_once() -> float:
        nonlocal motion_ptr, frame_idx
        joints = joints_per_motion[motion_ptr]
        frame = joints[frame_idx]
        joint_handle.points = frame
        bone_handle.points = _build_bone_segments(frame)
        if frame_handle is not None and rotations_per_motion:
            frame_handle.points = _build_joint_frame_segments(
                frame,
                rotations_per_motion[motion_ptr][frame_idx],
                frame_axis_length,
            )[0]

        frame_idx += 1
        if frame_idx >= joints.shape[0]:
            frame_idx = 0
            motion_ptr += 1
            if motion_ptr >= len(joints_per_motion):
                motion_ptr = 0 if loop else len(joints_per_motion) - 1

        return 1.0 / max(fps_per_motion[motion_ptr], 1e-6)

    if not blocking:
        step_once()
        return server

    while True:
        frame_dt = step_once()
        time.sleep(frame_dt)


def visualize_motion_file(
    pkl_path: str,
    target_fps: float | None = None,
    up_axis: str = "yup",
    use_transl: bool = False,
    show_ground: bool = True,
    show_joint_frames: bool = False,
    frame_axis_length: float = 0.08,
    host: str = "127.0.0.1",
    port: int = 8080,
    loop: bool = True,
    blocking: bool = True,
):
    """Load one motion pkl and visualize it as 3D animated skeleton in VISER."""
    lib = SmplMotionLib(up_axis=up_axis)
    lib.load_motions([pkl_path], target_fps=target_fps)
    return visualize_motion(
        lib,
        motion_id=0,
        use_transl=use_transl,
        show_ground=show_ground,
        show_joint_frames=show_joint_frames,
        frame_axis_length=frame_axis_length,
        host=host,
        port=port,
        loop=loop,
        blocking=blocking,
    )


def visualize_motion_directory(
    dir_path: str,
    target_fps: float | None = None,
    up_axis: str = "yup",
    use_transl: bool = False,
    show_ground: bool = True,
    show_joint_frames: bool = False,
    frame_axis_length: float = 0.08,
    host: str = "127.0.0.1",
    port: int = 8080,
    loop: bool = True,
    blocking: bool = True,
):
    """Load all .pkl motions from a directory and visualize them sequentially."""
    files = sorted(str(p) for p in Path(dir_path).glob("*.pkl"))
    if not files:
        raise ValueError(f"No .pkl files found in directory: {dir_path}")

    lib = SmplMotionLib(up_axis=up_axis)
    lib.load_motions(files, target_fps=target_fps)
    return visualize_motion_batch(
        lib,
        motion_ids=list(range(lib.get_num_motions())),
        use_transl=use_transl,
        show_ground=show_ground,
        show_joint_frames=show_joint_frames,
        frame_axis_length=frame_axis_length,
        host=host,
        port=port,
        loop=loop,
        blocking=blocking,
    )
