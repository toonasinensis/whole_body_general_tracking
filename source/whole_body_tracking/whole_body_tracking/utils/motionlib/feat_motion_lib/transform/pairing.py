from __future__ import annotations

from pathlib import Path

from ..types import MotionData, PairedMotionData, PairedPath, RobotMotionData


def pair_motion_paths(robot_paths: list[str | Path], smpl_dir: str | Path) -> list[PairedPath]:
    smpl_dir = Path(smpl_dir)
    robot_map = {Path(path).stem: Path(path) for path in robot_paths}
    smpl_map = {path.stem: path for path in smpl_dir.glob("*.pkl")}
    common = sorted(set(robot_map) & set(smpl_map))
    return [
        PairedPath(
            stem=stem,
            robot_path=robot_map[stem],
            smpl_path=smpl_map[stem],
        )
        for stem in common
    ]


def trim_paired_clips(
    stem: str,
    robot: RobotMotionData,
    smpl: MotionData,
    max_frame_diff: int = 2,
) -> PairedMotionData | None:
    diff = abs(robot.num_frames - smpl.num_frames)
    if diff > max_frame_diff:
        return None

    frame_count = min(robot.num_frames, smpl.num_frames)
    trimmed_robot = RobotMotionData(
        joint_pos=robot.joint_pos[:frame_count],
        joint_vel=robot.joint_vel[:frame_count],
        body_pos_w=robot.body_pos_w[:frame_count],
        body_quat_w=robot.body_quat_w[:frame_count],
        body_lin_vel_w=robot.body_lin_vel_w[:frame_count],
        body_ang_vel_w=robot.body_ang_vel_w[:frame_count],
        fps=robot.fps,
        source_fps=robot.source_fps,
        num_frames=frame_count,
        duration=(frame_count - 1) / robot.fps if frame_count > 1 else 0.0,
        joint_names=robot.joint_names,
        body_names=robot.body_names,
        path=robot.path,
    )
    trimmed_smpl = MotionData(
        pose_aa=smpl.pose_aa[:frame_count],
        smpl_joints=smpl.smpl_joints[:frame_count],
        transl=smpl.transl[:frame_count],
        fps=smpl.fps,
        source_fps=smpl.source_fps,
        num_frames=frame_count,
        duration=(frame_count - 1) / smpl.fps if frame_count > 1 else 0.0,
    )
    return PairedMotionData(stem=stem, robot=trimmed_robot, smpl=trimmed_smpl)
