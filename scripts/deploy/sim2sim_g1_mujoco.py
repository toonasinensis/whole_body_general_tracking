from __future__ import annotations

import argparse
import csv
import json
import numpy as np

# import time
from datetime import datetime
from pathlib import Path

from sim2sim_g1.math_utils import as_vector
from sim2sim_g1.metrics import METRIC_NAMES, MotionMetricAccumulator, motion_tracking_metrics
from sim2sim_g1.motion import MotionData, motion_files, motion_frame_root_state, motion_local_step_summary
from sim2sim_g1.mujoco_robot import G1_MJCF  # noqa: F401
from sim2sim_g1.mujoco_robot import (  # print_joint_map,; name_to_joint_ids,
    action_to_target,
    apply_pd_control,
    gains_from_metadata,
    initialize_default_pose,
    initialize_from_motion,
    name_to_actuator_ids,
    name_to_body_ids,
    name_to_joint_qvel_addrs,
)
from sim2sim_g1.observations import (  # print_obs_layout,
    ImuReader,
    TermMajorHistory,
    build_obs,
    print_imu_debug,
    prop_terms_from_metadata,
    validate_inputs,
)
from sim2sim_g1.onnx_policy import OnnxPolicy, load_metadata, onnx_input_names, validate_grouped_onnx_contract
from sim2sim_g1.terrain import MujocoHeightScanner, mujoco_xml_with_terrain_mesh, terrain_path_from_motion
from sim2sim_g1.viewer import ReferenceMotionPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 ONNX policy in MuJoCo.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--dataset_txt", default=None)
    parser.add_argument(
        "--motion_index",
        type=int,
        default=None,
        help="Run only one motion by index after applying --dataset_txt filtering.",
    )
    parser.add_argument(
        "--motion_path",
        default=None,
        help="Run only the motion whose path or basename matches this value after applying --dataset_txt filtering.",
    )
    parser.add_argument("--xml_path", default=str(G1_MJCF))
    parser.add_argument("--steps", type=int, default=20000000)
    parser.add_argument(
        "--motion_start_frame",
        type=int,
        default=5,
        help="Reference frame used to initialize each motion rollout. Default skips the first 5 frames.",
    )
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--kp", type=float, default=None, help="Override metadata joint stiffness with a scalar value.")
    parser.add_argument("--kd", type=float, default=None, help="Override metadata joint damping with a scalar value.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no_render", action="store_true", help="Disable MuJoCo viewer when wrapper enables --render.")
    parser.add_argument(
        "--camera_follow",
        action="store_true",
        default=True,
        help="Keep the MuJoCo viewer camera looking at the robot.",
    )
    parser.add_argument("--no_camera_follow", action="store_false", dest="camera_follow")
    parser.add_argument("--camera_body", default="pelvis", help="Body name used as the camera look-at target.")
    parser.add_argument("--camera_distance", type=float, default=3.0, help="MuJoCo viewer camera distance.")
    parser.add_argument("--camera_azimuth", type=float, default=135.0, help="MuJoCo viewer camera azimuth in degrees.")
    parser.add_argument(
        "--camera_elevation", type=float, default=-18.0, help="MuJoCo viewer camera elevation in degrees."
    )
    parser.add_argument(
        "--camera_lookat_offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.45),
        help="World-frame offset added to the camera look-at body position.",
    )
    parser.add_argument(
        "--hold_final_frame",
        action="store_true",
        default=True,
        help="When rendering, keep simulating after rollout completes while holding the final reference frame.",
    )
    parser.add_argument("--no_hold_final_frame", action="store_false", dest="hold_final_frame")
    parser.add_argument(
        "--hold_final_policy_mode",
        choices=["run_policy", "hold_target"],
        default="run_policy",
        help=(
            "After rollout completes, either keep running the policy with the final motion command "
            "or hold the final PD target."
        ),
    )
    parser.add_argument("--imu_quat_sensor", default="base_quat", help="MuJoCo framequat sensor name.")
    parser.add_argument("--imu_gyro_sensor", default="base_gyro", help="MuJoCo gyro sensor name.")
    parser.add_argument("--debug_imu", action="store_true", help="Print XML sensor IMU values once at startup.")
    parser.add_argument(
        "--debug_motion_alignment",
        action="store_true",
        help="Print motion/root/obs/action alignment values once at startup.",
    )
    parser.add_argument(
        "--velocity_command_source",
        choices=["motion", "keyboard", "zero"],
        default="motion",
        help="Source for ONNX input 'velcommand' when the policy exposes it.",
    )
    parser.add_argument("--keyboard_vx_step", type=float, default=0.1, help="Keyboard vx increment for numpad 8/5.")
    parser.add_argument("--keyboard_vy_step", type=float, default=0.1, help="Keyboard vy increment for numpad 4/6.")
    parser.add_argument(
        "--keyboard_yaw_step", type=float, default=0.1, help="Keyboard yaw-rate increment for numpad 7/9."
    )
    parser.add_argument("--keyboard_vx_limit", type=float, default=1.5, help="Absolute vx command limit.")
    parser.add_argument("--keyboard_vy_limit", type=float, default=1.0, help="Absolute vy command limit.")
    parser.add_argument("--keyboard_yaw_limit", type=float, default=1.5, help="Absolute yaw-rate command limit.")
    parser.add_argument(
        "--show_velocity_command",
        action="store_true",
        default=True,
        help="Render the current velocity command as an arrow in the MuJoCo viewer.",
    )
    parser.add_argument("--no_show_velocity_command", action="store_false", dest="show_velocity_command")
    parser.add_argument("--velocity_command_arrow_scale", type=float, default=0.45)
    parser.add_argument("--velocity_command_arrow_z", type=float, default=0.08)
    parser.add_argument("--debug_height_scan", action="store_true", help="Print height_scan ray/obs stats once.")
    parser.add_argument(
        "--init_from_motion",
        action="store_true",
        default=True,
        help="Initialize MuJoCo root, joints, and velocities from --motion_start_frame.",
    )
    parser.add_argument("--no_init_from_motion", action="store_false", dest="init_from_motion")
    parser.add_argument(
        "--init_root_body",
        default=None,
        help="Motion body used to initialize the floating base. Defaults to metadata root body or first motion body.",
    )
    parser.add_argument(
        "--spawn_height_offset",
        type=float,
        default=0.05,
        help="Additional z offset in meters applied to the floating base after initialization.",
    )
    parser.add_argument(
        "--show_reference",
        action="store_true",
        default=True,
        help="Render a translucent G1 that follows the reference motion in the MuJoCo viewer.",
    )
    parser.add_argument("--no_show_reference", action="store_false", dest="show_reference")
    parser.add_argument("--reference_root_body", default=None, help="Motion body used as reference floating-base root.")
    parser.add_argument("--reference_alpha", type=float, default=0.35, help="Transparency for the reference G1.")
    parser.add_argument(
        "--reference_update_interval",
        type=int,
        default=1,
        help="Update translucent reference geoms every N policy steps. Increase this if the viewer is slow.",
    )
    parser.add_argument(
        "--print_joint_map",
        action="store_true",
        default=True,
        help="Print ONNX/action joint order mapped to MuJoCo joint ids and qpos/qvel addresses.",
    )
    parser.add_argument("--no_print_joint_map", action="store_false", dest="print_joint_map")
    parser.add_argument("--dry_run", action="store_true", help="Build and validate one observation, then exit.")
    parser.add_argument(
        "--terrain_stl",
        default=None,
        help="STL terrain used for height_scan. Defaults to the selected motion npz terrain_stl field when present.",
    )
    parser.add_argument(
        "--terrain_mode",
        choices=["auto", "flat", "mesh"],
        default="auto",
        help="MuJoCo terrain source. auto injects an STL when available, otherwise uses the flat floor.",
    )
    parser.add_argument(
        "--terrain_mesh_group",
        type=int,
        default=0,
        help="MuJoCo geom group assigned to injected terrain mesh; height_scan should include this group.",
    )
    parser.add_argument(
        "--terrain_collision_backend",
        choices=["boxes", "mesh"],
        default="boxes",
        help="Collision representation for STL terrain. boxes avoids MuJoCo mesh convex-hull collision.",
    )
    parser.add_argument("--height_scan_body", default="torso_link", help="MuJoCo body used as height scanner frame.")
    parser.add_argument(
        "--height_scan_offset",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.80, 0.0, 20.0),
        help="RayCaster offset in the scanner body frame, matching IsaacLab height_scanner.",
    )
    parser.add_argument(
        "--height_scan_ground_z", type=float, default=0.0, help="Flat-ground height when no STL is set."
    )
    parser.add_argument(
        "--height_scan_obs_offset",
        type=float,
        default=0.5,
        help="Observation offset subtracted from height_scan, matching isaaclab.envs.mdp.height_scan offset.",
    )
    parser.add_argument(
        "--height_scan_geom_groups",
        type=int,
        nargs="+",
        default=(0,),
        help="MuJoCo geom groups included in height_scan ray casts. Default 0 matches the floor geom.",
    )
    parser.add_argument(
        "--show_height_scan",
        action="store_true",
        help="Render MuJoCo height_scan ray hits in the viewer when the policy has a terrain input.",
    )
    parser.add_argument(
        "--height_scan_vis_interval",
        type=int,
        default=1,
        help="Update height_scan visualization every N policy steps.",
    )
    parser.add_argument(
        "--height_scan_vis_points_only",
        action="store_true",
        help="Only render height_scan hit points, without vertical ray lines.",
    )
    parser.add_argument(
        "--height_scan_vis_z_offset",
        type=float,
        default=0.0,
        help="Lift only the rendered height_scan hit points by this many meters to avoid z-fighting.",
    )
    parser.add_argument(
        "--height_scan_vis_radius",
        type=float,
        default=0.02,
        help="Rendered height_scan hit point sphere radius in meters.",
    )
    parser.add_argument(
        "--debug_height_scan_vis",
        action="store_true",
        help="Print how many height_scan debug geoms are appended to the MuJoCo viewer.",
    )
    parser.add_argument(
        "--latent_sample_mode",
        choices=["normal", "zero"],
        default="normal",
        help="How to fill ONNX input 'z' when the policy exposes a latent input.",
    )
    parser.add_argument("--latent_mean", type=float, default=0.0, help="Mean for normal latent z sampling.")
    parser.add_argument(
        "--latent_std", type=float, default=1.0, help="Standard deviation for normal latent z sampling."
    )
    parser.add_argument(
        "--latent_seed",
        type=int,
        default=None,
        help="Random seed for latent z sampling. Omit for non-deterministic samples.",
    )
    parser.add_argument(
        "--latent_resample_interval",
        type=int,
        default=1,
        help="Resample latent z every N policy steps. Values >1 hold z longer for smoother motion.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print MuJoCo rollout progress every N policy steps. Set <=0 to disable.",
    )
    parser.add_argument(
        "--metrics_csv",
        default=None,
        help=(
            "CSV path for per-motion mean tracking metrics. Defaults to "
            "<dataset_txt>.sim2sim_metrics[_tag]_<timestamp>.csv, or "
            "<motion_file>.sim2sim_metrics[_tag]_<timestamp>.csv when --dataset_txt is empty. "
            "Set to an empty string to disable."
        ),
    )
    parser.add_argument("--metrics_tag", default=None, help="Optional tag inserted into the default metrics CSV name.")
    args = parser.parse_args()
    if args.no_render:
        args.render = False
    if args.latent_std < 0.0:
        raise ValueError("--latent_std must be non-negative.")
    if args.latent_resample_interval < 1:
        raise ValueError("--latent_resample_interval must be >= 1.")
    if args.height_scan_vis_interval < 1:
        raise ValueError("--height_scan_vis_interval must be >= 1.")
    if args.dataset_txt is not None and not args.dataset_txt.strip():
        args.dataset_txt = None
    return args


class LatentInputSampler:
    """Adds a sampled latent/noise input for ONNX policies exported with input 'z'."""

    def __init__(
        self,
        input_names: list[str],
        meta: dict,
        *,
        mean: float,
        std: float,
        seed: int | None,
        resample_interval: int,
        sample_mode: str,
    ) -> None:
        self.enabled = "z" in input_names
        self.mean = float(mean)
        self.std = float(std)
        self.resample_interval = max(1, int(resample_interval))
        self.sample_mode = sample_mode
        self.rng = np.random.default_rng(seed)
        self.current: np.ndarray | None = None
        self.shape: tuple[int, ...] | None = None
        if self.enabled:
            shapes = meta.get("observation_shapes", {})
            if "z" not in shapes:
                raise ValueError("ONNX input 'z' requires metadata observation_shapes['z']. Re-export the ONNX.")
            self.shape = (1, *tuple(int(dim) for dim in shapes["z"]))

    def print_config(self) -> None:
        if not self.enabled:
            return
        print(
            "[INFO] Latent/noise z sampler: "
            f"mode={self.sample_mode}, shape={self.shape}, mean={self.mean}, std={self.std}, "
            f"resample_interval={self.resample_interval}"
        )

    def reset(self) -> None:
        self.current = None

    def add_to_obs(self, obs: dict[str, np.ndarray], step: int) -> dict[str, np.ndarray]:
        if not self.enabled:
            return obs
        assert self.shape is not None
        if self.sample_mode == "zero":
            self.current = np.zeros(self.shape, dtype=np.float32)
        elif self.current is None or step % self.resample_interval == 0:
            self.current = self.rng.normal(self.mean, self.std, size=self.shape).astype(np.float32)
        obs["z"] = self.current
        return obs


class KeyboardVelocityCommand:
    """Keyboard-adjustable [vx, vy, yaw_rate] command in the robot yaw frame."""

    GLFW_KEY_KP_0 = 320
    GLFW_KEY_KP_4 = 324
    GLFW_KEY_KP_5 = 325
    GLFW_KEY_KP_6 = 326
    GLFW_KEY_KP_7 = 327
    GLFW_KEY_KP_8 = 328
    GLFW_KEY_KP_9 = 329

    def __init__(
        self,
        *,
        vx_step: float,
        vy_step: float,
        yaw_step: float,
        vx_limit: float,
        vy_limit: float,
        yaw_limit: float,
    ) -> None:
        self.command = np.zeros(3, dtype=np.float32)
        self.steps = np.asarray([vx_step, vy_step, yaw_step], dtype=np.float32)
        self.limits = np.asarray([vx_limit, vy_limit, yaw_limit], dtype=np.float32)

    def on_key(self, key: int) -> None:
        key = int(key)
        if key in (self.GLFW_KEY_KP_8, ord("8")):
            self.command[0] += self.steps[0]
        elif key in (self.GLFW_KEY_KP_5, ord("5")):
            self.command[0] -= self.steps[0]
        elif key in (self.GLFW_KEY_KP_4, ord("4")):
            self.command[1] += self.steps[1]
        elif key in (self.GLFW_KEY_KP_6, ord("6")):
            self.command[1] -= self.steps[1]
        elif key in (self.GLFW_KEY_KP_7, ord("7")):
            self.command[2] += self.steps[2]
        elif key in (self.GLFW_KEY_KP_9, ord("9")):
            self.command[2] -= self.steps[2]
        elif key in (self.GLFW_KEY_KP_0, ord("0")):
            self.command[:] = 0.0
        else:
            return
        self.command[:] = np.clip(self.command, -self.limits, self.limits)
        print(
            f"[VCMD] keyboard velcommand vx={self.command[0]:.3f}, vy={self.command[1]:.3f}, yaw={self.command[2]:.3f}"
        )

    def value(self) -> np.ndarray:
        return self.command.reshape(1, 3).astype(np.float32)

    def print_config(self) -> None:
        print(
            "[INFO] Keyboard velocity command: numpad 8/5=vx, 4/6=vy, 7/9=yaw_rate, 0=zero, "
            f"steps={self.steps.tolist()}, limits={self.limits.tolist()}"
        )


def print_motion_alignment_debug(
    data,
    motion: MotionData,
    meta: dict,
    init_root_body: str | None,
    frame: int,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    default_joint_pos: np.ndarray,
    obs: dict[str, np.ndarray],
    raw_action: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> None:
    root_body, root_idx, motion_pos0, motion_quat0, motion_lin_vel0, motion_ang_vel0 = motion_frame_root_state(
        motion, meta, frame, init_root_body
    )
    summary = motion_local_step_summary(motion, meta, init_root_body)
    joint_pos0 = np.asarray(motion["joint_pos"][frame], dtype=np.float64)
    joint_vel0 = (
        np.asarray(motion["joint_vel"][frame], dtype=np.float64) if "joint_vel" in motion else np.zeros_like(joint_pos0)
    )
    sim_joint_pos = np.asarray(data.qpos[joint_qpos], dtype=np.float64)
    sim_joint_vel = np.asarray(data.qvel[joint_qvel], dtype=np.float64)

    print("========== SIM2SIM MOTION ALIGNMENT DEBUG ==========")
    print(f"[SIMDBG] root body: {root_body} index={root_idx}")
    print(f"[SIMDBG] motion frame{frame} root pos: {motion_pos0.tolist()}")
    print(f"[SIMDBG] motion frame{frame} root quat wxyz: {motion_quat0.tolist()}")
    print(f"[SIMDBG] motion frame{frame} root lin_vel: {motion_lin_vel0.tolist()}")
    print(f"[SIMDBG] motion frame{frame} root ang_vel: {motion_ang_vel0.tolist()}")
    print(f"[SIMDBG] mujoco qpos root pos: {np.asarray(data.qpos[:3]).tolist()}")
    print(f"[SIMDBG] mujoco qpos root quat wxyz: {np.asarray(data.qpos[3:7]).tolist()}")
    print(f"[SIMDBG] mujoco qvel root lin_vel: {np.asarray(data.qvel[:3]).tolist()}")
    print(f"[SIMDBG] mujoco qvel root ang_vel: {np.asarray(data.qvel[3:6]).tolist()}")
    print(f"[SIMDBG] motion local step mean xyz: {summary['mean'].tolist()}")
    print(f"[SIMDBG] motion local step min xyz: {summary['min'].tolist()}")
    print(f"[SIMDBG] motion local step max xyz: {summary['max'].tolist()}")
    print(f"[SIMDBG] motion world total delta xyz: {summary['total'].tolist()}")
    print(f"[SIMDBG] motion joint{frame}-default norm: {np.linalg.norm(joint_pos0 - default_joint_pos):.6f}")
    print(f"[SIMDBG] mujoco joint-motion{frame} norm: {np.linalg.norm(sim_joint_pos - joint_pos0):.6f}")
    print(f"[SIMDBG] mujoco joint_vel-motion{frame} norm: {np.linalg.norm(sim_joint_vel - joint_vel0):.6f}")
    for name, value in obs.items():
        print(
            f"[SIMDBG] obs/{name}: shape={value.shape}, "
            f"norm={np.linalg.norm(value):.6f}, min={float(np.min(value)):.6f}, max={float(np.max(value)):.6f}"
        )
    if raw_action is not None:
        print(
            f"[SIMDBG] raw_action: shape={raw_action.shape}, norm={np.linalg.norm(raw_action):.6f},"
            f" min={float(np.min(raw_action)):.6f}, max={float(np.max(raw_action)):.6f}"
        )
    if target is not None:
        print(
            f"[SIMDBG] target: shape={target.shape}, "
            f"norm={np.linalg.norm(target):.6f}, min={float(np.min(target)):.6f}, max={float(np.max(target)):.6f}"
        )
    print("====================================================")


def _metrics_csv_path(args: argparse.Namespace) -> Path | None:
    if args.metrics_csv == "":
        return None
    if args.metrics_csv is not None:
        return Path(args.metrics_csv).expanduser()
    source_path = Path(args.dataset_txt or args.motion_file).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _safe_filename_part(args.metrics_tag or "")
    suffix = f".sim2sim_metrics_{timestamp}.csv" if not tag else f".sim2sim_metrics_{tag}_{timestamp}.csv"
    if source_path.is_dir() or not source_path.suffix:
        return source_path.with_name(source_path.name + suffix)
    return source_path.with_suffix(source_path.suffix + suffix)


def _safe_filename_part(value: str) -> str:
    safe = []
    for char in value.strip():
        if char.isalnum() or char in ("-", "_", "."):
            safe.append(char)
        elif char.isspace():
            safe.append("_")
    return "".join(safe).strip("._-")


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["motion_index", "motion_file", "num_frames", "samples", *METRIC_NAMES]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_motion_paths(args: argparse.Namespace, paths: list[str]) -> list[tuple[int, str]]:
    indexed_paths = list(enumerate(paths))
    if args.motion_index is not None:
        index = int(args.motion_index)
        if index < 0 or index >= len(indexed_paths):
            raise IndexError(f"--motion_index {index} is out of range for {len(indexed_paths)} motions.")
        indexed_paths = [indexed_paths[index]]
    if args.motion_path:
        query = str(Path(args.motion_path).expanduser())
        query_name = Path(query).name
        matches = [
            item
            for item in indexed_paths
            if str(Path(item[1]).expanduser()) == query
            or Path(item[1]).name == query_name
            or item[1] == args.motion_path
        ]
        if not matches:
            raise ValueError(f"--motion_path {args.motion_path!r} did not match any selected motion.")
        if len(matches) > 1:
            names = [path for _, path in matches[:10]]
            raise ValueError(f"--motion_path {args.motion_path!r} matched multiple motions: {names}")
        indexed_paths = matches
    return indexed_paths


def _manifest_terrain_paths(motion_file: str, dataset_txt: str | None) -> dict[str, Path]:
    if dataset_txt:
        return {}
    manifest_path = Path(motion_file).expanduser()
    if not manifest_path.is_file() or manifest_path.suffix != ".jsonl":
        return {}
    out: dict[str, Path] = {}
    for line_no, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "tracking_motion" not in record or "terrain_stl" not in record:
            continue
        motion_path = _resolve_manifest_path(manifest_path, record["tracking_motion"])
        terrain_path = _resolve_manifest_path(manifest_path, record["terrain_stl"])
        if not terrain_path.is_file():
            raise FileNotFoundError(f"{manifest_path}:{line_no}: terrain_stl not found: {terrain_path}")
        out[str(motion_path)] = terrain_path
    return out


def _resolve_manifest_path(manifest_path: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _terrain_path_for_motion(
    args: argparse.Namespace,
    motion: MotionData,
    motion_path: str,
    manifest_terrain_paths: dict[str, Path],
) -> Path | None:
    if args.terrain_stl:
        return Path(args.terrain_stl).expanduser().resolve()
    manifest_path = manifest_terrain_paths.get(str(Path(motion_path).expanduser().resolve()))
    if manifest_path is not None:
        return manifest_path
    path = terrain_path_from_motion(motion, None)
    return path.resolve() if path is not None else None


def _apply_spawn_height_offset(data, offset: float) -> None:
    offset = float(offset)
    if offset == 0.0:
        return
    data.qpos[2] += offset
    print(f"[INFO] Applied spawn height offset: z += {offset:.3f} m -> root_z={float(data.qpos[2]):.6f}")


def _motion_meta_for_rollout(motion: MotionData, meta: dict) -> dict:
    body_names = list(meta["motion_body_names"])
    motion_meta = dict(meta)
    motion_meta["motion_body_indices"] = list(range(len(body_names)))
    return motion_meta


def _align_motion_for_rollout(
    motion: MotionData,
    meta: dict,
    model,
    joint_names: list[str],
    body_ids: np.ndarray,
) -> tuple[MotionData, dict]:
    body_names = list(meta["motion_body_names"])
    aligned_motion, align_info = motion.aligned_to(
        joint_names=joint_names,
        body_names=body_names,
        model_nbody=int(model.nbody),
        body_ids=body_ids,
    )
    motion_meta = _motion_meta_for_rollout(aligned_motion, meta)
    motion_meta["motion_body_order_source"] = str(align_info.get("body_order_source", "unknown"))
    motion_meta["motion_joint_order_source"] = str(align_info.get("joint_order_source", "unknown"))
    return aligned_motion, motion_meta


def _validate_motion_for_rollout(motion: MotionData, meta: dict, joint_count: int) -> None:
    required = ("joint_pos", "body_pos_w", "body_quat_w")
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"Motion {motion.path} is missing required fields: {missing}")
    if int(motion["joint_pos"].shape[1]) != joint_count:
        raise ValueError(
            f"Motion {motion.path} joint_pos dim {motion['joint_pos'].shape[1]} "
            f"does not match policy/MuJoCo joint dim {joint_count}."
        )
    motion_body_indices = np.asarray(meta["motion_body_indices"], dtype=np.int64)
    body_dim = int(motion["body_pos_w"].shape[1])
    quat_body_dim = int(motion["body_quat_w"].shape[1])
    selected_body_count = len(list(meta["motion_body_names"]))
    if (
        body_dim != quat_body_dim
        or body_dim != selected_body_count
        or motion_body_indices.size != selected_body_count
        or motion_body_indices.size == 0
        or int(motion_body_indices.max()) >= body_dim
    ):
        raise ValueError(
            f"Motion {motion.path} body dim cannot cover selected motion_body_indices "
            f"{motion_body_indices.tolist()}: body_pos_w={motion['body_pos_w'].shape}, "
            f"body_quat_w={motion['body_quat_w'].shape}."
        )
    anchor_name = meta["anchor_body_name"]
    if anchor_name not in list(meta["motion_body_names"]):
        raise ValueError(f"anchor_body_name '{anchor_name}' is not in motion_body_names.")


def _new_reference_player(args, model, motion: MotionData, meta: dict, joint_qpos: np.ndarray):
    if not (args.render and args.show_reference):
        return None
    alpha = float(np.clip(args.reference_alpha, 0.0, 1.0))
    return ReferenceMotionPlayer(
        model,
        motion,
        meta,
        joint_qpos,
        root_body_name=args.reference_root_body,
        rgba=np.asarray([0.2, 0.7, 1.0, alpha], dtype=np.float32),
    )


def _new_height_scanner(
    args: argparse.Namespace,
    model,
    motion: MotionData,
    input_names: list[str],
    terrain_path: Path | None = None,
):
    if "terrain" not in input_names:
        return None
    scanner = MujocoHeightScanner(
        model,
        body_name=args.height_scan_body,
        terrain_path=terrain_path,
        offset=tuple(float(v) for v in args.height_scan_offset),
        ground_z=args.height_scan_ground_z,
        height_offset=args.height_scan_obs_offset,
        geom_groups=tuple(int(v) for v in args.height_scan_geom_groups),
    )
    scanner.print_config()
    return scanner


def _draw_viewer_overlays(
    viewer,
    reference_player,
    terrain_scanner,
    args: argparse.Namespace,
    frame: int,
    *,
    draw_reference: bool,
    draw_height_scan: bool,
    velocity_command: np.ndarray | None = None,
    model=None,
    data=None,
) -> None:
    if viewer is None:
        return
    if reference_player is not None and draw_reference:
        reference_player.draw(viewer, frame)
    elif reference_player is None and (draw_height_scan or velocity_command is not None):
        with viewer.lock():
            viewer.user_scn.ngeom = 0
    if draw_height_scan and terrain_scanner is not None:
        added = terrain_scanner.draw(
            viewer,
            points_only=args.height_scan_vis_points_only,
            z_offset=args.height_scan_vis_z_offset,
            point_radius=args.height_scan_vis_radius,
        )
        if args.debug_height_scan_vis:
            print(f"[HSCANVIS] appended_debug_geoms={added}")
    if velocity_command is not None and model is not None and data is not None:
        _draw_velocity_command_arrow(viewer, model, data, args, velocity_command)


def _velcommand_override(
    args: argparse.Namespace,
    keyboard_command: KeyboardVelocityCommand | None,
) -> np.ndarray | None:
    if args.velocity_command_source == "motion":
        return None
    if args.velocity_command_source == "zero":
        return np.zeros((1, 3), dtype=np.float32)
    if keyboard_command is None:
        return np.zeros((1, 3), dtype=np.float32)
    return keyboard_command.value()


def _draw_velocity_command_arrow(viewer, model, data, args: argparse.Namespace, command: np.ndarray) -> None:
    import mujoco

    cmd = np.asarray(command, dtype=np.float64).reshape(-1)
    if cmd.shape[0] < 3:
        return
    body_id = int(model.body(args.camera_body if args.camera_body else "pelvis").id)
    root_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
    body_xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    yaw = float(np.arctan2(body_xmat[1, 0], body_xmat[0, 0]))
    c = np.cos(yaw)
    s = np.sin(yaw)
    vel_w = np.asarray([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], 0.0], dtype=np.float64)
    start = np.asarray([root_pos[0], root_pos[1], args.velocity_command_arrow_z], dtype=np.float64)
    end = start + float(args.velocity_command_arrow_scale) * vel_w
    if np.linalg.norm(end - start) < 1e-4:
        end = start + np.asarray([0.001, 0.0, 0.0], dtype=np.float64)
    rgba = np.asarray([1.0, 0.1, 0.1, 1.0], dtype=np.float32)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    with viewer.lock():
        scn = viewer.user_scn
        max_geoms = getattr(scn, "maxgeom", None)
        if max_geoms is None:
            max_geoms = len(scn.geoms)
        max_geoms = int(max_geoms)
        if scn.ngeom >= max_geoms:
            return
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            mat,
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.035,
            start,
            end,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.rgba[:] = rgba
        scn.ngeom += 1


def _update_viewer_camera(viewer, model, data, args: argparse.Namespace) -> None:
    if viewer is None or not args.camera_follow:
        return
    try:
        body_id = int(model.body(args.camera_body).id)
    except KeyError as exc:
        raise ValueError(f"--camera_body {args.camera_body!r} is not a MuJoCo body name.") from exc
    lookat = np.asarray(data.xpos[body_id], dtype=np.float64) + np.asarray(args.camera_lookat_offset, dtype=np.float64)
    with viewer.lock():
        viewer.cam.lookat[:] = lookat
        viewer.cam.distance = float(args.camera_distance)
        viewer.cam.azimuth = float(args.camera_azimuth)
        viewer.cam.elevation = float(args.camera_elevation)


def _print_height_scan_debug(terrain_scanner) -> None:
    if terrain_scanner is None or terrain_scanner.last_scan is None:
        print("[HSCANDBG] height scanner has no cached scan yet.")
        return
    cache = terrain_scanner.last_scan
    values = np.asarray(cache["values"], dtype=np.float64).reshape(-1)
    hit_points = np.asarray(cache["hit_points"], dtype=np.float64)
    ray_starts = np.asarray(cache["ray_starts"], dtype=np.float64)
    scan_origin = np.asarray(cache["scan_origin"], dtype=np.float64)
    geom_ids = np.asarray(cache.get("geom_ids", []), dtype=np.int32).reshape(-1)
    hit_count = int(np.sum(geom_ids >= 0)) if geom_ids.size else 0
    print("========== MUJOCO HEIGHT SCAN DEBUG ==========")
    print(f"[HSCANDBG] scan_origin: {scan_origin.tolist()}")
    print(
        f"[HSCANDBG] ray_start_z range: ({float(np.min(ray_starts[:, 2])):.6f}, {float(np.max(ray_starts[:, 2])):.6f})"
    )
    print(
        "[HSCANDBG] hit_z range: "
        f"({float(np.min(hit_points[:, 2])):.6f}, {float(np.max(hit_points[:, 2])):.6f}), "
        f"mj_ray_hits={hit_count}/{hit_points.shape[0]}"
    )
    print(
        "[HSCANDBG] obs range: "
        f"({float(np.min(values)):.6f}, {float(np.max(values)):.6f}), "
        f"mean={float(np.mean(values)):.6f}"
    )
    print("[HSCANDBG] IsaacLab formula: torso_body_z - hit_z - obs_offset, ray offset is only ray start.")
    print("==============================================")


def _motion_start_frame(args: argparse.Namespace, motion: MotionData) -> int:
    if motion.num_frames <= 0:
        raise ValueError(f"Motion {motion.path} has no frames.")
    requested = int(args.motion_start_frame)
    start_frame = int(np.clip(requested, 0, motion.num_frames - 1))
    if start_frame != requested:
        print(
            f"[WARN] Requested motion_start_frame={requested} is outside motion frame range "
            f"[0, {motion.num_frames - 1}], clamped to {start_frame}."
        )
    return start_frame


def main() -> None:
    args = parse_args()

    import mujoco

    meta = load_metadata(args.onnx_path)
    input_names = onnx_input_names(args.onnx_path)
    validate_grouped_onnx_contract(input_names, meta)
    if args.velocity_command_source != "motion" and "velcommand" not in input_names:
        raise ValueError(
            f"--velocity_command_source {args.velocity_command_source!r} requires an ONNX input named 'velcommand'. "
            f"Current inputs: {input_names}"
        )
    if "smpl_cmd_mf" in input_names and meta.get("encoder_mode") not in (None, "robot", "encoder_g1", "g1"):
        print(
            f"[WARN] ONNX encoder_mode={meta.get('encoder_mode')} needs non-zero smpl_cmd_mf. "
            "This script currently feeds zero SMPL observations."
        )

    all_motion_paths = motion_files(args.motion_file, args.dataset_txt)
    motion_entries = _select_motion_paths(args, all_motion_paths)
    motion_paths = [path for _, path in motion_entries]
    if not motion_paths:
        raise ValueError("No motion files to run.")
    metrics_csv_path = _metrics_csv_path(args)
    print(f"[INFO] Motion count: {len(motion_paths)} selected from {len(all_motion_paths)}")
    if args.motion_index is not None or args.motion_path:
        for original_index, path in motion_entries:
            print(f"[INFO] Selected motion original_index={original_index}: {path}")
    if metrics_csv_path is not None:
        print(f"[INFO] Metrics CSV: {metrics_csv_path}")

    manifest_terrain_paths = _manifest_terrain_paths(args.motion_file, args.dataset_txt)
    first_selected_index, first_motion_path = motion_entries[0]
    first_motion = MotionData(first_motion_path)
    motion = first_motion
    print(f"[INFO] Motion 1/{len(motion_paths)} (original_index={first_selected_index}): {motion.path}")
    motion.print_config()

    active_terrain_path = _terrain_path_for_motion(args, first_motion, first_motion_path, manifest_terrain_paths)
    model_xml_path, terrain_tmpdir, active_model_terrain_path = mujoco_xml_with_terrain_mesh(
        args.xml_path,
        active_terrain_path,
        mode=args.terrain_mode,
        collision_backend=args.terrain_collision_backend,
        geom_group=args.terrain_mesh_group,
    )
    if active_model_terrain_path is None:
        print(f"[INFO] MuJoCo terrain: flat floor from XML ({args.xml_path})")
    else:
        print(f"[INFO] MuJoCo terrain: injected STL mesh {active_model_terrain_path}")
    model = mujoco.MjModel.from_xml_path(model_xml_path)
    model.opt.timestep = float(meta.get("sim_dt", model.opt.timestep))
    data = mujoco.MjData(model)
    joint_names = list(meta["action_joint_names"])
    body_names = list(meta["motion_body_names"])
    # joint_ids = name_to_joint_ids(model, joint_names)
    actuator_ids = name_to_actuator_ids(model, joint_names)
    joint_qpos, joint_qvel = name_to_joint_qvel_addrs(model, joint_names)
    body_ids = name_to_body_ids(model, body_names)
    imu_reader = ImuReader(
        model,
        quat_sensor_name=args.imu_quat_sensor,
        gyro_sensor_name=args.imu_gyro_sensor,
    )
    torque_limits = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)

    motion, motion_meta = _align_motion_for_rollout(motion, meta, model, joint_names, body_ids)
    _validate_motion_for_rollout(motion, motion_meta, len(joint_names))
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_names), 0.0)
    start_frame = _motion_start_frame(args, motion)
    if args.init_from_motion:
        init_root_body = initialize_from_motion(
            data, motion, motion_meta, joint_qpos, joint_qvel, args.init_root_body, frame=start_frame
        )
        print(f"[INFO] Initialized MuJoCo state from motion frame {start_frame} using root body: {init_root_body}")
    else:
        initialize_default_pose(data, meta, joint_names, joint_qpos)
        print("[INFO] Initialized MuJoCo state from default standing pose.")
    _apply_spawn_height_offset(data, args.spawn_height_offset)
    mujoco.mj_forward(model, data)
    imu_reader.print_config()
    if args.debug_imu:
        print_imu_debug(data, imu_reader)

    action_scale = as_vector(meta, "action_scale", len(joint_names), 1.0)
    action_offset = as_vector(meta, "action_offset", len(joint_names), 0.0)
    kp, kd = gains_from_metadata(meta, len(joint_names), args.kp, args.kd)
    print(
        "[INFO] PD gains: "
        f"kp_range=({float(np.min(kp)):.4f}, {float(np.max(kp)):.4f}), "
        f"kd_range=({float(np.min(kd)):.4f}, {float(np.max(kd)):.4f})"
    )

    decimation = args.decimation or int(meta.get("decimation", 1))
    reference_update_interval = max(1, int(args.reference_update_interval))
    height_scan_vis_interval = max(1, int(args.height_scan_vis_interval))
    latent_sampler = LatentInputSampler(
        input_names,
        meta,
        mean=args.latent_mean,
        std=args.latent_std,
        seed=args.latent_seed,
        resample_interval=args.latent_resample_interval,
        sample_mode=args.latent_sample_mode,
    )
    latent_sampler.print_config()
    keyboard_command = (
        KeyboardVelocityCommand(
            vx_step=args.keyboard_vx_step,
            vy_step=args.keyboard_vy_step,
            yaw_step=args.keyboard_yaw_step,
            vx_limit=args.keyboard_vx_limit,
            vy_limit=args.keyboard_vy_limit,
            yaw_limit=args.keyboard_yaw_limit,
        )
        if args.velocity_command_source == "keyboard"
        else None
    )
    if keyboard_command is not None:
        keyboard_command.print_config()
    # print_obs_layout(meta, prop_history, input_names)
    if args.dry_run:
        last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
        prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
        terrain_scanner = _new_height_scanner(args, model, motion, input_names, active_model_terrain_path)
        obs = build_obs(
            data,
            motion,
            start_frame,
            motion_meta,
            imu_reader,
            joint_qpos,
            joint_qvel,
            last_action,
            prop_history,
            body_ids,
            terrain_scanner,
            _velcommand_override(args, keyboard_command),
        )
        latent_sampler.add_to_obs(obs, step=0)
        validate_inputs(obs, input_names, meta)
        if args.debug_height_scan:
            _print_height_scan_debug(terrain_scanner)
        if args.debug_motion_alignment:
            print_motion_alignment_debug(
                data,
                motion,
                motion_meta,
                args.init_root_body,
                start_frame,
                joint_qpos,
                joint_qvel,
                default_joint_pos,
                obs,
            )
        print("[INFO] dry_run observation validation passed.")
        return

    policy = OnnxPolicy(args.onnx_path)
    input_names = policy.input_names
    if args.debug_motion_alignment:
        last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
        debug_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
        terrain_scanner = _new_height_scanner(args, model, motion, input_names, active_model_terrain_path)
        debug_obs = build_obs(
            data,
            motion,
            start_frame,
            motion_meta,
            imu_reader,
            joint_qpos,
            joint_qvel,
            last_action,
            debug_history,
            body_ids,
            terrain_scanner,
            _velcommand_override(args, keyboard_command),
        )
        latent_sampler.add_to_obs(debug_obs, step=0)
        validate_inputs(debug_obs, input_names, meta)
        if args.debug_height_scan:
            _print_height_scan_debug(terrain_scanner)
        debug_action = policy.run(debug_obs)
        debug_target = action_to_target(debug_action, action_scale, action_offset)
        print_motion_alignment_debug(
            data,
            motion,
            motion_meta,
            args.init_root_body,
            start_frame,
            joint_qpos,
            joint_qvel,
            default_joint_pos,
            debug_obs,
            debug_action,
            debug_target,
        )

    viewer_cm = None
    viewer = None
    if args.render:
        import mujoco.viewer

        viewer_cm = mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=keyboard_command.on_key if keyboard_command is not None else None,
        )
        viewer = viewer_cm.__enter__()
        _update_viewer_camera(viewer, model, data, args)
    else:
        print("[INFO] MuJoCo viewer disabled. Pass --render to watch the rollout.")

    print(
        f"[INFO] Running MuJoCo sim2sim: max_steps_per_motion={args.steps}, decimation={decimation}, "
        f"dt={model.opt.timestep:.6f}, render={args.render}"
    )
    metric_rows: list[dict[str, float | int | str]] = []
    hold_target: np.ndarray | None = None
    hold_last_action: np.ndarray | None = None
    hold_reference_player = None
    hold_terrain_scanner = None
    hold_motion = None
    hold_motion_meta = None
    hold_prop_history = None
    hold_frame = 0

    try:
        for selected_index, (motion_index, motion_path) in enumerate(motion_entries):
            if selected_index == 0:
                motion = first_motion
            else:
                motion = MotionData(motion_path)
                print(
                    f"[INFO] Motion {selected_index + 1}/{len(motion_entries)} "
                    f"(original_index={motion_index}): {motion.path}"
                )
                motion.print_config()
            motion_terrain_path = _terrain_path_for_motion(args, motion, motion_path, manifest_terrain_paths)
            if active_model_terrain_path is not None and motion_terrain_path is not None:
                if motion_terrain_path.resolve() != active_model_terrain_path.resolve():
                    raise ValueError(
                        "Current MuJoCo mesh terrain mode supports one compiled terrain per run. "
                        f"First terrain is {active_model_terrain_path}, but motion {motion.path} requests "
                        f"{motion_terrain_path}. Use --motion_index/--motion_path for one terrain at a time."
                    )
            motion, motion_meta = _align_motion_for_rollout(motion, meta, model, joint_names, body_ids)
            _validate_motion_for_rollout(motion, motion_meta, len(joint_names))
            start_frame = _motion_start_frame(args, motion)

            if args.init_from_motion:
                init_root_body = initialize_from_motion(
                    data, motion, motion_meta, joint_qpos, joint_qvel, args.init_root_body, frame=start_frame
                )
                print(f"[INFO] Aligned MuJoCo state to motion frame {start_frame} using root body: {init_root_body}")
            else:
                initialize_default_pose(data, meta, joint_names, joint_qpos)
                print("[INFO] Reset MuJoCo state to default standing pose.")
            _apply_spawn_height_offset(data, args.spawn_height_offset)
            mujoco.mj_forward(model, data)

            reference_player = _new_reference_player(args, model, motion, motion_meta, joint_qpos)
            if reference_player is not None:
                reference_player.print_config()
            if viewer is not None and reference_player is not None:
                _draw_viewer_overlays(
                    viewer,
                    reference_player,
                    terrain_scanner=None,
                    args=args,
                    frame=start_frame,
                    draw_reference=True,
                    draw_height_scan=False,
                )
                _update_viewer_camera(viewer, model, data, args)
                viewer.sync()

            last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
            prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
            terrain_scanner = _new_height_scanner(args, model, motion, input_names, active_model_terrain_path)
            latent_sampler.reset()
            # start_root_pos = np.asarray(data.qpos[:3], dtype=np.float64).copy()
            rollout_steps = min(max(int(args.steps), 0), int(motion.num_frames) - start_frame)
            accumulator = MotionMetricAccumulator(
                motion_index=motion_index,
                motion_file=motion.path,
                num_frames=motion.num_frames,
            )
            print(
                f"[INFO] Running motion {selected_index + 1}/{len(motion_entries)} "
                f"(original_index={motion_index}): "
                f"policy_steps={rollout_steps}, frames={motion.num_frames}, start_frame={start_frame}"
            )

            for step in range(rollout_steps):
                t = start_frame + step
                metrics = motion_tracking_metrics(model, data, motion, t, motion_meta, joint_qpos, joint_qvel, body_ids)
                accumulator.update(metrics)

                obs = build_obs(
                    data,
                    motion,
                    t,
                    motion_meta,
                    imu_reader,
                    joint_qpos,
                    joint_qvel,
                    last_action,
                    prop_history,
                    body_ids,
                    terrain_scanner,
                    _velcommand_override(args, keyboard_command),
                )
                latent_sampler.add_to_obs(obs, step=step)
                validate_inputs(obs, input_names, meta)
                if args.debug_height_scan and step == 0:
                    _print_height_scan_debug(terrain_scanner)
                raw_action = policy.run(obs)
                target = action_to_target(raw_action, action_scale, action_offset)
                last_action = raw_action
                hold_target = target
                hold_last_action = last_action
                hold_reference_player = reference_player
                hold_terrain_scanner = terrain_scanner
                hold_motion = motion
                hold_motion_meta = motion_meta
                hold_prop_history = prop_history
                hold_frame = t

                draw_reference = reference_player is not None and step % reference_update_interval == 0
                draw_height_scan = (
                    bool(args.show_height_scan) and terrain_scanner is not None and step % height_scan_vis_interval == 0
                )
                draw_velocity_command = bool(args.show_velocity_command) and "velcommand" in obs
                if viewer is not None and (draw_reference or draw_height_scan or draw_velocity_command):
                    _draw_viewer_overlays(
                        viewer,
                        reference_player,
                        terrain_scanner,
                        args,
                        t,
                        draw_reference=draw_reference or draw_height_scan or draw_velocity_command,
                        draw_height_scan=draw_height_scan,
                        velocity_command=obs.get("velcommand"),
                        model=model,
                        data=data,
                    )
                # input("Press Enter to step the simulation...")  # Step on Enter key press
                if viewer is not None:
                    import time

                    time.sleep(decimation * model.opt.timestep)
                    _update_viewer_camera(viewer, model, data, args)
                    viewer.sync()

                for _ in range(decimation):
                    apply_pd_control(
                        data,
                        actuator_ids,
                        joint_qpos,
                        joint_qvel,
                        target,
                        kp,
                        kd,
                        torque_limits,
                    )
                    mujoco.mj_step(model, data)
                if args.log_interval > 0 and (step % args.log_interval == 0 or step == rollout_steps - 1):
                    print(
                        f"[INFO] motion {selected_index + 1}/{len(motion_entries)} "
                        f"original_index={motion_index} frame={t}/{motion.num_frames - 1} "
                        f"step={step + 1}/{rollout_steps}"
                    )

            row = accumulator.row()
            metric_rows.append(row)
            if metrics_csv_path is not None:
                _write_metrics_csv(metrics_csv_path, metric_rows)
                print(
                    f"[INFO] Updated metrics CSV: {metrics_csv_path} ({len(metric_rows)}/{len(motion_entries)} motions)"
                )
            print(
                "[INFO] Motion metrics mean: "
                f"error_anchor_pos={float(row['error_anchor_pos']):.6f}, "
                f"error_body_pos={float(row['error_body_pos']):.6f}, "
                f"error_joint_pos={float(row['error_joint_pos']):.6f}, "
                f"samples={int(row['samples'])}"
            )

        print(
            f"[INFO] sim2sim completed: motions={len(motion_entries)}, "
            f"policy_steps={sum(int(row['samples']) for row in metric_rows)}, "
            f"sim_time={sum(int(row['samples']) for row in metric_rows) * decimation * model.opt.timestep:.3f}s"
        )
        if metrics_csv_path is not None:
            print(f"[INFO] Wrote per-motion mean metrics: {metrics_csv_path}")
        if viewer is not None and args.hold_final_frame:
            print(
                "[INFO] Continuing MuJoCo simulation on the final reference frame "
                f"(policy_mode={args.hold_final_policy_mode}). "
                "Close the viewer window or press Ctrl-C to exit."
            )
            hold_step = 0
            while viewer.is_running():
                import time

                hold_velocity_command = None
                if (
                    args.hold_final_policy_mode == "run_policy"
                    and hold_motion is not None
                    and hold_motion_meta is not None
                    and hold_prop_history is not None
                    and hold_last_action is not None
                ):
                    hold_obs = build_obs(
                        data,
                        hold_motion,
                        hold_frame,
                        hold_motion_meta,
                        imu_reader,
                        joint_qpos,
                        joint_qvel,
                        hold_last_action,
                        hold_prop_history,
                        body_ids,
                        hold_terrain_scanner,
                        _velcommand_override(args, keyboard_command),
                    )
                    latent_sampler.add_to_obs(hold_obs, step=hold_step)
                    validate_inputs(hold_obs, input_names, meta)
                    hold_velocity_command = hold_obs.get("velcommand")
                    hold_last_action = policy.run(hold_obs)
                    hold_target = action_to_target(hold_last_action, action_scale, action_offset)
                    hold_step += 1
                if hold_target is not None:
                    for _ in range(decimation):
                        apply_pd_control(
                            data,
                            actuator_ids,
                            joint_qpos,
                            joint_qvel,
                            hold_target,
                            kp,
                            kd,
                            torque_limits,
                        )
                        mujoco.mj_step(model, data)
                if hold_terrain_scanner is not None:
                    hold_terrain_scanner.scan(data)
                if (
                    hold_reference_player is not None
                    or (args.show_height_scan and hold_terrain_scanner is not None)
                    or (args.show_velocity_command and hold_velocity_command is not None)
                ):
                    _draw_viewer_overlays(
                        viewer,
                        hold_reference_player,
                        hold_terrain_scanner,
                        args,
                        hold_frame,
                        draw_reference=hold_reference_player is not None,
                        draw_height_scan=bool(args.show_height_scan) and hold_terrain_scanner is not None,
                        velocity_command=hold_velocity_command if args.show_velocity_command else None,
                        model=model,
                        data=data,
                    )
                time.sleep(decimation * model.opt.timestep)
                viewer.sync()
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)
        if terrain_tmpdir is not None:
            terrain_tmpdir.cleanup()


if __name__ == "__main__":
    main()
