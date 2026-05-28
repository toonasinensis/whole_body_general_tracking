from __future__ import annotations

import argparse
import csv
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
from sim2sim_g1.viewer import ReferenceMotionPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run G1 ONNX policy in MuJoCo.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--dataset_txt", default=None)
    parser.add_argument("--xml_path", default=str(G1_MJCF))
    parser.add_argument("--steps", type=int, default=20000000)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--kp", type=float, default=None, help="Override metadata joint stiffness with a scalar value.")
    parser.add_argument("--kd", type=float, default=None, help="Override metadata joint damping with a scalar value.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no_render", action="store_true", help="Disable MuJoCo viewer when wrapper enables --render.")
    parser.add_argument("--imu_quat_sensor", default="base_quat", help="MuJoCo framequat sensor name.")
    parser.add_argument("--imu_gyro_sensor", default="base_gyro", help="MuJoCo gyro sensor name.")
    parser.add_argument("--debug_imu", action="store_true", help="Print XML sensor IMU values once at startup.")
    parser.add_argument(
        "--debug_motion_alignment",
        action="store_true",
        help="Print motion/root/obs/action alignment values once at startup.",
    )
    parser.add_argument(
        "--init_from_motion",
        action="store_true",
        default=True,
        help="Initialize MuJoCo root, joints, and velocities from motion frame 0.",
    )
    parser.add_argument("--no_init_from_motion", action="store_false", dest="init_from_motion")
    parser.add_argument(
        "--init_root_body",
        default=None,
        help="Motion body used to initialize the floating base. Defaults to metadata root body or first motion body.",
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
    if args.dataset_txt is not None and not args.dataset_txt.strip():
        args.dataset_txt = None
    return args


def print_motion_alignment_debug(
    data,
    motion: MotionData,
    meta: dict,
    init_root_body: str | None,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    default_joint_pos: np.ndarray,
    obs: dict[str, np.ndarray],
    raw_action: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> None:
    root_body, root_idx, motion_pos0, motion_quat0, motion_lin_vel0, motion_ang_vel0 = motion_frame_root_state(
        motion, meta, 0, init_root_body
    )
    summary = motion_local_step_summary(motion, meta, init_root_body)
    joint_pos0 = np.asarray(motion["joint_pos"][0], dtype=np.float64)
    joint_vel0 = (
        np.asarray(motion["joint_vel"][0], dtype=np.float64) if "joint_vel" in motion else np.zeros_like(joint_pos0)
    )
    sim_joint_pos = np.asarray(data.qpos[joint_qpos], dtype=np.float64)
    sim_joint_vel = np.asarray(data.qvel[joint_qvel], dtype=np.float64)

    print("========== SIM2SIM MOTION ALIGNMENT DEBUG ==========")
    print(f"[SIMDBG] root body: {root_body} index={root_idx}")
    print(f"[SIMDBG] motion frame0 root pos: {motion_pos0.tolist()}")
    print(f"[SIMDBG] motion frame0 root quat wxyz: {motion_quat0.tolist()}")
    print(f"[SIMDBG] motion frame0 root lin_vel: {motion_lin_vel0.tolist()}")
    print(f"[SIMDBG] motion frame0 root ang_vel: {motion_ang_vel0.tolist()}")
    print(f"[SIMDBG] mujoco qpos root pos: {np.asarray(data.qpos[:3]).tolist()}")
    print(f"[SIMDBG] mujoco qpos root quat wxyz: {np.asarray(data.qpos[3:7]).tolist()}")
    print(f"[SIMDBG] mujoco qvel root lin_vel: {np.asarray(data.qvel[:3]).tolist()}")
    print(f"[SIMDBG] mujoco qvel root ang_vel: {np.asarray(data.qvel[3:6]).tolist()}")
    print(f"[SIMDBG] motion local step mean xyz: {summary['mean'].tolist()}")
    print(f"[SIMDBG] motion local step min xyz: {summary['min'].tolist()}")
    print(f"[SIMDBG] motion local step max xyz: {summary['max'].tolist()}")
    print(f"[SIMDBG] motion world total delta xyz: {summary['total'].tolist()}")
    print(f"[SIMDBG] motion joint0-default norm: {np.linalg.norm(joint_pos0 - default_joint_pos):.6f}")
    print(f"[SIMDBG] mujoco joint-motion0 norm: {np.linalg.norm(sim_joint_pos - joint_pos0):.6f}")
    print(f"[SIMDBG] mujoco joint_vel-motion0 norm: {np.linalg.norm(sim_joint_vel - joint_vel0):.6f}")
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


def main() -> None:
    args = parse_args()

    import mujoco

    meta = load_metadata(args.onnx_path)
    input_names = onnx_input_names(args.onnx_path)
    validate_grouped_onnx_contract(input_names, meta)
    if meta.get("encoder_mode") not in (None, "robot", "encoder_g1", "g1"):
        print(
            f"[WARN] ONNX encoder_mode={meta.get('encoder_mode')} needs non-zero smpl_cmd_mf. "
            "This script currently feeds zero SMPL observations."
        )

    motion_paths = motion_files(args.motion_file, args.dataset_txt)
    if not motion_paths:
        raise ValueError("No motion files to run.")
    metrics_csv_path = _metrics_csv_path(args)
    print(f"[INFO] Motion count: {len(motion_paths)}")
    if metrics_csv_path is not None:
        print(f"[INFO] Metrics CSV: {metrics_csv_path}")

    first_motion = MotionData(motion_paths[0])
    motion = first_motion
    print(f"[INFO] Motion 1/{len(motion_paths)}: {motion.path}")
    motion.print_config()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
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
    if args.init_from_motion:
        init_root_body = initialize_from_motion(data, motion, motion_meta, joint_qpos, joint_qvel, args.init_root_body)
        print(f"[INFO] Initialized MuJoCo state from motion frame 0 using root body: {init_root_body}")
    else:
        initialize_default_pose(data, meta, joint_names, joint_qpos)
        print("[INFO] Initialized MuJoCo state from default standing pose.")
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
    # print_obs_layout(meta, prop_history, input_names)
    if args.dry_run:
        last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
        prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
        obs = build_obs(data, motion, 0, motion_meta, imu_reader, joint_qpos, joint_qvel, last_action, prop_history)
        validate_inputs(obs, input_names, meta)
        if args.debug_motion_alignment:
            print_motion_alignment_debug(
                data,
                motion,
                motion_meta,
                args.init_root_body,
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
        debug_obs = build_obs(
            data, motion, 0, motion_meta, imu_reader, joint_qpos, joint_qvel, last_action, debug_history
        )
        validate_inputs(debug_obs, input_names, meta)
        debug_action = policy.run(debug_obs)
        debug_target = action_to_target(debug_action, action_scale, action_offset)
        print_motion_alignment_debug(
            data,
            motion,
            motion_meta,
            args.init_root_body,
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

        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()
    else:
        print("[INFO] MuJoCo viewer disabled. Pass --render to watch the rollout.")

    print(
        f"[INFO] Running MuJoCo sim2sim: max_steps_per_motion={args.steps}, decimation={decimation}, "
        f"dt={model.opt.timestep:.6f}, render={args.render}"
    )
    metric_rows: list[dict[str, float | int | str]] = []

    try:
        for motion_index, motion_path in enumerate(motion_paths):
            if motion_index == 0:
                motion = first_motion
            else:
                motion = MotionData(motion_path)
                print(f"[INFO] Motion {motion_index + 1}/{len(motion_paths)}: {motion.path}")
                motion.print_config()
            motion, motion_meta = _align_motion_for_rollout(motion, meta, model, joint_names, body_ids)
            _validate_motion_for_rollout(motion, motion_meta, len(joint_names))

            if args.init_from_motion:
                init_root_body = initialize_from_motion(
                    data, motion, motion_meta, joint_qpos, joint_qvel, args.init_root_body
                )
                print(f"[INFO] Aligned MuJoCo state to motion frame 0 using root body: {init_root_body}")
            else:
                initialize_default_pose(data, meta, joint_names, joint_qpos)
                print("[INFO] Reset MuJoCo state to default standing pose.")
            mujoco.mj_forward(model, data)

            reference_player = _new_reference_player(args, model, motion, motion_meta, joint_qpos)
            if reference_player is not None:
                reference_player.print_config()
            if viewer is not None and reference_player is not None:
                reference_player.draw(viewer, 0)
                viewer.sync()

            last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
            prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
            # start_root_pos = np.asarray(data.qpos[:3], dtype=np.float64).copy()
            rollout_steps = min(max(int(args.steps), 0), int(motion.num_frames))
            accumulator = MotionMetricAccumulator(
                motion_index=motion_index,
                motion_file=motion.path,
                num_frames=motion.num_frames,
            )
            print(
                f"[INFO] Running motion {motion_index + 1}/{len(motion_paths)}: "
                f"policy_steps={rollout_steps}, frames={motion.num_frames}"
            )

            for step in range(rollout_steps):
                t = step
                metrics = motion_tracking_metrics(model, data, motion, t, motion_meta, joint_qpos, joint_qvel, body_ids)
                accumulator.update(metrics)

                obs = build_obs(
                    data, motion, t, motion_meta, imu_reader, joint_qpos, joint_qvel, last_action, prop_history
                )
                validate_inputs(obs, input_names, meta)
                raw_action = policy.run(obs)
                target = action_to_target(raw_action, action_scale, action_offset)
                last_action = raw_action

                if viewer is not None and reference_player is not None and step % reference_update_interval == 0:
                    reference_player.draw(viewer, t)
                # input("Press Enter to step the simulation...")  # Step on Enter key press
                if viewer is not None:
                    import time

                    time.sleep(decimation * model.opt.timestep)
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

            row = accumulator.row()
            metric_rows.append(row)
            if metrics_csv_path is not None:
                _write_metrics_csv(metrics_csv_path, metric_rows)
                print(
                    f"[INFO] Updated metrics CSV: {metrics_csv_path} ({len(metric_rows)}/{len(motion_paths)} motions)"
                )
            print(
                "[INFO] Motion metrics mean: "
                f"error_anchor_pos={float(row['error_anchor_pos']):.6f}, "
                f"error_body_pos={float(row['error_body_pos']):.6f}, "
                f"error_joint_pos={float(row['error_joint_pos']):.6f}, "
                f"samples={int(row['samples'])}"
            )

        print(
            f"[INFO] sim2sim completed: motions={len(motion_paths)}, "
            f"policy_steps={sum(int(row['samples']) for row in metric_rows)}, "
            f"sim_time={sum(int(row['samples']) for row in metric_rows) * decimation * model.opt.timestep:.3f}s"
        )
        if metrics_csv_path is not None:
            print(f"[INFO] Wrote per-motion mean metrics: {metrics_csv_path}")
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
