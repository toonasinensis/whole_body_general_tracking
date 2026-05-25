from __future__ import annotations

import argparse
import numpy as np
import time

from sim2sim_g1.math_utils import as_vector, quat_apply_inverse
from sim2sim_g1.motion import MotionData, first_motion_file, motion_frame_root_state, motion_local_step_summary
from sim2sim_g1.mujoco_robot import G1_MJCF  # noqa: F401
from sim2sim_g1.mujoco_robot import (  # print_joint_map,; name_to_joint_ids,
    action_to_target,
    apply_pd_control,
    gains_from_metadata,
    initialize_default_pose,
    initialize_from_motion,
    name_to_actuator_ids,
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
    parser.add_argument("--steps", type=int, default=2000)
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
    args = parser.parse_args()
    if args.no_render:
        args.render = False
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

    motion_path = first_motion_file(args.motion_file, args.dataset_txt)
    print(f"[INFO] Motion: {motion_path}")
    motion = MotionData(motion_path)
    motion.print_config()

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    model.opt.timestep = float(meta.get("sim_dt", model.opt.timestep))
    data = mujoco.MjData(model)
    joint_names = list(meta["action_joint_names"])
    # joint_ids = name_to_joint_ids(model, joint_names)
    actuator_ids = name_to_actuator_ids(model, joint_names)
    joint_qpos, joint_qvel = name_to_joint_qvel_addrs(model, joint_names)
    imu_reader = ImuReader(
        model,
        quat_sensor_name=args.imu_quat_sensor,
        gyro_sensor_name=args.imu_gyro_sensor,
    )
    reference_player = None
    if args.render and args.show_reference:
        alpha = float(np.clip(args.reference_alpha, 0.0, 1.0))
        reference_player = ReferenceMotionPlayer(
            model,
            motion,
            meta,
            joint_qpos,
            root_body_name=args.reference_root_body,
            rgba=np.asarray([0.2, 0.7, 1.0, alpha], dtype=np.float32),
        )
    torque_limits = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)

    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_names), 0.0)
    if args.init_from_motion:
        init_root_body = initialize_from_motion(data, motion, meta, joint_qpos, joint_qvel, args.init_root_body)
        print(f"[INFO] Initialized MuJoCo state from motion frame 0 using root body: {init_root_body}")
    else:
        initialize_default_pose(data, meta, joint_names, joint_qpos)
        print("[INFO] Initialized MuJoCo state from default standing pose.")
    mujoco.mj_forward(model, data)
    imu_reader.print_config()
    if args.debug_imu:
        print_imu_debug(data, imu_reader)
    if reference_player is not None:
        reference_player.print_config()

    action_scale = as_vector(meta, "action_scale", len(joint_names), 1.0)
    action_offset = as_vector(meta, "action_offset", len(joint_names), 0.0)
    kp, kd = gains_from_metadata(meta, len(joint_names), args.kp, args.kd)
    print(
        "[INFO] PD gains: "
        f"kp_range=({float(np.min(kp)):.4f}, {float(np.max(kp)):.4f}), "
        f"kd_range=({float(np.min(kd)):.4f}, {float(np.max(kd)):.4f})"
    )

    decimation = args.decimation or int(meta.get("decimation", 1))
    last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
    prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
    reference_update_interval = max(1, int(args.reference_update_interval))
    # print_obs_layout(meta, prop_history, input_names)
    if args.dry_run:
        obs = build_obs(data, motion, 0, meta, imu_reader, joint_qpos, joint_qvel, last_action, prop_history)
        validate_inputs(obs, input_names, meta)
        if args.debug_motion_alignment:
            print_motion_alignment_debug(
                data,
                motion,
                meta,
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
        debug_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
        debug_obs = build_obs(data, motion, 0, meta, imu_reader, joint_qpos, joint_qvel, last_action, debug_history)
        validate_inputs(debug_obs, input_names, meta)
        debug_action = policy.run(debug_obs)
        debug_target = action_to_target(debug_action, action_scale, action_offset)
        print_motion_alignment_debug(
            data,
            motion,
            meta,
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
        if reference_player is not None:
            reference_player.draw(viewer, 0)
    else:
        print("[INFO] MuJoCo viewer disabled. Pass --render to watch the rollout.")

    print(
        f"[INFO] Running MuJoCo sim2sim: steps={args.steps}, decimation={decimation}, "
        f"dt={model.opt.timestep:.6f}, render={args.render}"
    )
    start_root_pos = np.asarray(data.qpos[:3], dtype=np.float64).copy()

    try:
        for step in range(args.steps):
            t = min(step, motion["joint_pos"].shape[0] - 1)
            obs = build_obs(data, motion, t, meta, imu_reader, joint_qpos, joint_qvel, last_action, prop_history)
            validate_inputs(obs, input_names, meta)
            raw_action = policy.run(obs)
            target = action_to_target(raw_action, action_scale, action_offset)
            last_action = raw_action
            root_quat, _, _ = imu_reader.read(data)
            root_delta_w = np.asarray(data.qpos[:3], dtype=np.float64) - start_root_pos
            root_delta_b = quat_apply_inverse(root_quat, root_delta_w[None, :])[0]
            if args.log_interval > 0 and (step == 0 or (step + 1) % args.log_interval == 0 or step == args.steps - 1):
                target_delta = target - action_offset
                print(
                    f"[INFO] step {step + 1:>5}/{args.steps}: "
                    f"root_delta_w={root_delta_w.tolist()}, "
                    f"root_delta_b={root_delta_b.tolist()}, "
                    f"raw_action_norm={np.linalg.norm(raw_action):.4f}, "
                    f"target_delta_norm={np.linalg.norm(target_delta):.4f}, "
                    f"qpos_target_err={np.linalg.norm(target - data.qpos[joint_qpos]):.4f}, "
                    f"obs_prop_norm={np.linalg.norm(obs['prop']):.4f}"
                )
            if viewer is not None and reference_player is not None and step % reference_update_interval == 0:
                reference_player.draw(viewer, t)

            time.sleep(decimation * model.opt.timestep)
            if viewer is not None:
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

        print(
            f"[INFO] sim2sim completed: policy_steps={args.steps}, "
            f"sim_time={args.steps * decimation * model.opt.timestep:.3f}s"
        )
    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
