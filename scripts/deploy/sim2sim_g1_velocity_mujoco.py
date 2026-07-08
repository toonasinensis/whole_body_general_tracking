from __future__ import annotations

import argparse
import numpy as np
import time
from pathlib import Path

from sim2sim_g1.math_utils import as_vector, quat_apply_inverse
from sim2sim_g1.mujoco_robot import (
    G1_MJCF,
    action_to_target,
    apply_pd_control,
    gains_from_metadata,
    initialize_default_pose,
    name_to_actuator_ids,
    name_to_joint_qvel_addrs,
)
from sim2sim_g1.observations import ImuReader, TermMajorHistory, prop_terms_from_metadata
from sim2sim_g1.onnx_policy import OnnxPolicy, load_metadata, onnx_input_names
from sim2sim_g1.terrain import MujocoHeightScanner
from sim2sim_g1.velocity_command import (
    KeyboardVelocityCommand,
    LinuxJoystickVelocityCommand,
    VelocityArrowConfig,
    VelocityArrowVisualizer,
    robot_velocity_yaw_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean velocity-command-only G1 ONNX runner in MuJoCo.")
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--xml_path", default=str(G1_MJCF))
    parser.add_argument("--steps", type=int, default=2000, help="Number of policy steps to simulate.")
    parser.add_argument("--dry_run", action="store_true", help="Build one observation and run one ONNX forward.")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--spawn_height_offset", type=float, default=0.05)
    parser.add_argument("--cmd_vx", type=float, default=0.4)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_yaw", type=float, default=0.0)
    parser.add_argument("--keyboard", action="store_true", help="Use numpad 8/5, 4/6, 7/9, 0 for velocity command.")
    parser.add_argument("--keyboard_vx_step", type=float, default=0.1)
    parser.add_argument("--keyboard_vy_step", type=float, default=0.1)
    parser.add_argument("--keyboard_yaw_step", type=float, default=0.1)
    parser.add_argument("--joystick", action="store_true", help="Use a Linux joystick device for velocity command.")
    parser.add_argument("--joystick_device", default="/dev/input/js0")
    parser.add_argument("--joystick_axis_vx", type=int, default=1)
    parser.add_argument("--joystick_axis_vy", type=int, default=0)
    parser.add_argument("--joystick_axis_yaw", type=int, default=3)
    parser.add_argument("--joystick_deadzone", type=float, default=0.08)
    parser.add_argument("--joystick_scale_vx", type=float, default=0.8)
    parser.add_argument("--joystick_scale_vy", type=float, default=0.4)
    parser.add_argument("--joystick_scale_yaw", type=float, default=2.0)
    parser.add_argument("--joystick_invert_vx", type=float, default=-1.0)
    parser.add_argument("--joystick_invert_vy", type=float, default=-1.0)
    parser.add_argument("--joystick_invert_yaw", type=float, default=-1.0)
    parser.add_argument("--debug_joystick", action="store_true", help="Print raw joystick axes and mapped command.")
    parser.add_argument("--debug_joystick_interval", type=int, default=10)
    parser.add_argument("--cmd_vx_limit", type=float, default=1.5)
    parser.add_argument("--cmd_vy_limit", type=float, default=1.0)
    parser.add_argument("--cmd_yaw_limit", type=float, default=1.5)
    parser.add_argument("--terrain_mode", choices=["zeros", "flat_height", "mj_ray"], default="mj_ray")
    parser.add_argument("--terrain_flat_value", type=float, default=0.0)
    parser.add_argument("--height_scan_body", default="torso_link")
    parser.add_argument("--height_scan_offset", type=float, nargs=3, default=(0.0, 0.0, 20.0))
    parser.add_argument("--height_scan_ground_z", type=float, default=0.0)
    parser.add_argument("--height_scan_obs_offset", type=float, default=0.5)
    parser.add_argument("--height_scan_geom_groups", type=int, nargs="+", default=(0,))
    parser.add_argument("--wbc_cmd_value", type=float, default=0.0)
    parser.add_argument("--vel_task_mask_value", type=float, default=1.0)
    parser.add_argument("--aux_mask_value", type=float, default=0.0)
    parser.add_argument("--imu_quat_sensor", default="base_quat")
    parser.add_argument("--imu_gyro_sensor", default="base_gyro")
    parser.add_argument("--height_body", default="pelvis")
    parser.add_argument("--fall_height", type=float, default=0.45)
    parser.add_argument("--fail_on_fall", action="store_true")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--camera_body", default="pelvis")
    parser.add_argument("--camera_distance", type=float, default=3.0)
    parser.add_argument("--camera_azimuth", type=float, default=135.0)
    parser.add_argument("--camera_elevation", type=float, default=-18.0)
    parser.add_argument("--camera_lookat_offset", type=float, nargs=3, default=(0.0, 0.0, 0.45))
    parser.add_argument("--show_cmd_arrow", action="store_true")
    parser.add_argument("--cmd_arrow_scale", type=float, default=0.45)
    parser.add_argument("--cmd_arrow_z", type=float, default=0.08)
    parser.add_argument(
        "--no_command_sensitivity",
        action="store_true",
        help="Do not print one-step action differences caused by changing velocity command.",
    )
    return parser.parse_args()


def _shape_dim(shape: list[int] | tuple[int, ...]) -> int:
    out = 1
    for dim in shape:
        out *= int(dim)
    return int(out)


def _make_static_obs(input_names: list[str], meta: dict, args: argparse.Namespace) -> dict[str, np.ndarray]:
    shapes = meta.get("observation_shapes", {})
    obs = {}
    if "wbc_cmd" in input_names:
        obs["wbc_cmd"] = np.full((1, _shape_dim(shapes["wbc_cmd"])), args.wbc_cmd_value, dtype=np.float32)
    if "vel_task_mask" in input_names:
        obs["vel_task_mask"] = np.full(
            (1, _shape_dim(shapes["vel_task_mask"])), args.vel_task_mask_value, dtype=np.float32
        )
    if "aux_mask" in input_names:
        obs["aux_mask"] = np.full((1, _shape_dim(shapes["aux_mask"])), args.aux_mask_value, dtype=np.float32)
    if "terrain" in input_names and args.terrain_mode != "mj_ray":
        value = args.terrain_flat_value if args.terrain_mode == "flat_height" else 0.0
        obs["terrain"] = np.full((1, _shape_dim(shapes["terrain"])), value, dtype=np.float32)
    return obs


def _build_prop(
    data,
    meta: dict,
    imu_reader: ImuReader,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory,
) -> np.ndarray:
    _, root_ang_vel_b, gravity_b = imu_reader.read(data)
    default_joint_pos = as_vector(meta, "default_joint_pos", len(joint_qpos), 0.0)
    default_joint_vel = as_vector(meta, "default_joint_vel", len(joint_qvel), 0.0)
    prop_terms = {
        "projected_gravity": gravity_b,
        "base_ang_vel": root_ang_vel_b,
        "joint_pos": data.qpos[joint_qpos][None, :] - default_joint_pos[None, :],
        "joint_vel": data.qvel[joint_qvel][None, :] - default_joint_vel[None, :],
        "actions": last_action,
    }
    return prop_history.update(prop_terms)


def _build_obs(
    data,
    input_names: list[str],
    meta: dict,
    args: argparse.Namespace,
    imu_reader: ImuReader,
    joint_qpos: np.ndarray,
    joint_qvel: np.ndarray,
    last_action: np.ndarray,
    prop_history: TermMajorHistory,
    command: np.ndarray,
    static_obs: dict[str, np.ndarray],
    terrain_scanner: MujocoHeightScanner | None,
) -> dict[str, np.ndarray]:
    obs = dict(static_obs)
    obs["prop"] = _build_prop(data, meta, imu_reader, joint_qpos, joint_qvel, last_action, prop_history)
    if "velcommand" in input_names:
        obs["velcommand"] = command.reshape(1, 3).astype(np.float32)
    if "base_velocity_cmd" in input_names:
        obs["base_velocity_cmd"] = command.reshape(1, 3).astype(np.float32)
    if "terrain" in input_names and "terrain" not in obs:
        if terrain_scanner is None:
            raise ValueError("ONNX has terrain input but terrain scanner is disabled.")
        terrain = terrain_scanner.scan(data)
        expected_dim = _shape_dim(meta["observation_shapes"]["terrain"])
        if terrain.shape[1] != expected_dim:
            raise ValueError(f"terrain dim {terrain.shape[1]} does not match ONNX terrain dim {expected_dim}.")
        obs["terrain"] = terrain
    missing = [name for name in input_names if name not in obs]
    if missing:
        raise ValueError(f"Velocity runner cannot build ONNX inputs {missing}; inputs are {input_names}.")
    return {name: obs[name] for name in input_names}


def _update_camera(viewer, model, data, args: argparse.Namespace) -> None:
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


def _print_command_sensitivity(policy: OnnxPolicy, obs: dict[str, np.ndarray], raw_action: np.ndarray) -> None:
    command_key = "velcommand" if "velcommand" in obs else "base_velocity_cmd" if "base_velocity_cmd" in obs else None
    if command_key is None:
        return
    base = np.asarray(raw_action, dtype=np.float32)
    base_cmd = np.asarray(obs[command_key], dtype=np.float32).copy()
    tests = (
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.4, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.8, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0, 0.5]], dtype=np.float32),
    )
    print("[INFO] velocity command sensitivity:")
    for cmd in tests:
        test_obs = {name: value.copy() for name, value in obs.items()}
        test_obs[command_key] = cmd
        action = policy.run(test_obs)
        print(
            f"[INFO]   cmd={cmd.reshape(-1).tolist()} "
            f"delta_norm={float(np.linalg.norm(action - base)):.6f} "
            f"action_norm={float(np.linalg.norm(action)):.6f}"
        )
    obs[command_key] = base_cmd


def main() -> None:
    args = parse_args()

    import mujoco

    meta = load_metadata(args.onnx_path)
    input_names = onnx_input_names(args.onnx_path)
    if "prop" not in input_names:
        raise ValueError(f"Velocity runner requires ONNX input 'prop', got {input_names}.")
    if "velcommand" not in input_names and "base_velocity_cmd" not in input_names:
        raise ValueError(f"Velocity runner requires 'velcommand' or 'base_velocity_cmd', got {input_names}.")

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml_path).expanduser()))
    data = mujoco.MjData(model)

    joint_names = meta.get("action_joint_names") or meta.get("joint_names")
    if not joint_names:
        raise ValueError("ONNX metadata must provide action_joint_names or joint_names.")
    joint_qpos, joint_qvel = name_to_joint_qvel_addrs(model, joint_names)
    actuator_ids = name_to_actuator_ids(model, joint_names)
    decimation = int(args.decimation or meta.get("decimation", 4))
    action_scale = as_vector(meta, "action_scale", len(joint_names), 0.25)
    action_offset = as_vector(meta, "action_offset", len(joint_names), 0.0)
    kp, kd = gains_from_metadata(meta, len(joint_names), args.kp, args.kd)
    torque_limits = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    if not np.any(model.actuator_ctrllimited[actuator_ids]):
        torque_limits = np.full((len(joint_names), 2), [-300.0, 300.0], dtype=np.float64)

    initialize_default_pose(data, meta, joint_names, joint_qpos)
    data.qpos[2] += float(args.spawn_height_offset)
    mujoco.mj_forward(model, data)

    imu_reader = ImuReader(model, args.imu_quat_sensor, args.imu_gyro_sensor)
    prop_history = TermMajorHistory(prop_terms_from_metadata(meta, len(joint_names)))
    last_action = np.zeros((1, len(joint_names)), dtype=np.float32)
    static_obs = _make_static_obs(input_names, meta, args)
    terrain_scanner = None
    if "terrain" in input_names and args.terrain_mode == "mj_ray":
        terrain_scanner = MujocoHeightScanner(
            model,
            body_name=args.height_scan_body,
            offset=tuple(float(v) for v in args.height_scan_offset),
            ground_z=float(args.height_scan_ground_z),
            height_offset=float(args.height_scan_obs_offset),
            geom_groups=tuple(int(v) for v in args.height_scan_geom_groups),
        )
        terrain_scanner.print_config()
    keyboard_command = KeyboardVelocityCommand(
        initial=np.asarray([args.cmd_vx, args.cmd_vy, args.cmd_yaw], dtype=np.float32),
        step=np.asarray([args.keyboard_vx_step, args.keyboard_vy_step, args.keyboard_yaw_step], dtype=np.float32),
        limit=np.asarray([args.cmd_vx_limit, args.cmd_vy_limit, args.cmd_yaw_limit], dtype=np.float32),
    )
    joystick_command = None
    if args.joystick:
        try:
            joystick_command = LinuxJoystickVelocityCommand(
                device=args.joystick_device,
                axis_vx=args.joystick_axis_vx,
                axis_vy=args.joystick_axis_vy,
                axis_yaw=args.joystick_axis_yaw,
                scale=np.asarray(
                    [args.joystick_scale_vx, args.joystick_scale_vy, args.joystick_scale_yaw], dtype=np.float32
                ),
                limit=np.asarray([args.cmd_vx_limit, args.cmd_vy_limit, args.cmd_yaw_limit], dtype=np.float32),
                deadzone=args.joystick_deadzone,
                invert=np.asarray(
                    [args.joystick_invert_vx, args.joystick_invert_vy, args.joystick_invert_yaw], dtype=np.float32
                ),
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not open joystick device {args.joystick_device!r}. "
                "Check `ls /dev/input/js*`, permissions, or run with JOYSTICK=0."
            ) from exc
        joystick_command.print_config()
    arrow_visualizer = VelocityArrowVisualizer(
        VelocityArrowConfig(
            body_name=args.camera_body,
            z=args.cmd_arrow_z,
            scale=args.cmd_arrow_scale,
        )
    )
    policy = OnnxPolicy(args.onnx_path)

    print("[INFO] Clean velocity MuJoCo runner")
    print(f"[INFO] ONNX: {args.onnx_path}")
    print(f"[INFO] inputs: {input_names}")
    print(f"[INFO] joints={len(joint_names)} decimation={decimation} timestep={model.opt.timestep}")
    print(f"[INFO] command source: {'joystick' if joystick_command is not None else 'keyboard/static'}")
    initial_command = joystick_command.value() if joystick_command is not None else keyboard_command.value()
    print(f"[INFO] initial command {initial_command.reshape(-1).tolist()}")
    print(f"[INFO] masks: vel_task_mask={args.vel_task_mask_value}, aux_mask={args.aux_mask_value}")

    obs = _build_obs(
        data,
        input_names,
        meta,
        args,
        imu_reader,
        joint_qpos,
        joint_qvel,
        last_action,
        prop_history,
        initial_command,
        static_obs,
        terrain_scanner,
    )
    raw_action = policy.run(obs)
    if raw_action.shape[-1] != len(joint_names):
        raise ValueError(f"ONNX action dim {raw_action.shape[-1]} != joint/action dim {len(joint_names)}.")
    target = action_to_target(raw_action, action_scale, action_offset)
    print(
        f"[INFO] first action: shape={raw_action.shape}, min={float(raw_action.min()):.4f}, "
        f"max={float(raw_action.max()):.4f}, norm={float(np.linalg.norm(raw_action)):.4f}"
    )
    if not args.no_command_sensitivity:
        _print_command_sensitivity(policy, obs, raw_action)
    if args.dry_run:
        return

    viewer_cm = None
    viewer = None
    if args.render:
        import mujoco.viewer

        key_callback = keyboard_command.on_key if args.keyboard else None
        viewer_cm = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        viewer = viewer_cm.__enter__()

    height_body_id = int(model.body(args.height_body).id)
    fell = False
    min_height = float("inf")
    try:
        for step in range(int(args.steps)):
            command = joystick_command.value() if joystick_command is not None else keyboard_command.value()
            if (
                args.debug_joystick
                and joystick_command is not None
                and args.debug_joystick_interval > 0
                and step % args.debug_joystick_interval == 0
            ):
                axes = joystick_command.axes.copy()
                mapped = {
                    "vx_axis": (args.joystick_axis_vx, float(axes[args.joystick_axis_vx])),
                    "vy_axis": (args.joystick_axis_vy, float(axes[args.joystick_axis_vy])),
                    "yaw_axis": (args.joystick_axis_yaw, float(axes[args.joystick_axis_yaw])),
                }
                print(
                    f"[JOYDBG] step={step} axes0_7={[round(float(v), 3) for v in axes[:8]]} "
                    f"mapped={mapped} cmd={command.reshape(-1).tolist()}"
                )
            obs = _build_obs(
                data,
                input_names,
                meta,
                args,
                imu_reader,
                joint_qpos,
                joint_qvel,
                last_action,
                prop_history,
                command,
                static_obs,
                terrain_scanner,
            )
            raw_action = policy.run(obs)
            target = action_to_target(raw_action, action_scale, action_offset)
            last_action = raw_action
            for _ in range(decimation):
                apply_pd_control(data, actuator_ids, joint_qpos, joint_qvel, target, kp, kd, torque_limits)
                mujoco.mj_step(model, data)

            height = float(data.xpos[height_body_id, 2])
            # min_height = min(min_height, height)
            # if height < float(args.fall_height):
            #     fell = True
            if args.log_interval > 0 and (step % args.log_interval == 0 or step == args.steps - 1):
                actual_vel_yaw = robot_velocity_yaw_frame(model, data, args.camera_body)
                print(
                    f"[INFO] step={step + 1}/{args.steps} height={height:.3f} min_height={min_height:.3f} "
                    f"root_xy=({data.qpos[0]:.3f},{data.qpos[1]:.3f}) "
                    f"cmd={command.reshape(-1).tolist()} actual_vel_yaw={actual_vel_yaw.tolist()}"
                )
            if viewer is not None:
                # _update_camera(viewer, model, data, args)
                if args.show_cmd_arrow:
                    arrow_visualizer.draw(viewer, model, data, command)
                viewer.sync()
                time.sleep(decimation * model.opt.timestep)
            # if fell and args.fail_on_fall:
            #     raise RuntimeError(
            #         f"Robot height body '{args.height_body}' fell below {args.fall_height:.3f} m "
            #         f"at policy step {step + 1}: height={height:.3f}."
            #     )
    finally:
        if joystick_command is not None:
            joystick_command.close()
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)

    # status = "FELL" if fell else "OK"
    # print(f"[INFO] velocity sim finished: status={status}, min_height={min_height:.3f}, steps={args.steps}")


if __name__ == "__main__":
    main()
