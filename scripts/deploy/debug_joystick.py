#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import os
import time

from sim2sim_g1.velocity_command import LinuxJoystickVelocityCommand


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug Linux /dev/input/js* joystick axis values.")
    parser.add_argument("--device", default="/dev/input/js0")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--axis_vx", type=int, default=1)
    parser.add_argument("--axis_vy", type=int, default=0)
    parser.add_argument("--axis_yaw", type=int, default=3)
    parser.add_argument("--scale_vx", type=float, default=0.8)
    parser.add_argument("--scale_vy", type=float, default=0.4)
    parser.add_argument("--scale_yaw", type=float, default=0.8)
    parser.add_argument("--limit_vx", type=float, default=1.5)
    parser.add_argument("--limit_vy", type=float, default=1.0)
    parser.add_argument("--limit_yaw", type=float, default=1.5)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--invert_vx", type=float, default=-1.0)
    parser.add_argument("--invert_vy", type=float, default=-1.0)
    parser.add_argument("--invert_yaw", type=float, default=-1.0)
    parser.add_argument("--show_all_axes", action="store_true", default=True)
    parser.add_argument("--no_show_all_axes", action="store_false", dest="show_all_axes")
    return parser.parse_args()


def _bar(value: float, width: int = 20) -> str:
    value = float(np.clip(value, -1.0, 1.0))
    center = width
    pos = int(round(center + value * width))
    chars = [" "] * (2 * width + 1)
    chars[center] = "|"
    chars[pos] = "*"
    return "".join(chars)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.device):
        raise FileNotFoundError(f"Joystick device not found: {args.device}")

    joystick = LinuxJoystickVelocityCommand(
        device=args.device,
        axis_vx=args.axis_vx,
        axis_vy=args.axis_vy,
        axis_yaw=args.axis_yaw,
        scale=np.asarray([args.scale_vx, args.scale_vy, args.scale_yaw], dtype=np.float32),
        limit=np.asarray([args.limit_vx, args.limit_vy, args.limit_yaw], dtype=np.float32),
        deadzone=args.deadzone,
        invert=np.asarray([args.invert_vx, args.invert_vy, args.invert_yaw], dtype=np.float32),
    )
    joystick.print_config()
    print("[INFO] Move sticks now. Ctrl-C to exit.")
    print("[INFO] axis bars are normalized raw values in [-1, 1]. cmd is [vx, vy, yaw_rate].")

    period = 1.0 / max(float(args.rate), 1.0)
    end = time.time() + float(args.duration)
    try:
        while time.time() < end:
            cmd = joystick.value().reshape(-1)
            axes = joystick.axes.copy()
            pieces = [
                f"cmd=[{cmd[0]: .3f}, {cmd[1]: .3f}, {cmd[2]: .3f}]",
                (
                    f"mapped_axes=(vx:{args.axis_vx}={axes[args.axis_vx]: .3f}, "
                    f"vy:{args.axis_vy}={axes[args.axis_vy]: .3f}, yaw:{args.axis_yaw}={axes[args.axis_yaw]: .3f})"
                ),
            ]
            print(" ".join(pieces))
            if args.show_all_axes:
                for axis_id, value in enumerate(axes[:8]):
                    print(f"  axis{axis_id}: {value: .3f} [{_bar(float(value))}]")
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        joystick.close()


if __name__ == "__main__":
    main()
