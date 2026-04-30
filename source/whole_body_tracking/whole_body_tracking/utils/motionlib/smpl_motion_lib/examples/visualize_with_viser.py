"""Run VISER visualization for one pkl or a directory of pkl motions."""

from __future__ import annotations

import argparse
from pathlib import Path

from smpl_motion_lib import visualize_motion_directory, visualize_motion_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=False,
        help="Path to one .pkl file or a directory containing .pkl files.",
        default="/home/thl/Downloads/data/TEST_50hz",
    )
    parser.add_argument("--target-fps", type=float, default=50)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-loop", action="store_true", default=False)
    parser.add_argument("--no-ground", action="store_true", default=False)
    parser.add_argument("--no-transl", action="store_true", default=False)
    parser.add_argument("--smpl-y-up", action="store_true", default=True)
    parser.add_argument("--show-joint-frames", action="store_true", default=True)
    parser.add_argument("--frame-axis-length", type=float, default=0.08)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    common_kwargs = dict(
        target_fps=args.target_fps,
        use_transl=not args.no_transl,
        show_ground=not args.no_ground,
        show_joint_frames=args.show_joint_frames,
        frame_axis_length=args.frame_axis_length,
        host=args.host,
        port=args.port,
        loop=not args.no_loop,
        blocking=True,
    )

    # Open http://127.0.0.1:8080 in browser (or the host/port you pass).
    if input_path.is_dir():
        visualize_motion_directory(str(input_path), **common_kwargs)
    else:
        visualize_motion_file(str(input_path), **common_kwargs)


if __name__ == "__main__":
    main()
