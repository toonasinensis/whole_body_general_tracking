"""Split motion NPZ into segments (with optional overlap) and/or generate Motion_Dataset info.yaml.

Two subcommands:

  1. split   Split one NPZ into fixed-duration segments; optional overlap via --overlap-ratio.
  2. info    Generate info.yaml from NPZ files in a dataset directory (same as make_motion_info_yaml).

Usage examples:
  # Split with 50% overlap between consecutive segments
  python scripts/split_motion_npz.py split \
      --input assets/roban_motions/newdance_01_Skeleton.npz \
      --output-dir assets/roban_motions_bins_50 \
      --segment-seconds 4 \
      --overlap-ratio 0.9
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np


def get_fps(data) -> float:
    """Extract fps as a float from NPZ."""
    if "fps" not in data.files:
        raise ValueError("Input NPZ does not contain 'fps' key.")
    fps_arr = np.asarray(data["fps"]).reshape(-1)
    if fps_arr.size == 0:
        raise ValueError("'fps' array is empty.")
    return float(fps_arr[0])


def infer_total_frames(data) -> int:
    """Infer total frame count from NPZ time-series arrays."""
    time_keys_priority = [
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    for key in time_keys_priority:
        if key in data.files:
            return int(data[key].shape[0])
    for key in data.files:
        arr = data[key]
        if hasattr(arr, "shape") and len(arr.shape) >= 1:
            return int(arr.shape[0])
    raise ValueError("Could not infer total_frames from NPZ contents.")


def compute_segment_ranges(
    total_frames: int,
    frames_per_segment: int,
    overlap_ratio: float,
) -> list[tuple[int, int]]:
    """Compute (start, end) frame ranges for segments.
    
    Always drop the last partial segment
    overlap_ratio in [0, 1): fraction of segment duration that overlaps with the next.
    0 = no overlap; 0.5 = 50% overlap (step = half segment).
    """
    if overlap_ratio < 0 or overlap_ratio >= 1:
        raise ValueError("overlap_ratio must be in [0, 1).")
    step = max(1, int(round(frames_per_segment * (1.0 - overlap_ratio))))
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < total_frames:
        end = start + frames_per_segment
        if end > total_frames:
            break
        ranges.append((start, end))
        start += step
    return ranges


def run_split(args: argparse.Namespace) -> None:
    """
    Split a long motion NPZ into fixed-duration segments and write them as new NPZ files.

    For each generated segment NPZ:

    - Required keys:
        - fps: float or 1-element array, the FPS of the motion.
        - joint_pos: np.ndarray, shape (motion_length + one_sec_frames, num_joints)
        - joint_vel: np.ndarray, shape (motion_length + one_sec_frames, num_joints)
        - body_pos_w: np.ndarray, shape (motion_length, num_bodies, 3)
        - body_quat_w: np.ndarray, shape (motion_length, num_bodies, 4)
        - body_lin_vel_w: np.ndarray, shape (motion_length, num_bodies, 3)
        - body_ang_vel_w: np.ndarray, shape (motion_length, num_bodies, 3)

    Here:
        - motion_length == number of frames in the segment (end - start)
        - one_sec_frames == int(round(fps * 1.0))

    The joint_pos/joint_vel arrays are extended by one_sec_frames into the future.
    If any of those future frames would go past the end of the original motion,
    they are clamped to the last frame of the original motion (i.e., repeat the
    final frame).

    All other time-series keys (that have shape[0] == total_frames) are sliced
    to shape (motion_length, ...). Non time-series keys are copied unchanged.
    """
    ##############
    # basic info #
    ##############
    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {in_path}")

    data = np.load(in_path)
    fps = get_fps(data)
    segment_seconds = float(args.segment_seconds)
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be > 0.")

    frames_per_segment = int(round(fps * segment_seconds))
    if frames_per_segment <= 0:
        raise ValueError(
            f"Computed frames_per_segment <= 0 (fps={fps}, segment_seconds={segment_seconds})."
        )

    overlap_ratio = float(args.overlap_ratio)
    total_frames = infer_total_frames(data)

    ############################
    # Determine segment ranges #
    ############################
    ranges = compute_segment_ranges(
        total_frames=total_frames,
        frames_per_segment=frames_per_segment,
        overlap_ratio=overlap_ratio,
    )

    #########################################
    # Prepare one-second horizon for joints #
    #########################################
    if "joint_pos" not in data.files or "joint_vel" not in data.files:
        raise ValueError("Input NPZ must contain 'joint_pos' and 'joint_vel'.")
    joint_pos_full = np.asarray(data["joint_pos"])
    joint_vel_full = np.asarray(data["joint_vel"])
    if joint_pos_full.shape != joint_vel_full.shape:
        raise ValueError(
            f"'joint_pos' and 'joint_vel' must have the same shape, "
            f"got {joint_pos_full.shape} vs {joint_vel_full.shape}."
        )
    if joint_pos_full.shape[0] != total_frames:
        raise ValueError(
            f"'joint_pos' first dimension ({joint_pos_full.shape[0]}) "
            f"does not match inferred total_frames ({total_frames})."
        )
    one_sec_frames = int(round(fps * 1.0))
    if one_sec_frames <= 0:
        raise ValueError(f"Computed one_sec_frames <= 0 for fps={fps}.")

    ####################################
    # Create segments and write to NPZ #
    ####################################
    for seg_idx, (start, end) in enumerate(ranges):
        segment_dict: dict[str, np.ndarray] = {}
        for key in data.files:
            arr = data[key]
            segment_dict[key] = arr[start:end]
        
        target_length = segment_dict["joint_pos"].shape[0] + one_sec_frames
        
        # extend segment_dict["joint_pos"] and segment_dict["joint_vel"] by one_sec_frames
        if end + one_sec_frames > total_frames:
            # concat the original motion frames
            extended_pos = np.concatenate([segment_dict["joint_pos"], joint_pos_full[end:total_frames]], axis=0)
            extended_vel = np.concatenate([segment_dict["joint_vel"], joint_vel_full[end:total_frames]], axis=0)
            remaining = target_length - extended_pos.shape[0]
            if remaining > 0:
                # Pad with the last frame (repeat the final frame)
                last_pos_frame = joint_pos_full[total_frames-1:total_frames]
                last_vel_frame = joint_vel_full[total_frames-1:total_frames]
                
                pad_pos = np.repeat(last_pos_frame, remaining, axis=0)
                pad_vel = np.repeat(last_vel_frame, remaining, axis=0)
                
                segment_dict["joint_pos"] = np.concatenate([extended_pos, pad_pos], axis=0)
                segment_dict["joint_vel"] = np.concatenate([extended_vel, pad_vel], axis=0)
            else:
                segment_dict["joint_pos"] = extended_pos
                segment_dict["joint_vel"] = extended_vel
        else:
            segment_dict["joint_pos"] = np.concatenate([segment_dict["joint_pos"], joint_pos_full[end:end+one_sec_frames]], axis=0)
            segment_dict["joint_vel"] = np.concatenate([segment_dict["joint_vel"], joint_vel_full[end:end+one_sec_frames]], axis=0)

        segment_dict["fps"] = np.array([fps])
        out_path = out_dir / f"{in_path.stem}_seg{seg_idx:04d}.npz"
        np.savez(out_path, **segment_dict)
        print(f"Saved segment {seg_idx} to {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split motion NPZ into segments (with optional overlap) and/or generate info.yaml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand.")

    # --- split ---
    split_p = subparsers.add_parser("split", help="Split one NPZ into fixed-duration segments.")
    
    split_p.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to input motion NPZ."
    )
    split_p.add_argument(
        "--output-dir", 
        type=str, 
        required=True, 
        help="Directory to write split NPZ files."
    )
    split_p.add_argument(
        "--segment-seconds",
        type=float,
        default=1.0,
        help="Target segment duration in seconds (default: 1.0).",
    )
    split_p.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.0,
        help="Overlap between consecutive segments, in [0, 1). 0 = no overlap, 0.5 = 50%% overlap (default: 0).",
    )

    return parser.parse_args(argv)


# Default arguments used when running this script without CLI arguments.
# Adjust these paths to match your local debugging setup.
DEBUG_SPLIT_ARGS: list[str] = [
    "split",
    "--input",
    "motiondata/npz/kuavo5_new_year_dance.npz",
    "--output-dir",
    "assets/motion_bins_50",
    "--segment-seconds",
    "2.0",
    "--overlap-ratio",
    "0.9",
]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_split(args)


if __name__ == "__main__":
    # If no CLI arguments are provided, fall back to DEBUG_SPLIT_ARGS so the
    # script can be run easily from an IDE/debugger.
    if len(sys.argv) == 1:
        print(f"[split_motion_npz] No CLI args provided. Using DEBUG_SPLIT_ARGS: {DEBUG_SPLIT_ARGS}")
        main(DEBUG_SPLIT_ARGS)
    else:
        main()
