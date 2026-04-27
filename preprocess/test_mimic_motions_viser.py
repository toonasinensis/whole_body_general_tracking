"""Interactive multi-motion viewer based on viser.

Displays many G1 / Roban motions simultaneously.  Each robot is rendered with
the actual URDF mesh model (default) or as a skeleton stick-figure (--skeleton).
Click a robot's base frame to see the motion file name in the GUI.

Examples
--------
# Full mesh model, 12 motions:
conda run -n my_env python preprocess/test_mimic_motions_viser.py \
  --robot_cfg g1 \
  --motion_dir /home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1/检查数据集 \
  --show_count 12 --columns 4

# Skeleton mode (faster, supports more simultaneous motions):
conda run -n my_env python preprocess/test_mimic_motions_viser.py \
  --robot_cfg g1 --skeleton \
  --motion_dir /home/thl/wt_wbc/wbc_parkour/whole_body_tracking/data/g1/检查数据集 \
  --show_count 40 --columns 8

  python preprocess/test_mimic_motions_viser.py \
    --robot_cfg g1 \
    --motion_txt /home/thl/wt_wbc/wbc_parkour/whole_body_tracking/logs/rsl_rl/g1_flat/high_jump.txt \
    --motion_dir /home/thl/Documents/g1-mimic-npz

"""

from __future__ import annotations

import argparse
import numpy as np
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import viser
    import viser.extras
    import yourdfpy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "viser and yourdfpy are required.\nInstall: conda run -n my_env python -m pip install viser"
    ) from exc

# ---------------------------------------------------------------------------
# Paths & Robot presets
# ---------------------------------------------------------------------------

_ASSET_DIR = Path(__file__).parent.parent / "source/whole_body_tracking/whole_body_tracking/assets"

G1_URDF_PATH = _ASSET_DIR / "unitree_description/urdf/g1/main.urdf"
G1_MOTION_FILE = str(Path(__file__).parent.parent / "data/g1/EVAL")

G1_ANCHOR_BODY_NAME = "pelvis"
G1_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
# Actual IsaacLab joint order (from robot.joint_names / actuator groups)
G1_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

G1_SKELETON_EDGES = [
    ("pelvis", "torso_link"),
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]

ROBAN_URDF_PATH = _ASSET_DIR / "roban_s22/urdf/biped_s17_hands.urdf"
ROBAN_MOTION_FILE = "data/roban_motions/210531"
ROBAN_ANCHOR_BODY_NAME = "waist_yaw_link"
ROBAN_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "leg_l2_link",
    "leg_l4_link",
    "leg_l6_link",
    "leg_r2_link",
    "leg_r4_link",
    "leg_r6_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_l5_link",
    "zarm_r2_link",
    "zarm_r4_link",
    "zarm_r5_link",
]
ROBAN_SKELETON_EDGES = [
    ("base_link", "waist_yaw_link"),
    ("waist_yaw_link", "leg_l2_link"),
    ("leg_l2_link", "leg_l4_link"),
    ("leg_l4_link", "leg_l6_link"),
    ("waist_yaw_link", "leg_r2_link"),
    ("leg_r2_link", "leg_r4_link"),
    ("leg_r4_link", "leg_r6_link"),
    ("waist_yaw_link", "zarm_l2_link"),
    ("zarm_l2_link", "zarm_l4_link"),
    ("zarm_l4_link", "zarm_l5_link"),
    ("waist_yaw_link", "zarm_r2_link"),
    ("zarm_r2_link", "zarm_r4_link"),
    ("zarm_r4_link", "zarm_r5_link"),
]

ROBOT_PRESETS = {
    "g1": {
        "anchor_body_name": G1_ANCHOR_BODY_NAME,
        "body_names": G1_BODY_NAMES,
        "default_motion_file": G1_MOTION_FILE,
        "skeleton_edges": G1_SKELETON_EDGES,
        "urdf_path": G1_URDF_PATH,
        "isaac_joint_names": G1_ISAAC_JOINT_NAMES,
    },
    "roban": {
        "anchor_body_name": ROBAN_ANCHOR_BODY_NAME,
        "body_names": ROBAN_BODY_NAMES,
        "default_motion_file": ROBAN_MOTION_FILE,
        "skeleton_edges": ROBAN_SKELETON_EDGES,
        "urdf_path": ROBAN_URDF_PATH,
        "isaac_joint_names": None,  # fill in if Roban joint order is known
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MotionData:
    path: Path
    npz: np.lib.npyio.NpzFile
    body_pos: np.ndarray  # [T, num_bodies, 3]  float32
    body_quat: np.ndarray  # [T, num_bodies, 4]  float32  (w, x, y, z)
    joint_pos: np.ndarray  # [T, num_joints]      float32
    fps: float
    anchor_origin: np.ndarray  # body_pos[0, anchor_idx]  used to zero-out initial position


@dataclass
class MotionViewHandles:
    # skeleton mode
    frame: object = None
    lines: object = None
    points: object = None
    label: object = None
    # urdf model mode
    urdf_base_frame: object = None
    viser_urdf: object = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Viser motion inspector – G1 / Roban with full URDF model.")
    parser.add_argument("--robot_cfg", type=str, default="g1", choices=["g1", "roban"])
    parser.add_argument("--motion_dir", type=str, default=None)
    parser.add_argument(
        "--motion_txt",
        type=str,
        nargs="+",
        default=None,
        help="One or more txt files. Each non-empty line is an .npz path relative to --motion_dir.",
    )
    parser.add_argument(
        "--show_count", type=int, default=12, help="Motions to display at once (default 12 for mesh mode)."
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell_spacing", type=float, default=2.5, help="Grid spacing in meters.")
    parser.add_argument("--skeleton", action="store_true", help="Use fast skeleton stick-figure instead of URDF mesh.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_edge_indices(body_names: list[str], edge_names: list[tuple[str, str]]) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(body_names)}
    edges_idx = [(name_to_idx[a], name_to_idx[b]) for a, b in edge_names if a in name_to_idx and b in name_to_idx]
    if not edges_idx:
        raise ValueError("No valid skeleton edges found.")
    return np.asarray(edges_idx, dtype=np.int64)


def collect_motion_paths(motion_dir: Path) -> list[Path]:
    paths = sorted(motion_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz motions found in {motion_dir}")
    return paths


def collect_motion_paths_from_txt(motion_dir: Path, txt_files: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for txt_file in txt_files:
        if not txt_file.exists():
            raise FileNotFoundError(f"motion txt not found: {txt_file}")

        with txt_file.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                p = Path(line).expanduser()
                if not p.is_absolute():
                    p = motion_dir / p
                p = p.resolve()

                if p.suffix.lower() != ".npz":
                    raise ValueError(f"Not an .npz path in {txt_file}:{line_no}: {line}")
                if not p.exists():
                    raise FileNotFoundError(f"Missing motion file listed in {txt_file}:{line_no}: {p}")

                paths.append(p)

    if not paths:
        raise ValueError("No valid .npz entries found in --motion_txt files.")
    return paths


def load_motion(path: Path, required_bodies: int, anchor_idx: int) -> MotionData:
    npz = np.load(path, allow_pickle=True, mmap_mode="r")
    for key in ("body_pos_w", "body_quat_w", "joint_pos"):
        if key not in npz:
            npz.close()
            raise KeyError(f"{key} missing in {path}")
    body_pos = np.asarray(npz["body_pos_w"][:, :required_bodies, :], dtype=np.float32)
    body_quat = np.asarray(npz["body_quat_w"][:, :required_bodies, :], dtype=np.float32)
    joint_pos = np.asarray(npz["joint_pos"], dtype=np.float32)
    fps = float(np.asarray(npz["fps"]).flat[0]) if "fps" in npz else 30.0
    return MotionData(
        path=path,
        npz=npz,
        body_pos=body_pos,
        body_quat=body_quat,
        joint_pos=joint_pos,
        fps=fps,
        anchor_origin=body_pos[0, anchor_idx].copy(),
    )


def grid_offsets(num_items: int, columns: int, spacing: float) -> np.ndarray:
    offsets = np.zeros((num_items, 3), dtype=np.float32)
    for i in range(num_items):
        offsets[i, 0] = (i % columns) * spacing
        offsets[i, 1] = -(i // columns) * spacing
    if num_items > 0:
        offsets[:, 0] -= np.mean(offsets[:, 0])
        offsets[:, 1] -= np.mean(offsets[:, 1])
    return offsets


def make_npz_to_urdf_perm(urdf_joint_names: list[str], isaac_joint_names: list[str]) -> np.ndarray:
    """
    NPZ joint_pos is stored in IsaacLab actuator-group order (isaac_joint_names).
    Returns perm such that:  urdf_cfg = npz_joint_pos[perm]
    i.e. perm[urdf_slot] = npz_slot that holds the same joint.
    """
    isaac_idx = {name: i for i, name in enumerate(isaac_joint_names)}
    missing = [n for n in urdf_joint_names if n not in isaac_idx]
    if missing:
        raise ValueError(f"URDF joints not found in isaac_joint_names: {missing}")
    perm = np.array([isaac_idx[name] for name in urdf_joint_names], dtype=np.int64)
    return perm


def seg_colors(num_edges: int, selected: bool) -> np.ndarray:
    c = np.array([1.0, 0.8, 0.15] if selected else [0.2, 0.8, 1.0], dtype=np.float32)
    return np.tile(c.reshape(1, 1, 3), (num_edges, 2, 1))


def pt_colors(num_pts: int, selected: bool) -> np.ndarray:
    c = np.array([1.0, 0.75, 0.2] if selected else [0.8, 0.9, 1.0], dtype=np.float32)
    return np.tile(c, (num_pts, 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    preset = ROBOT_PRESETS[args.robot_cfg]
    body_names: list[str] = list(preset["body_names"])
    anchor_idx: int = body_names.index(preset["anchor_body_name"])
    urdf_path: Path = Path(preset["urdf_path"])

    motion_dir = Path(args.motion_dir or preset["default_motion_file"]).expanduser().resolve()
    if args.motion_txt:
        txt_files = [Path(p).expanduser().resolve() for p in args.motion_txt]
        subset_paths = collect_motion_paths_from_txt(motion_dir, txt_files)
        all_paths = subset_paths
    else:
        all_paths = collect_motion_paths(motion_dir)
        subset_paths = all_paths[args.start_index : args.start_index + args.show_count]

    if not subset_paths:
        raise ValueError(f"No motions selected: total={len(all_paths)}, start={args.start_index}")

    label_index_offset = 0 if args.motion_txt else args.start_index

    motions: list[MotionData] = []
    for p in subset_paths:
        try:
            motions.append(load_motion(p, required_bodies=len(body_names), anchor_idx=anchor_idx))
        except Exception as exc:
            print(f"[WARN] skip {p.name}: {exc}")
    if not motions:
        raise RuntimeError("No valid motions loaded.")

    print(f"[INFO] motion_dir = {motion_dir}")
    if args.motion_txt:
        print(f"[INFO] motion_txt = {', '.join(str(p) for p in txt_files)}")
        print(f"[INFO] listed={len(subset_paths)}  loaded={len(motions)}")
    else:
        print(f"[INFO] total={len(all_paths)}  selected={len(subset_paths)}  loaded={len(motions)}")
    use_mesh = not args.skeleton
    print(f"[INFO] render_mode = {'URDF mesh' if use_mesh else 'skeleton'}")

    # ------------------------------------------------------------------
    # URDF + joint remapping (only needed for mesh mode)
    # ------------------------------------------------------------------
    npz_to_urdf_perm: np.ndarray | None = None
    if use_mesh:
        if not urdf_path.exists():
            print(f"[WARN] URDF not found at {urdf_path}. Falling back to skeleton mode.")
            use_mesh = False
        else:
            print(f"[INFO] Loading URDF: {urdf_path}")
            _probe_urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=False, build_collision_scene_graph=False)
            urdf_joint_names = list(_probe_urdf.actuated_joint_names)
            print(f"[INFO] URDF joint count = {len(urdf_joint_names)}")
            isaac_jnames: list[str] = list(preset.get("isaac_joint_names") or sorted(urdf_joint_names))
            npz_to_urdf_perm = make_npz_to_urdf_perm(urdf_joint_names, isaac_jnames)

            # ── joint-order diagnostic ────────────────────────────────────────
            print("\n[DEBUG] Joint order mapping  (NPZ=IsaacLab-actual → URDF-declaration)")
            print(f"  {'NPZ(Isaac) pos':>16}  {'joint name':45}  {'URDF pos':>9}  first-frame-value")
            sample_jpos = np.asarray(motions[0].joint_pos[0], dtype=np.float32)
            for urdf_pos, jname in enumerate(urdf_joint_names):
                npz_pos = int(npz_to_urdf_perm[urdf_pos])
                val = float(sample_jpos[npz_pos]) if npz_pos < len(sample_jpos) else float("nan")
                print(f"  {npz_pos:>16}  {jname:45}  {urdf_pos:>9}  {val:+.4f}")
            print()
            # ─────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # Viser server
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.add_grid("/ground", width=60.0, height=60.0)

    with server.gui.add_folder("viewer"):
        playing = server.gui.add_checkbox("playing", initial_value=True)
        speed = server.gui.add_slider("speed", min=0.1, max=4.0, step=0.05, initial_value=1.0)
        show_labels = server.gui.add_checkbox("show_labels", initial_value=True)
        selected_txt = server.gui.add_text("selected_motion", initial_value="(click a robot to inspect)")
        server.gui.add_text(
            "page_info",
            initial_value=(
                f"txt_mode count={len(motions)}"
                if args.motion_txt
                else f"start={args.start_index}  count={len(motions)}  total={len(all_paths)}"
            ),
        )

    offsets = grid_offsets(num_items=len(motions), columns=max(1, args.columns), spacing=args.cell_spacing)
    selected_idx: dict[str, int] = {"value": 0}

    # skeleton helpers
    edge_idx: np.ndarray | None = None
    if True:  # always pre-build for skeleton fallback labels
        edge_idx = build_edge_indices(body_names, list(preset["skeleton_edges"]))

    # ------------------------------------------------------------------
    # Build scene handles per motion
    # ------------------------------------------------------------------
    handles: list[MotionViewHandles] = []

    for i, motion in enumerate(motions):
        rel_name = (
            str(motion.path.relative_to(motion_dir)) if motion.path.is_relative_to(motion_dir) else str(motion.path)
        )
        short_name = f"[{label_index_offset + i}] {motion.path.name}"

        body0 = motion.body_pos[0]
        anchor_pos0 = body0[anchor_idx]
        grid_pos = offsets[i].copy()
        grid_pos[2] = anchor_pos0[2]  # keep natural height at t=0

        h = MotionViewHandles()

        if use_mesh:
            # Parent frame – we'll move this every tick to animate the root pose.
            h.urdf_base_frame = server.scene.add_frame(
                f"/motions/{i:03d}/base",
                position=tuple(grid_pos),
                show_axes=False,
            )
            urdf_obj = viser.extras.ViserUrdf(
                server,
                urdf_path,
                root_node_name=f"/motions/{i:03d}/base",
            )
            # Apply initial joint config
            if motion.joint_pos.shape[0] > 0 and npz_to_urdf_perm is not None:
                urdf_obj.update_cfg(motion.joint_pos[0][npz_to_urdf_perm])
            h.viser_urdf = urdf_obj

            # Clickable proxy frame at pelvis height so user can select
            click_frame = server.scene.add_frame(
                f"/motions/{i:03d}/click_proxy",
                position=tuple(grid_pos),
                show_axes=False,
            )

            def _make_click_cb(robot_idx: int, name: str, _cf=click_frame, _bt=selected_txt):
                def _cb(_event):
                    selected_idx["value"] = robot_idx
                    _bt.value = name

                return _cb

            click_frame.on_click(_make_click_cb(i, rel_name))
            h.frame = click_frame
        else:
            # --- Skeleton mode ---
            body = body0.copy()
            body[:, :2] += offsets[i, :2]
            seg = body[edge_idx]

            h.frame = server.scene.add_frame(
                f"/motions/{i:03d}/frame",
                position=tuple(body[anchor_idx]),
                show_axes=False,
            )
            h.lines = server.scene.add_line_segments(
                f"/motions/{i:03d}/lines",
                points=seg,
                colors=seg_colors(edge_idx.shape[0], i == 0),
                line_width=2.0,
            )
            h.points = server.scene.add_point_cloud(
                f"/motions/{i:03d}/points",
                points=body,
                colors=pt_colors(body.shape[0], i == 0),
                point_size=0.03,
            )

            def _make_click_cb(robot_idx: int, name: str, _sf=h.frame):
                def _cb(_event):
                    selected_idx["value"] = robot_idx
                    selected_txt.value = name

                return _cb

            h.frame.on_click(_make_click_cb(i, rel_name))

        # Label above the robot (both modes)
        label = server.scene.add_label(f"/motions/{i:03d}/label", text=short_name)
        label.position = tuple(grid_pos + np.array([0, 0, 1.2], dtype=np.float32))
        h.label = label

        handles.append(h)

    selected_txt.value = f"[{label_index_offset}] {motions[0].path.name}"

    # ------------------------------------------------------------------
    # Animation loop
    # ------------------------------------------------------------------
    frame_cursor = np.zeros(len(motions), dtype=np.float64)
    t_prev = time.time()

    print(f"[INFO] viser running at http://localhost:{args.port}")
    print("[INFO] Press Ctrl+C to quit.")

    try:  # noqa: SIM1
        while True:
            t_now = time.time()
            dt = max(1e-4, t_now - t_prev)
            t_prev = t_now

            if playing.value:
                for i, motion in enumerate(motions):
                    frame_cursor[i] += float(speed.value) * motion.fps * dt

            for i, motion in enumerate(motions):
                T = int(motion.body_pos.shape[0])
                if T == 0:
                    continue
                t = int(frame_cursor[i]) % T
                body = motion.body_pos[t]
                anchor_pos = body[anchor_idx]
                is_sel = i == selected_idx["value"]
                h = handles[i]

                if use_mesh:
                    # root pose = grid_offset + (anchor_pos – anchor at t=0)
                    root_pos = offsets[i].copy()
                    root_pos[:2] += (anchor_pos - motion.anchor_origin)[:2]
                    root_pos[2] = anchor_pos[2]

                    # URDF quaternion: body_quat_w is (w, x, y, z) as stored by Isaac
                    quat_wxyz = motion.body_quat[t, anchor_idx]  # [w, x, y, z]

                    h.urdf_base_frame.position = tuple(root_pos.tolist())
                    h.urdf_base_frame.wxyz = tuple(quat_wxyz.tolist())
                    h.frame.position = tuple((root_pos + np.array([0, 0, 0], dtype=np.float32)).tolist())

                    if npz_to_urdf_perm is not None:
                        h.viser_urdf.update_cfg(motion.joint_pos[t][npz_to_urdf_perm])
                else:
                    body_w = body.copy()
                    body_w[:, :2] += offsets[i, :2]
                    seg = body_w[edge_idx]
                    h.frame.position = tuple(body_w[anchor_idx].tolist())
                    h.lines.points = seg
                    h.lines.colors = seg_colors(edge_idx.shape[0], is_sel)
                    h.points.points = body_w
                    h.points.colors = pt_colors(body_w.shape[0], is_sel)

                # Keep label above the robot's head
                label_pos = offsets[i].copy()
                label_pos[:2] += (anchor_pos - motion.anchor_origin)[:2]
                label_pos[2] = anchor_pos[2] + 1.2
                h.label.position = tuple(label_pos.tolist())
                h.label.visible = bool(show_labels.value)

            time.sleep(1.0 / 60.0)

    except KeyboardInterrupt:
        pass
    finally:
        for m in motions:
            m.npz.close()
        server.stop()


if __name__ == "__main__":
    main()
