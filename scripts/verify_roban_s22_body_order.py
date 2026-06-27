#!/usr/bin/env python3
"""Verify RobanS22 runtime body order against attach_npz_names_roban.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Verify RobanS22 IsaacLab runtime joint/body name order.")
parser.add_argument(
    "--asset_path",
    default=None,
    help="Optional RobanS22 URDF path. Defaults to this repo's biped_s17_hands.urdf.",
)
parser.add_argument(
    "--save_json",
    default=None,
    help="Optional path to save runtime joint_names/body_names as JSON.",
)
parser.add_argument(
    "--print_only",
    action="store_true",
    help="Only print runtime order, do not fail on mismatch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACH_SCRIPT = REPO_ROOT / "scripts" / "attach_npz_names_roban.py"
DEFAULT_ASSET_PATH = (
    REPO_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "roban_s22"
    / "urdf"
    / "biped_s17_hands.urdf"
)

spec = importlib.util.spec_from_file_location("attach_npz_names_roban", ATTACH_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {ATTACH_SCRIPT}")
attach_npz_names_roban = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attach_npz_names_roban)
ROBAN_S22_FULL_BODY_NAMES = attach_npz_names_roban.ROBAN_S22_FULL_BODY_NAMES
ROBAN_S22_JOINT_NAMES = attach_npz_names_roban.ROBAN_S22_JOINT_NAMES

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


def _make_roban_cfg(asset_path: Path) -> ArticulationCfg:
    """Create a minimal Roban articulation config for reading runtime names."""
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            asset_path=str(asset_path),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "motor": ImplicitActuatorCfg(
                joint_names_expr=["waist_yaw_joint", "leg_.*", "zarm_.*"],
                stiffness=0.0,
                damping=0.0,
            )
        },
    )


@configclass
class VerifySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = _make_roban_cfg(Path(args_cli.asset_path).expanduser() if args_cli.asset_path else DEFAULT_ASSET_PATH).replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def _print_order(title: str, names: list[str]) -> None:
    print(f"\n{title} ({len(names)}):")
    for index, name in enumerate(names):
        print(f"{index:02d}: {name}")


def _print_mismatch(kind: str, runtime: list[str], expected: list[str]) -> None:
    print(f"\n[{kind}] mismatch:")
    print(f"  runtime dim : {len(runtime)}")
    print(f"  expected dim: {len(expected)}")
    for index in range(max(len(runtime), len(expected))):
        runtime_name = runtime[index] if index < len(runtime) else "<missing>"
        expected_name = expected[index] if index < len(expected) else "<missing>"
        mark = "OK" if runtime_name == expected_name else "DIFF"
        print(f"  {index:02d}: runtime={runtime_name} | expected={expected_name} | {mark}")


def main() -> int:
    asset_path = Path(args_cli.asset_path).expanduser() if args_cli.asset_path else DEFAULT_ASSET_PATH
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    print(f"[INFO] Asset path: {asset_path}", flush=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    print("[INFO] Creating scene...", flush=True)
    scene = InteractiveScene(VerifySceneCfg(num_envs=1, env_spacing=2.0))
    print("[INFO] Resetting simulation...", flush=True)
    sim.reset()
    print("[INFO] Reading runtime names...", flush=True)

    robot = scene["robot"]
    runtime_joint_names = list(robot.joint_names)
    runtime_body_names = list(robot.body_names)

    _print_order("Runtime joint_names", runtime_joint_names)
    _print_order("Runtime body_names", runtime_body_names)

    joint_match = runtime_joint_names == list(ROBAN_S22_JOINT_NAMES)
    body_match = runtime_body_names == list(ROBAN_S22_FULL_BODY_NAMES)

    print("\nCompare with scripts/attach_npz_names_roban.py:")
    print(f"  JOINT_MATCH = {joint_match}")
    print(f"  BODY_MATCH  = {body_match}")

    if not joint_match:
        _print_mismatch("joint_names", runtime_joint_names, list(ROBAN_S22_JOINT_NAMES))
    if not body_match:
        _print_mismatch("body_names", runtime_body_names, list(ROBAN_S22_FULL_BODY_NAMES))

    if args_cli.save_json is not None:
        output_path = Path(args_cli.save_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "asset_path": str(asset_path),
                    "joint_names": runtime_joint_names,
                    "body_names": runtime_body_names,
                    "joint_match_attach_npz_names_roban": joint_match,
                    "body_match_attach_npz_names_roban": body_match,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nSaved runtime order to: {output_path}")

    return 0 if args_cli.print_only or (joint_match and body_match) else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
