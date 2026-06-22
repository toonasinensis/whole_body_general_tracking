#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug G1 MJCF USD prim layout.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if getattr(args, "experience", "") == "":
        args.experience = str(Path(__file__).with_name("kit_app") / "isaaclab.core.headless.5_1.kit")
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext


def _load_g1_cfg(articulation_root_prim_path: str | None) -> ArticulationCfg:
    wbt_source = Path(__file__).resolve().parents[2] / "source" / "whole_body_tracking" / "whole_body_tracking"
    assets_dir = wbt_source / "assets"
    g1_path = wbt_source / "robots" / "g1.py"
    package = types.ModuleType("whole_body_tracking")
    package.__path__ = [str(wbt_source)]
    assets = types.ModuleType("whole_body_tracking.assets")
    assets.ASSET_DIR = str(assets_dir)
    sys.modules.setdefault("whole_body_tracking", package)
    sys.modules["whole_body_tracking.assets"] = assets

    spec = importlib.util.spec_from_file_location("_omniretarget_g1_cfg", g1_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    cfg = module.G1_CYLINDER_CFG
    cfg.articulation_root_prim_path = articulation_root_prim_path
    cfg.spawn = sim_utils.MjcfFileCfg(
        asset_path=str(assets_dir / "unitree_description" / "mjcf" / "g1.xml"),
        fix_base=False,
        import_sites=False,
        self_collision=True,
        force_usd_conversion=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    )
    return cfg


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    cfg = _load_g1_cfg(None)
    cfg.spawn.func("/World/Robot", cfg.spawn)
    sim.render()
    stage = get_current_stage()
    print("=== Prim tree ===", flush=True)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/Robot"):
            api = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            if api or rigid or path.count("/") <= 4:
                print(path, prim.GetTypeName(), "articulation=", api, "rigid=", rigid, flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
