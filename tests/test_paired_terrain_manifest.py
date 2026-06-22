from __future__ import annotations

import importlib.util
import json
import numpy as np
import sys
import trimesh
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "terrains"
    / "paired_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("paired_manifest", MODULE_PATH)
paired_manifest = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules["paired_manifest"] = paired_manifest
SPEC.loader.exec_module(paired_manifest)

TerrainMotionPairManifest = paired_manifest.TerrainMotionPairManifest
build_paired_terrain_cfg = paired_manifest.build_paired_terrain_cfg
pair_index_from_env_id = paired_manifest.pair_index_from_env_id
pair_index_from_level_type = paired_manifest.pair_index_from_level_type


def _write_pair(tmp_path: Path, name: str) -> dict:
    motion = tmp_path / f"{name}.npz"
    terrain = tmp_path / f"{name}.stl"
    motion.write_bytes(b"npz")
    terrain.write_bytes(b"solid terrain\nendsolid terrain\n")
    return {"name": name, "tracking_motion": str(motion), "terrain_stl": str(terrain)}


def test_manifest_preserves_order_and_paths(tmp_path: Path) -> None:
    pairs = [_write_pair(tmp_path, "climb_00_z_scale_0.8"), _write_pair(tmp_path, "climb_01_z_scale_1.0")]
    manifest_path = tmp_path / "pairs.jsonl"
    manifest_path.write_text("\n".join(json.dumps(pair) for pair in pairs) + "\n", encoding="utf-8")

    manifest = TerrainMotionPairManifest.from_jsonl(manifest_path)

    assert manifest.names == ("climb_00_z_scale_0.8", "climb_01_z_scale_1.0")
    assert tuple(Path(path).stem for path in manifest.motion_paths) == manifest.names
    assert tuple(Path(path).stem for path in manifest.terrain_paths) == manifest.names


def test_manifest_resolves_relative_paths_from_manifest_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    motion_dir = data_dir / "motions"
    terrain_dir = data_dir / "terrains"
    motion_dir.mkdir()
    terrain_dir.mkdir()
    motion = motion_dir / "climb_00_z_scale_0.8.npz"
    terrain = terrain_dir / "climb_00_z_scale_0.8.stl"
    motion.write_bytes(b"npz")
    terrain.write_bytes(b"solid terrain\nendsolid terrain\n")
    manifest_path = data_dir / "pairs.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "climb_00_z_scale_0.8",
                "tracking_motion": "motions/climb_00_z_scale_0.8.npz",
                "terrain_stl": "terrains/climb_00_z_scale_0.8.stl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = TerrainMotionPairManifest.from_jsonl(manifest_path)

    assert Path(manifest.motion_paths[0]) == motion.resolve()
    assert Path(manifest.terrain_paths[0]) == terrain.resolve()


def test_manifest_rejects_mismatched_stems(tmp_path: Path) -> None:
    motion = tmp_path / "motion_a.npz"
    terrain = tmp_path / "terrain_b.stl"
    motion.write_bytes(b"npz")
    terrain.write_bytes(b"solid terrain\nendsolid terrain\n")
    manifest_path = tmp_path / "pairs.jsonl"
    manifest_path.write_text(
        json.dumps({"name": "motion_a", "tracking_motion": str(motion), "terrain_stl": str(terrain)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stem mismatch"):
        TerrainMotionPairManifest.from_jsonl(manifest_path)


def test_pair_index_from_level_type_wraps_grid_slots() -> None:
    assert pair_index_from_level_type(0, 0, num_rows=2, num_cols=3, pair_count=5) == 0
    assert pair_index_from_level_type(1, 1, num_rows=2, num_cols=3, pair_count=5) == 4
    assert pair_index_from_level_type(1, 2, num_rows=2, num_cols=3, pair_count=5) == 0


def test_pair_index_from_env_id_repeats_manifest_order() -> None:
    assert pair_index_from_env_id(0, pair_count=145) == 0
    assert pair_index_from_env_id(144, pair_count=145) == 144
    assert pair_index_from_env_id(145, pair_count=145) == 0
    assert pair_index_from_env_id(146, pair_count=145) == 1


def test_paired_terrain_cfg_adds_ground_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    @dataclass
    class DummyTrimeshPlatformCfg:
        proportion: float
        stl_file_path: str
        add_ground_plane: bool = False
        ground_size: tuple[float, float] | None = None
        ground_thickness: float = 0.05
        ground_z: float = 0.0

    @dataclass
    class DummySTLTerrainGeneratorCfg:
        size: tuple[float, float]
        border_width: float
        num_rows: int
        num_cols: int
        horizontal_scale: float
        vertical_scale: float
        slope_threshold: float
        use_cache: bool
        sub_terrains: dict

    stl_cfg_module = types.ModuleType("whole_body_tracking.terrains.stl_trimesh_cfg")
    stl_cfg_module.TrimeshPlatformCfg = DummyTrimeshPlatformCfg
    generator_cfg_module = types.ModuleType("whole_body_tracking.terrains.stl_terrain_generator_cfg")
    generator_cfg_module.STLTerrainGeneratorCfg = DummySTLTerrainGeneratorCfg
    monkeypatch.setitem(sys.modules, "whole_body_tracking", types.ModuleType("whole_body_tracking"))
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains", types.ModuleType("whole_body_tracking.terrains"))
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains.stl_trimesh_cfg", stl_cfg_module)
    monkeypatch.setitem(sys.modules, "whole_body_tracking.terrains.stl_terrain_generator_cfg", generator_cfg_module)

    motion = tmp_path / "climb_00_z_scale_0.8.npz"
    terrain = tmp_path / "climb_00_z_scale_0.8.stl"
    motion.write_bytes(b"npz")
    trimesh.creation.box(extents=(0.4, 0.4, 0.4)).export(terrain)
    manifest_path = tmp_path / "pairs.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "climb_00_z_scale_0.8",
                "tracking_motion": str(motion),
                "terrain_stl": str(terrain),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = build_paired_terrain_cfg(str(manifest_path), terrain_size=(6.0, 6.0), ground_thickness=0.05)
    sub_cfg = next(iter(cfg.sub_terrains.values()))

    assert sub_cfg.add_ground_plane is True
    assert sub_cfg.ground_size == (6.0, 6.0)
    assert np.isclose(sub_cfg.ground_thickness, 0.05)
