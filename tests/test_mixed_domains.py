from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAINS_ROOT = ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "terrains"

sys.modules.setdefault("whole_body_tracking", types.ModuleType("whole_body_tracking"))
sys.modules.setdefault("whole_body_tracking.terrains", types.ModuleType("whole_body_tracking.terrains"))

PM_SPEC = importlib.util.spec_from_file_location(
    "whole_body_tracking.terrains.paired_manifest", TERRAINS_ROOT / "paired_manifest.py"
)
paired_manifest = importlib.util.module_from_spec(PM_SPEC)
assert PM_SPEC is not None and PM_SPEC.loader is not None
sys.modules["whole_body_tracking.terrains.paired_manifest"] = paired_manifest
PM_SPEC.loader.exec_module(paired_manifest)

SPEC = importlib.util.spec_from_file_location(
    "whole_body_tracking.terrains.mixed_domains", TERRAINS_ROOT / "mixed_domains.py"
)
mixed_domains = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules["whole_body_tracking.terrains.mixed_domains"] = mixed_domains
SPEC.loader.exec_module(mixed_domains)

allocate_domain_env_counts = mixed_domains.allocate_domain_env_counts
domain_env_starts = mixed_domains.domain_env_starts
build_mixed_domain_layout = mixed_domains.build_mixed_domain_layout


@dataclass
class FlatCfg:
    name: str = "flat_lafan"
    env_ratio: float = 0.25
    pairing: str = "unpaired"
    terrain_kind: str = "flat"
    motion_file: str = ""
    dataset_txt: str = ""
    max_motion_num: int = -1
    terrain_cell_count: int = 2


@dataclass
class MeshCfg:
    name: str = "mesh_pairs"
    env_ratio: float = 0.75
    pairing: str = "strict_pair"
    terrain_kind: str = "stl_grid"
    pairs_jsonl: str = ""
    pair_limit: int | None = None


def _write_flat_dataset(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
    data = tmp_path / "data"
    motion_dir = data / "flat"
    motion_dir.mkdir(parents=True)
    for name in names:
        (motion_dir / f"{name}.npz").write_bytes(b"npz")
    dataset = data / "flat.txt"
    dataset.write_text("\n".join(f"flat/{name}.npz" for name in names) + "\n", encoding="utf-8")
    return data, dataset


def _write_pairs(tmp_path: Path, names: list[str]) -> Path:
    root = tmp_path / "pairs"
    motion_dir = root / "motions"
    terrain_dir = root / "terrains"
    motion_dir.mkdir(parents=True)
    terrain_dir.mkdir(parents=True)
    lines = []
    for name in names:
        (motion_dir / f"{name}.npz").write_bytes(b"npz")
        (terrain_dir / f"{name}.stl").write_bytes(b"solid terrain\nendsolid terrain\n")
        lines.append(
            json.dumps(
                {
                    "name": name,
                    "tracking_motion": f"motions/{name}.npz",
                    "terrain_stl": f"terrains/{name}.stl",
                }
            )
        )
    manifest = root / "pairs.jsonl"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_allocate_domain_env_counts_uses_ratios() -> None:
    assert allocate_domain_env_counts(32, (0.25, 0.75)) == (8, 24)
    assert allocate_domain_env_counts(10, (1.0, 1.0, 2.0)) == (3, 2, 5)


def test_mesh_domain_local_indices_start_after_flat_envs() -> None:
    counts = allocate_domain_env_counts(16, (0.25, 0.75))
    starts = domain_env_starts(counts)

    assert counts == (4, 12)
    assert starts == (0, 4)
    mesh_pair_count = 8
    mesh_env_ids = list(range(starts[1], starts[1] + mesh_pair_count))
    mesh_local_pair_ids = [(env_id - starts[1]) % mesh_pair_count for env_id in mesh_env_ids]

    assert mesh_local_pair_ids == list(range(mesh_pair_count))


def test_mixed_layout_ranges_do_not_overlap(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b", "climb_c"])

    flat_cfg = FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=2)
    mesh_cfg = MeshCfg(pairs_jsonl=str(pairs))
    layout = build_mixed_domain_layout([flat_cfg, mesh_cfg])

    flat, mesh = layout.domains
    assert flat.motion_start == 0
    assert flat.motion_count == 2
    assert mesh.motion_start == 2
    assert mesh.motion_count == 3
    assert flat.terrain_start == 0
    assert flat.terrain_count == 2
    assert mesh.terrain_start == 2
    assert mesh.terrain_count == 3
    assert set(range(flat.motion_start, flat.motion_start + flat.motion_count)).isdisjoint(
        range(mesh.motion_start, mesh.motion_start + mesh.motion_count)
    )
    assert set(range(flat.terrain_start, flat.terrain_start + flat.terrain_count)).isdisjoint(
        range(mesh.terrain_start, mesh.terrain_start + mesh.terrain_count)
    )


def test_mixed_layout_can_insert_ground_separator_cells(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk", "run"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b", "climb_c"])

    flat_cfg = FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=2)
    mesh_cfg = MeshCfg(pairs_jsonl=str(pairs))
    layout = build_mixed_domain_layout(
        [flat_cfg, mesh_cfg],
        domain_separator_cell_count=4,
        pad_to_grid=True,
    )

    flat, mesh = layout.domains
    assert flat.terrain_start == 0
    assert flat.terrain_count == 2
    assert layout.terrain_names[2:6] == tuple(f"flat_lafan_separator_ground_{index:03d}" for index in range(4))
    assert mesh.terrain_start == 6
    assert mesh.terrain_count == 3
    assert layout.terrain_names[mesh.terrain_start : mesh.terrain_start + mesh.terrain_count] == (
        "climb_a",
        "climb_b",
        "climb_c",
    )
    assert layout.terrain_count == layout.num_rows * layout.num_cols


def test_strict_pair_local_terrain_index_matches_motion_index(tmp_path: Path) -> None:
    flat_root, flat_dataset = _write_flat_dataset(tmp_path, ["walk"])
    pairs = _write_pairs(tmp_path, ["climb_a", "climb_b"])

    layout = build_mixed_domain_layout(
        [
            FlatCfg(motion_file=str(flat_root), dataset_txt=str(flat_dataset), terrain_cell_count=1),
            MeshCfg(pairs_jsonl=str(pairs)),
        ]
    )

    mesh = layout.domains[1]
    for local_index, name in enumerate(("climb_a", "climb_b")):
        motion_id = mesh.motion_start + local_index
        terrain_id = mesh.terrain_start + local_index
        assert Path(layout.motion_paths[motion_id]).stem == name
        assert layout.terrain_names[terrain_id] == name
