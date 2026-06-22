from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whole_body_tracking.terrains.stl_terrain_generator_cfg import STLTerrainGeneratorCfg


DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL = "data/omniretarget/g1_terrain/pairs.jsonl"


@dataclass(frozen=True)
class TerrainMotionPair:
    name: str
    motion_path: str
    terrain_path: str


@dataclass(frozen=True)
class TerrainMotionPairManifest:
    pairs: tuple[TerrainMotionPair, ...]

    @classmethod
    def from_jsonl(cls, path: str | Path, *, limit: int | None = None) -> TerrainMotionPairManifest:
        manifest_path = resolve_repo_path(path)
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        pairs: list[TerrainMotionPair] = []
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_num, line in enumerate(stream, start=1):
                text = line.strip()
                if not text:
                    continue
                raw = json.loads(text)
                name = str(raw.get("name") or Path(raw["tracking_motion"]).stem)
                motion_path = str(resolve_manifest_path(raw["tracking_motion"], manifest_path))
                terrain_path = str(resolve_manifest_path(raw["terrain_stl"], manifest_path))
                _validate_pair(manifest_path, line_num, name, motion_path, terrain_path)
                pairs.append(TerrainMotionPair(name=name, motion_path=motion_path, terrain_path=terrain_path))
                if limit is not None and len(pairs) >= limit:
                    break

        if not pairs:
            raise ValueError(f"{manifest_path}: no terrain-motion pairs found")
        return cls(pairs=tuple(pairs))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(pair.name for pair in self.pairs)

    @property
    def motion_paths(self) -> tuple[str, ...]:
        return tuple(pair.motion_path for pair in self.pairs)

    @property
    def terrain_paths(self) -> tuple[str, ...]:
        return tuple(pair.terrain_path for pair in self.pairs)

    @property
    def count(self) -> int:
        return len(self.pairs)


def package_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = package_root() / candidate
    return candidate.resolve()


def resolve_manifest_path(path: str | Path, manifest_path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(manifest_path).expanduser().resolve().parent / candidate
    return candidate.resolve()


def _validate_pair(manifest_path: Path, line_num: int, name: str, motion_path: str, terrain_path: str) -> None:
    motion = Path(motion_path)
    terrain = Path(terrain_path)
    if not motion.is_file():
        raise FileNotFoundError(f"{manifest_path}:{line_num}: missing motion npz: {motion}")
    if not terrain.is_file():
        raise FileNotFoundError(f"{manifest_path}:{line_num}: missing terrain stl: {terrain}")
    if motion.stem != terrain.stem:
        raise ValueError(f"{manifest_path}:{line_num}: motion/STL stem mismatch: {motion.stem!r} != {terrain.stem!r}")
    if name != motion.stem:
        raise ValueError(f"{manifest_path}:{line_num}: pair name {name!r} does not match motion stem {motion.stem!r}")


def pair_index_from_level_type(
    terrain_level: int, terrain_type: int, *, num_rows: int, num_cols: int, pair_count: int
) -> int:
    if num_rows <= 0 or num_cols <= 0:
        raise ValueError(f"num_rows and num_cols must be positive, got {num_rows}x{num_cols}")
    if pair_count <= 0:
        raise ValueError(f"pair_count must be positive, got {pair_count}")
    index = int(terrain_level) * int(num_cols) + int(terrain_type)
    if index < 0:
        raise ValueError(f"negative terrain pair index from level={terrain_level}, type={terrain_type}")
    return index % int(pair_count)


def pair_index_from_env_id(env_id: int, *, pair_count: int) -> int:
    if pair_count <= 0:
        raise ValueError(f"pair_count must be positive, got {pair_count}")
    return int(env_id) % int(pair_count)


def build_paired_terrain_cfg(
    pairs_jsonl: str = DEFAULT_OMNIRETARGET_G1_PAIRS_JSONL,
    *,
    num_rows: int | None = 1,
    num_cols: int | None = None,
    terrain_size: tuple[float, float] = (6.0, 6.0),
    border_width: float = 0.0,
    add_ground_plane: bool = True,
    ground_thickness: float = 0.05,
    ground_z: float = 0.0,
    use_cache: bool = False,
    limit: int | None = None,
) -> STLTerrainGeneratorCfg:
    from whole_body_tracking.terrains.stl_terrain_generator_cfg import STLTerrainGeneratorCfg
    from whole_body_tracking.terrains.stl_trimesh_cfg import TrimeshPlatformCfg

    manifest = TerrainMotionPairManifest.from_jsonl(pairs_jsonl, limit=limit)
    if num_cols is None and num_rows is None:
        num_cols = int(math.ceil(math.sqrt(manifest.count)))
        num_rows = int(math.ceil(manifest.count / num_cols))
    elif num_cols is None:
        num_cols = int(math.ceil(manifest.count / int(num_rows)))
    elif num_rows is None:
        num_rows = int(math.ceil(manifest.count / int(num_cols)))

    sub_terrains = {}
    for pair in manifest.pairs:
        sub_terrains[pair.name] = TrimeshPlatformCfg(
            proportion=1.0 / manifest.count,
            stl_file_path=pair.terrain_path,
            add_ground_plane=add_ground_plane,
            ground_size=terrain_size,
            ground_thickness=ground_thickness,
            ground_z=ground_z,
        )
    return STLTerrainGeneratorCfg(
        size=terrain_size,
        border_width=border_width,
        num_rows=int(num_rows),
        num_cols=int(num_cols),
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=use_cache,
        sub_terrains=sub_terrains,
    )
