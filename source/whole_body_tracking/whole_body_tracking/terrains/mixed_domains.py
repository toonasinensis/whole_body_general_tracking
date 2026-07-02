from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from whole_body_tracking.terrains.paired_manifest import TerrainMotionPairManifest, resolve_repo_path

PairingMode = Literal["unpaired", "strict_pair"]
TerrainKind = Literal["flat", "stl_grid", "hf_grid"]


@dataclass(frozen=True)
class MixedDomainSpec:
    name: str
    env_ratio: float
    pairing: PairingMode
    terrain_kind: TerrainKind
    task_profile: str
    motion_start: int
    motion_count: int
    terrain_start: int
    terrain_count: int
    motion_names: tuple[str, ...]
    terrain_names: tuple[str, ...]


@dataclass(frozen=True)
class MixedDomainLayout:
    domains: tuple[MixedDomainSpec, ...]
    motion_paths: tuple[str, ...]
    terrain_paths: tuple[str | None, ...]
    terrain_names: tuple[str, ...]
    num_rows: int
    num_cols: int

    @property
    def motion_count(self) -> int:
        return len(self.motion_paths)

    @property
    def terrain_count(self) -> int:
        return len(self.terrain_names)


def read_motion_dataset(
    motion_file: str | Path,
    dataset_txt: str | Path | None,
    *,
    max_motion_num: int = -1,
) -> tuple[str, ...]:
    motion_root = resolve_repo_path(motion_file)
    if dataset_txt is None:
        paths = sorted(str(path.resolve()) for path in motion_root.rglob("*.npz"))
    else:
        dataset_path = resolve_repo_path(dataset_txt)
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        paths = []
        with dataset_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                entry = line.strip()
                if not entry:
                    continue
                candidate = Path(entry).expanduser()
                if not candidate.is_absolute():
                    candidate = motion_root / candidate
                if candidate.suffix.lower() != ".npz":
                    continue
                if not candidate.is_file():
                    raise FileNotFoundError(candidate)
                paths.append(str(candidate.resolve()))

    if max_motion_num != -1:
        paths = paths[: int(max_motion_num)]
    if not paths:
        raise FileNotFoundError(f"No motion npz files found from motion_file={motion_file}, dataset_txt={dataset_txt}")
    return tuple(paths)


def allocate_domain_env_counts(num_envs: int, ratios: tuple[float, ...]) -> tuple[int, ...]:
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}")
    if not ratios:
        raise ValueError("At least one domain ratio is required")
    total = sum(max(0.0, float(ratio)) for ratio in ratios)
    if total <= 0.0:
        raise ValueError(f"Domain ratios must sum to a positive value, got {ratios}")

    scaled = [max(0.0, float(ratio)) / total * num_envs for ratio in ratios]
    counts = [int(math.floor(value)) for value in scaled]
    remainder = num_envs - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: scaled[index] - counts[index], reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def domain_env_starts(counts: tuple[int, ...]) -> tuple[int, ...]:
    starts: list[int] = []
    cursor = 0
    for count in counts:
        if count < 0:
            raise ValueError(f"Domain env counts must be non-negative, got {counts}")
        starts.append(cursor)
        cursor += int(count)
    return tuple(starts)


def build_mixed_domain_layout(
    domain_cfgs,
    *,
    num_rows: int | None = None,
    num_cols: int | None = None,
    domain_separator_cell_count: int = 0,
    pad_to_grid: bool = False,
) -> MixedDomainLayout:
    specs: list[MixedDomainSpec] = []
    motion_paths: list[str] = []
    terrain_paths: list[str | None] = []
    terrain_names: list[str] = []
    separator_cell_count = max(0, int(domain_separator_cell_count))

    for domain_index, domain_cfg in enumerate(domain_cfgs):
        if domain_index > 0 and separator_cell_count > 0:
            previous_name = specs[-1].name if specs else "domain"
            _append_ground_cells(
                terrain_paths,
                terrain_names,
                prefix=f"{previous_name}_separator",
                count=separator_cell_count,
            )

        name = str(domain_cfg.name)
        pairing = str(domain_cfg.pairing)
        terrain_kind = str(domain_cfg.terrain_kind)
        env_ratio = float(domain_cfg.env_ratio)
        task_profile = str(getattr(domain_cfg, "task_profile", "wbc_tracking"))
        motion_start = len(motion_paths)
        terrain_start = len(terrain_names)

        if pairing == "unpaired" and terrain_kind == "flat":
            domain_motion_paths = read_motion_dataset(
                domain_cfg.motion_file,
                domain_cfg.dataset_txt,
                max_motion_num=int(getattr(domain_cfg, "max_motion_num", -1)),
            )
            cell_count = int(getattr(domain_cfg, "terrain_cell_count", 1))
            if cell_count <= 0:
                raise ValueError(f"{name}: terrain_cell_count must be positive, got {cell_count}")

            motion_names = tuple(Path(path).stem for path in domain_motion_paths)
            domain_terrain_names = tuple(f"{name}_flat_{index:03d}" for index in range(cell_count))
            motion_paths.extend(domain_motion_paths)
            terrain_paths.extend([None] * cell_count)
            terrain_names.extend(domain_terrain_names)

        elif pairing == "unpaired" and terrain_kind == "hf_grid":
            domain_motion_paths = read_motion_dataset(
                domain_cfg.motion_file,
                domain_cfg.dataset_txt,
                max_motion_num=int(getattr(domain_cfg, "max_motion_num", -1)),
            )
            cell_count = int(getattr(domain_cfg, "terrain_cell_count", 1))
            if cell_count <= 0:
                raise ValueError(f"{name}: terrain_cell_count must be positive, got {cell_count}")

            motion_names = tuple(Path(path).stem for path in domain_motion_paths)
            terrain_profile = str(getattr(domain_cfg, "terrain_profile", "long_runway"))
            domain_terrain_names = tuple(f"{name}_{terrain_profile}_{index:03d}" for index in range(cell_count))
            motion_paths.extend(domain_motion_paths)
            terrain_paths.extend([None] * cell_count)
            terrain_names.extend(domain_terrain_names)

        elif pairing == "strict_pair" and terrain_kind == "stl_grid":
            manifest = TerrainMotionPairManifest.from_jsonl(
                domain_cfg.pairs_jsonl,
                limit=getattr(domain_cfg, "pair_limit", None),
            )
            motion_names = manifest.names
            domain_terrain_names = manifest.names
            motion_paths.extend(manifest.motion_paths)
            terrain_paths.extend(manifest.terrain_paths)
            terrain_names.extend(domain_terrain_names)

        else:
            raise ValueError(f"Unsupported mixed domain {name!r}: pairing={pairing!r}, terrain_kind={terrain_kind!r}")

        specs.append(
            MixedDomainSpec(
                name=name,
                env_ratio=env_ratio,
                pairing=pairing,  # type: ignore[arg-type]
                terrain_kind=terrain_kind,  # type: ignore[arg-type]
                task_profile=task_profile,
                motion_start=motion_start,
                motion_count=len(motion_paths) - motion_start,
                terrain_start=terrain_start,
                terrain_count=len(terrain_names) - terrain_start,
                motion_names=motion_names,
                terrain_names=domain_terrain_names,
            )
        )

    if not specs:
        raise ValueError("No mixed motion domains configured")
    terrain_count = len(terrain_names)
    if terrain_count <= 0:
        raise ValueError("No terrains configured for mixed domains")

    if num_cols is None and num_rows is None:
        num_cols = int(math.ceil(math.sqrt(terrain_count)))
        num_rows = int(math.ceil(terrain_count / num_cols))
    elif num_cols is None:
        num_cols = int(math.ceil(terrain_count / int(num_rows)))
    elif num_rows is None:
        num_rows = int(math.ceil(terrain_count / int(num_cols)))

    capacity = int(num_rows) * int(num_cols)
    if capacity < len(terrain_names):
        raise ValueError(
            f"Terrain grid {num_rows}x{num_cols} has {capacity} cells, "
            f"but mixed domains require {len(terrain_names)} cells"
        )
    if pad_to_grid and capacity > len(terrain_names):
        _append_ground_cells(
            terrain_paths,
            terrain_names,
            prefix="mixed_padding",
            count=capacity - len(terrain_names),
        )

    return MixedDomainLayout(
        domains=tuple(specs),
        motion_paths=tuple(motion_paths),
        terrain_paths=tuple(terrain_paths),
        terrain_names=tuple(terrain_names),
        num_rows=int(num_rows),
        num_cols=int(num_cols),
    )


def _append_ground_cells(
    terrain_paths: list[str | None],
    terrain_names: list[str],
    *,
    prefix: str,
    count: int,
) -> None:
    for index in range(int(count)):
        terrain_paths.append(None)
        terrain_names.append(f"{prefix}_ground_{index:03d}")


def build_mixed_terrain_cfg(
    domain_cfgs,
    *,
    terrain_size: tuple[float, float] = (6.0, 6.0),
    border_width: float = 0.0,
    ground_thickness: float = 0.01,
    ground_z: float = 0.0,
    domain_separator_cell_count: int = 0,
    use_cache: bool = False,
) -> tuple[STLTerrainGeneratorCfg, MixedDomainLayout]:
    from whole_body_tracking.terrains.stl_terrain_generator_cfg import STLTerrainGeneratorCfg
    from whole_body_tracking.terrains.stl_trimesh_cfg import TrimeshPlatformCfg

    layout = build_mixed_domain_layout(
        domain_cfgs,
        domain_separator_cell_count=domain_separator_cell_count,
        pad_to_grid=True,
    )
    proportion = 1.0 / float(layout.terrain_count)
    sub_terrains = {}
    hf_names = _hf_terrain_name_map(domain_cfgs)
    for name, terrain_path in zip(layout.terrain_names, layout.terrain_paths, strict=True):
        if name in hf_names:
            sub_terrains[name] = _make_hf_terrain_cfg(hf_names[name], proportion)
            continue
        sub_terrains[name] = TrimeshPlatformCfg(
            proportion=proportion,
            stl_file_path=terrain_path,
            add_ground_plane=True,
            ground_size=terrain_size,
            ground_thickness=ground_thickness,
            ground_z=ground_z,
        )

    terrain_cfg = STLTerrainGeneratorCfg(
        size=terrain_size,
        border_width=border_width,
        num_rows=layout.num_rows,
        num_cols=layout.num_cols,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=use_cache,
        sub_terrains=sub_terrains,
    )
    return terrain_cfg, layout


def _hf_terrain_name_map(domain_cfgs) -> dict[str, str]:
    out: dict[str, str] = {}
    for domain_cfg in domain_cfgs:
        if str(getattr(domain_cfg, "pairing", "")) != "unpaired":
            continue
        if str(getattr(domain_cfg, "terrain_kind", "")) != "hf_grid":
            continue
        name = str(domain_cfg.name)
        terrain_profile = str(getattr(domain_cfg, "terrain_profile", "long_runway"))
        for index in range(int(getattr(domain_cfg, "terrain_cell_count", 1))):
            out[f"{name}_{terrain_profile}_{index:03d}"] = terrain_profile
    return out


def _make_hf_terrain_cfg(terrain_profile: str, proportion: float):
    from whole_body_tracking.terrains.height_field import HfLargeStepPlatformTerrainCfg, HfLongRunwayTerrainCfg

    if terrain_profile == "long_runway":
        return HfLongRunwayTerrainCfg(
            proportion=proportion,
            shoulder_width=0.0,
            shoulder_height_range=(0.0, 0.0),
        )
    if terrain_profile == "large_step_platform":
        return HfLargeStepPlatformTerrainCfg(
            proportion=proportion,
            step_height_range=(0.03, 0.25),
            step_depth=1.0,
            start_platform_length=2.0,
            max_steps=4,
        )
    if terrain_profile == "velocity_runway_steps":
        return HfLargeStepPlatformTerrainCfg(
            proportion=proportion,
            step_height_range=(0.03, 0.25),
            step_depth=1.2,
            start_platform_length=3.0,
            max_steps=5,
        )
    raise ValueError(
        f"Unsupported procedural terrain_profile={terrain_profile!r}. "
        "Expected one of: long_runway, large_step_platform, velocity_runway_steps."
    )
