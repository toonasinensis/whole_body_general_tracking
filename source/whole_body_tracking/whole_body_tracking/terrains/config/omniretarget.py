from __future__ import annotations

import glob
import math
import os
from pathlib import Path

from whole_body_tracking.terrains.paired_manifest import resolve_repo_path
from whole_body_tracking.terrains.stl_terrain_generator_cfg import STLTerrainGeneratorCfg
from whole_body_tracking.terrains.stl_trimesh_cfg import TrimeshPlatformCfg

DEFAULT_OMNIRETARGET_G1_TERRAIN_DIR = "data/omniretarget/g1_terrain/terrains/stl"


def build_omniretarget_g1_terrain_cfg(
    terrain_dir: str = DEFAULT_OMNIRETARGET_G1_TERRAIN_DIR,
    *,
    num_rows: int | None = None,
    num_cols: int | None = None,
    terrain_size: tuple[float, float] = (6.0, 6.0),
    border_width: float = 0.0,
    use_cache: bool = False,
) -> STLTerrainGeneratorCfg:
    """Build an STL terrain grid whose sorted order matches sorted OmniRetarget motion stems."""
    terrain_root = resolve_repo_path(terrain_dir)
    stl_files = sorted(glob.glob(os.path.join(str(terrain_root), "*.stl")))
    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {terrain_root}")

    if num_cols is None:
        num_cols = int(math.ceil(math.sqrt(len(stl_files))))
    if num_rows is None:
        num_rows = int(math.ceil(len(stl_files) / num_cols))

    sub_terrains = {
        Path(path).stem: TrimeshPlatformCfg(proportion=1.0 / len(stl_files), stl_file_path=path) for path in stl_files
    }
    return STLTerrainGeneratorCfg(
        size=terrain_size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=use_cache,
        sub_terrains=sub_terrains,
    )


OMNIRETARGET_G1_TERRAINS_CFG = build_omniretarget_g1_terrain_cfg
