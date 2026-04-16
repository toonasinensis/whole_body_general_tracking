# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import whole_body_tracking.terrains as terrain_gen

GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat_": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(-0.0, 0.0), noise_step=0.02, border_width=0.0
        ),
    },
)
"""Rough terrains configuration."""


import glob
import os

from whole_body_tracking.terrains.stl_trimesh_cfg import TrimeshPlatformCfg


# 自动扫描 assets/terrains/ 目录下的所有 STL 文件，生成均匀分配比例的地形配置
def _build_stl_sub_terrains(terrain_dir: str) -> dict:
    # Keep the same lexicographic ordering strategy as motion npz loading.
    stl_files = sorted(glob.glob(os.path.join(terrain_dir, "*.stl")))
    if not stl_files:
        raise FileNotFoundError(f"No STL files found in {terrain_dir}")
    proportion = 1.0 / len(stl_files)

    # 打印地形序列
    print("\n=== STL Terrain Sequence ===")
    for idx, f in enumerate(stl_files):
        basename = os.path.basename(f)
        print(f"Terrain {idx}: {basename}")
    print(f"Total: {len(stl_files)} terrains\n")

    return {
        os.path.splitext(os.path.basename(f))[0]: TrimeshPlatformCfg(proportion=proportion, stl_file_path=f)
        for f in stl_files
    }


# TODO: 有课程的STL platform terrains cfg
