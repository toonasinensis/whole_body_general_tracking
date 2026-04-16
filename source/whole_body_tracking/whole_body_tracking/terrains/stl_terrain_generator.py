# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import omni.log

from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.trimesh.utils import make_border  # noqa: F401
from isaaclab.terrains.utils import color_meshes_by_height, find_flat_patches  # noqa: F401
from isaaclab.utils.dict import dict_to_md5_hash  # noqa: F401
from isaaclab.utils.io import dump_yaml  # noqa: F401
from isaaclab.utils.timer import Timer
from isaaclab.utils.warp import convert_to_warp_mesh  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg  # noqa: F401

    from .stl_terrain_generator_cfg import STLTerrainGeneratorCfg
    from .stl_trimesh_cfg import TrimeshPlatformCfg


class STLTerrainGenerator(TerrainGenerator):
    def __init__(self, cfg: STLTerrainGeneratorCfg, device: str = "cpu"):
        """Initialize the terrain generator.

        Args:
            cfg: Configuration for the terrain generator.
            device: The device to use for the flat patches tensor.
        """
        # check inputs
        if len(cfg.sub_terrains) == 0:
            raise ValueError("No sub-terrains specified! Please add at least one sub-terrain.")
        # store inputs
        self.cfg = cfg
        self.device = device

        # set common values to all sub-terrains config
        from .height_field import HfTerrainBaseCfg  # prevent circular import

        for sub_cfg in self.cfg.sub_terrains.values():
            # size of all terrains
            sub_cfg.size = self.cfg.size
            # params for height field terrains
            if isinstance(sub_cfg, HfTerrainBaseCfg):
                sub_cfg.horizontal_scale = self.cfg.horizontal_scale
                sub_cfg.vertical_scale = self.cfg.vertical_scale
                sub_cfg.slope_threshold = self.cfg.slope_threshold

        # throw a warning if the cache is enabled but the seed is not set
        if self.cfg.use_cache and self.cfg.seed is None:
            omni.log.warn(
                "Cache is enabled but the seed is not set. The terrain generation will not be reproducible."
                " Please set the seed in the terrain generator configuration to make the generation reproducible."
            )

        # if the seed is not set, we assume there is a global seed set and use that.
        # this ensures that the terrain is reproducible if the seed is set at the beginning of the program.
        if self.cfg.seed is not None:
            seed = self.cfg.seed
        else:
            seed = np.random.get_state()[1][0]
        # set the seed for reproducibility
        # note: we create a new random number generator to avoid affecting the global state
        #  in the other places where random numbers are used.
        self.np_rng = np.random.default_rng(seed)

        # buffer for storing valid patches
        self.flat_patches = {}
        # create a list of all sub-terrains
        self.terrain_meshes = list()
        self.terrain_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3))

        # parse configuration and add sub-terrains
        # create terrains based on curriculum or randomly
        if self.cfg.curriculum:
            with Timer("[INFO] Generating terrains based on curriculum took"):
                self._generate_curriculum_terrains()
        else:
            with Timer("[INFO] Generating terrains randomly took"):
                # self._generate_random_terrains()
                self._generate_random_terrains_stl()
        # add a border around the terrains
        self._add_terrain_border()
        # combine all the sub-terrains into a single mesh
        self.terrain_mesh = trimesh.util.concatenate(self.terrain_meshes)

        # color the terrain mesh
        if self.cfg.color_scheme == "height":
            self.terrain_mesh = color_meshes_by_height(self.terrain_mesh)
        elif self.cfg.color_scheme == "random":
            self.terrain_mesh.visual.vertex_colors = self.np_rng.choice(
                range(256), size=(len(self.terrain_mesh.vertices), 4)
            )
        elif self.cfg.color_scheme == "none":
            pass
        else:
            raise ValueError(f"Invalid color scheme: {self.cfg.color_scheme}.")

        # offset the entire terrain and origins so that it is centered
        # -- terrain mesh
        transform = np.eye(4)
        transform[:2, -1] = -self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5
        self.terrain_mesh.apply_transform(transform)
        # -- terrain origins
        self.terrain_origins += transform[:3, -1]
        # -- valid patches
        terrain_origins_torch = torch.tensor(self.terrain_origins, dtype=torch.float, device=self.device).unsqueeze(2)
        for name, value in self.flat_patches.items():
            self.flat_patches[name] = value + terrain_origins_torch

    def _generate_random_terrains_stl(self):
        """Add terrains in deterministic index order so Terrain i matches sub-terrain i."""
        # keep insertion order from cfg.sub_terrains (prepared in config)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())
        if len(sub_terrains_cfgs) == 0:
            raise ValueError("No STL sub-terrains available for deterministic placement.")

        # fill terrain grid in row-major order with deterministic mapping:
        # index -> sub_index (with wrap-around if grid is larger than sub-terrain count)
        for index in range(self.cfg.num_rows * self.cfg.num_cols):
            # coordinate index of the sub-terrain
            (sub_row, sub_col) = np.unravel_index(index, (self.cfg.num_rows, self.cfg.num_cols))
            sub_index = index % len(sub_terrains_cfgs)
            mesh, origin = self._get_terrain_mesh(sub_terrains_cfgs[sub_index])
            # add to sub-terrains
            self._add_sub_terrain_stl(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[sub_index])

    def _add_sub_terrain_stl(
        self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, sub_terrain_cfg: TrimeshPlatformCfg
    ):
        """Add input sub-terrain to the list of sub-terrains.

        This function adds the input sub-terrain mesh to the list of sub-terrains and updates the origin
        of the sub-terrain in the list of origins. It also samples flat patches if specified.

        Args:
            mesh: The mesh of the sub-terrain.
            origin: The origin of the sub-terrain.
            row: The row index of the sub-terrain.
            col: The column index of the sub-terrain.
        """
        # transform the mesh to the correct position
        transform = np.eye(4)
        transform[0:2, -1] = (row + 0.5) * self.cfg.size[0], (col + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        # add mesh to the list
        self.terrain_meshes.append(mesh)
        # add origin to the list
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _get_terrain_mesh(self, cfg: TrimeshPlatformCfg) -> tuple[trimesh.Trimesh, np.ndarray]:
        mesh = trimesh.load(cfg.stl_file_path).copy()
        origin = np.array([0.0, 0.0, 0.0])  # 这个origin是复活点不是mesh位置
        return mesh, origin
