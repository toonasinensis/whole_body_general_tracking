from dataclasses import MISSING
from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg

@configclass
class TrimeshPlatformCfg:
    proportion: str = MISSING
    stl_file_path: str = None
