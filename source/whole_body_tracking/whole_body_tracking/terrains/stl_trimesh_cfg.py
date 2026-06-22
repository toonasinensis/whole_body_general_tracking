from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class TrimeshPlatformCfg:
    proportion: str = MISSING
    stl_file_path: str = None
    add_ground_plane: bool = False
    ground_size: tuple[float, float] | None = None
    ground_thickness: float = 0.01
    ground_z: float = 0.0
