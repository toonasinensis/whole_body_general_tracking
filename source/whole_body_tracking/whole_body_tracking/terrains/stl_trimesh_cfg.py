from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class TrimeshPlatformCfg:
    proportion: str = MISSING
    stl_file_path: str = None
