#!/usr/bin/env python3
from __future__ import annotations

import sys

from sim2sim_g1_mujoco import main


if __name__ == "__main__":
    if "--robot_config" not in sys.argv:
        sys.argv[1:1] = ["--robot_config", "roban"]
    main()
