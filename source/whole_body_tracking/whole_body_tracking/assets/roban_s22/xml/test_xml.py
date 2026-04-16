import numpy as np
import time
from pathlib import Path

import mujoco
import mujoco.viewer

XML_PATH = Path(__file__).resolve().parent / "scene.xml"
HANG_HEIGHT = 1.2


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    # Set root pose and joint pos to zero-like observation pose.
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.nq >= 7:
        data.qpos[0:3] = np.array([0.0, 0.0, HANG_HEIGHT], dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    mujoco.mj_forward(model, data)
    print(f"Loaded: {XML_PATH}")
    print(f"qpos set done. nq={model.nq}, nv={model.nv}, nu={model.nu}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
