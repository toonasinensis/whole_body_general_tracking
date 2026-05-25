from __future__ import annotations

import numpy as np

from .motion import MotionData


class ReferenceMotionPlayer:
    def __init__(
        self,
        model,
        motion: MotionData,
        meta: dict,
        joint_qpos: np.ndarray,
        root_body_name: str | None,
        rgba: np.ndarray,
    ):
        import mujoco

        if "body_pos_w" not in motion or "body_quat_w" not in motion:
            raise ValueError("Reference display needs motion npz fields 'body_pos_w' and 'body_quat_w'.")
        if "joint_pos" not in motion:
            raise ValueError("Reference display needs motion npz field 'joint_pos'.")

        self.model = model
        self.motion = motion
        self.data = mujoco.MjData(model)
        self.joint_qpos = np.asarray(joint_qpos, dtype=np.int32)
        self.rgba = np.asarray(rgba, dtype=np.float32)
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.frame_count = motion.num_frames
        self.motion_body_names = list(meta["motion_body_names"])
        self.root_body_name = root_body_name or self.motion_body_names[0]
        if self.root_body_name not in self.motion_body_names:
            raise ValueError(
                f"Reference root body '{self.root_body_name}' is not in motion_body_names: {self.motion_body_names}"
            )
        self.root_body_index = self.motion_body_names.index(self.root_body_name)
        if int(motion["joint_pos"].shape[1]) != len(self.joint_qpos):
            raise ValueError(
                f"Reference joint_pos dim {motion['joint_pos'].shape[1]} does not match MuJoCo joint dim "
                f"{len(self.joint_qpos)}."
            )
        try:
            self.vopt.geomgroup[:] = 0
            self.vopt.geomgroup[2] = 1
        except Exception:
            pass
        self._last_frame = None

    def update_data(self, frame: int) -> None:
        import mujoco

        frame = int(np.clip(frame, 0, self.frame_count - 1))
        if self._last_frame == frame:
            return
        self._last_frame = frame
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qpos[:3] = self.motion["body_pos_w"][frame, self.root_body_index]
        self.data.qpos[3:7] = self.motion["body_quat_w"][frame, self.root_body_index]
        self.data.qpos[self.joint_qpos] = self.motion["joint_pos"][frame]
        mujoco.mj_forward(self.model, self.data)

    def draw(self, viewer, frame: int) -> None:
        import mujoco

        self.update_data(frame)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            mujoco.mjv_addGeoms(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                mujoco.mjtCatBit.mjCAT_DYNAMIC,
                viewer.user_scn,
            )
            for geom_id in range(viewer.user_scn.ngeom):
                viewer.user_scn.geoms[geom_id].rgba[:] = self.rgba

    def print_config(self) -> None:
        print(
            f"[INFO] Reference motion display: root_body={self.root_body_name}, "
            f"frames={self.frame_count}, rgba={self.rgba.tolist()}"
        )
