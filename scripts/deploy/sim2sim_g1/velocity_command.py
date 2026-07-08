from __future__ import annotations

import array
import fcntl
import numpy as np
import os
import struct
from dataclasses import dataclass


class KeyboardVelocityCommand:
    """Numpad velocity command editor for [vx, vy, yaw_rate]."""

    GLFW_KEY_KP_0 = 320
    GLFW_KEY_KP_4 = 324
    GLFW_KEY_KP_5 = 325
    GLFW_KEY_KP_6 = 326
    GLFW_KEY_KP_7 = 327
    GLFW_KEY_KP_8 = 328
    GLFW_KEY_KP_9 = 329

    def __init__(self, initial: np.ndarray, step: np.ndarray, limit: np.ndarray) -> None:
        self.command = np.asarray(initial, dtype=np.float32).copy()
        self.step = np.asarray(step, dtype=np.float32)
        self.limit = np.asarray(limit, dtype=np.float32)

    def on_key(self, key: int) -> None:
        if key in (self.GLFW_KEY_KP_8, ord("8")):
            self.command[0] += self.step[0]
        elif key in (self.GLFW_KEY_KP_5, ord("5")):
            self.command[0] -= self.step[0]
        elif key in (self.GLFW_KEY_KP_4, ord("4")):
            self.command[1] += self.step[1]
        elif key in (self.GLFW_KEY_KP_6, ord("6")):
            self.command[1] -= self.step[1]
        elif key in (self.GLFW_KEY_KP_7, ord("7")):
            self.command[2] += self.step[2]
        elif key in (self.GLFW_KEY_KP_9, ord("9")):
            self.command[2] -= self.step[2]
        elif key in (self.GLFW_KEY_KP_0, ord("0")):
            self.command[:] = 0.0
        else:
            return
        self.command[:] = np.clip(self.command, -self.limit, self.limit)
        print(f"[VCMD] keyboard vx={self.command[0]:.3f} vy={self.command[1]:.3f} yaw={self.command[2]:.3f}")

    def poll(self) -> None:
        return

    def value(self) -> np.ndarray:
        return self.command.reshape(1, 3).astype(np.float32)


class LinuxJoystickVelocityCommand:
    """Linux joystick command reader using /dev/input/js* without third-party deps."""

    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80
    JSIOCGNAME_128 = 0x80806A13

    def __init__(
        self,
        *,
        device: str,
        axis_vx: int,
        axis_vy: int,
        axis_yaw: int,
        scale: np.ndarray,
        limit: np.ndarray,
        deadzone: float,
        invert: np.ndarray,
    ) -> None:
        self.device = str(device)
        self.axis_vx = int(axis_vx)
        self.axis_vy = int(axis_vy)
        self.axis_yaw = int(axis_yaw)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.limit = np.asarray(limit, dtype=np.float32)
        self.deadzone = float(deadzone)
        self.invert = np.asarray(invert, dtype=np.float32)
        self.axes = np.zeros(32, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.fd = os.open(self.device, os.O_RDONLY | os.O_NONBLOCK)
        self.name = self._read_name()

    def _read_name(self) -> str:
        buf = array.array("B", [0] * 128)
        try:
            fcntl.ioctl(self.fd, self.JSIOCGNAME_128, buf)
            return bytes(buf).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        except OSError:
            return self.device

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def poll(self) -> None:
        while True:
            try:
                event = os.read(self.fd, 8)
            except BlockingIOError:
                break
            if len(event) < 8:
                break
            _, value, event_type, number = struct.unpack("IhBB", event)
            is_init = bool(int(event_type) & self.JS_EVENT_INIT)
            event_type = int(event_type) & ~self.JS_EVENT_INIT
            if is_init:
                continue
            if event_type != self.JS_EVENT_AXIS:
                continue
            axis = int(number)
            if axis >= self.axes.shape[0]:
                continue
            raw = float(value) / 32767.0
            if abs(raw) < self.deadzone:
                raw = 0.0
            self.axes[axis] = float(np.clip(raw, -1.0, 1.0))
        selected = np.asarray(
            [self.axes[self.axis_vx], self.axes[self.axis_vy], self.axes[self.axis_yaw]],
            dtype=np.float32,
        )
        self.command[:] = np.clip(selected * self.scale * self.invert, -self.limit, self.limit)

    def value(self) -> np.ndarray:
        self.poll()
        return self.command.reshape(1, 3).astype(np.float32)

    def print_config(self) -> None:
        print(
            "[INFO] Joystick velocity command: "
            f"device={self.device}, name={self.name!r}, axes=(vx:{self.axis_vx}, vy:{self.axis_vy}, "
            f"yaw:{self.axis_yaw}), scale={self.scale.tolist()}, deadzone={self.deadzone}, "
            f"invert={self.invert.tolist()}, limit={self.limit.tolist()}"
        )


@dataclass
class VelocityArrowConfig:
    body_name: str = "pelvis"
    z: float = 0.08
    scale: float = 0.45
    cmd_radius: float = 0.035
    actual_radius: float = 0.025


class VelocityArrowVisualizer:
    """Draw command velocity and measured robot velocity arrows in the viewer."""

    def __init__(self, cfg: VelocityArrowConfig) -> None:
        self.cfg = cfg

    def draw(self, viewer, model, data, command_yaw_frame: np.ndarray) -> None:
        import mujoco

        body_id = int(model.body(self.cfg.body_name).id)
        root_pos = np.asarray(data.xpos[body_id], dtype=np.float64)
        body_xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        yaw = float(np.arctan2(body_xmat[1, 0], body_xmat[0, 0]))
        c, s = np.cos(yaw), np.sin(yaw)
        cmd = np.asarray(command_yaw_frame, dtype=np.float64).reshape(-1)
        cmd_w = np.asarray([c * cmd[0] - s * cmd[1], s * cmd[0] + c * cmd[1], 0.0], dtype=np.float64)
        actual_w = np.asarray([data.qvel[0], data.qvel[1], 0.0], dtype=np.float64)
        start = np.asarray([root_pos[0], root_pos[1], self.cfg.z], dtype=np.float64)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            _append_arrow(
                viewer.user_scn,
                mujoco,
                start,
                start + self.cfg.scale * cmd_w,
                radius=self.cfg.cmd_radius,
                rgba=np.asarray([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
            )
            _append_arrow(
                viewer.user_scn,
                mujoco,
                start + np.asarray([0.0, 0.0, 0.08], dtype=np.float64),
                start + np.asarray([0.0, 0.0, 0.08], dtype=np.float64) + self.cfg.scale * actual_w,
                radius=self.cfg.actual_radius,
                rgba=np.asarray([0.1, 0.35, 1.0, 1.0], dtype=np.float32),
            )


def _append_arrow(scn, mujoco, start: np.ndarray, end: np.ndarray, *, radius: float, rgba: np.ndarray) -> None:
    max_geoms = int(getattr(scn, "maxgeom", len(scn.geoms)))
    if scn.ngeom >= max_geoms:
        return
    if np.linalg.norm(end - start) < 1e-4:
        end = start + np.asarray([0.001, 0.0, 0.0], dtype=np.float64)
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, float(radius), start, end)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.rgba[:] = rgba
    scn.ngeom += 1


def robot_velocity_yaw_frame(model, data, body_name: str = "pelvis") -> np.ndarray:
    body_id = int(model.body(body_name).id)
    body_xmat = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    yaw = float(np.arctan2(body_xmat[1, 0], body_xmat[0, 0]))
    c, s = np.cos(yaw), np.sin(yaw)
    vel_w = np.asarray([data.qvel[0], data.qvel[1], 0.0], dtype=np.float64)
    vx = c * vel_w[0] + s * vel_w[1]
    vy = -s * vel_w[0] + c * vel_w[1]
    return np.asarray([vx, vy, data.qvel[5]], dtype=np.float32)
