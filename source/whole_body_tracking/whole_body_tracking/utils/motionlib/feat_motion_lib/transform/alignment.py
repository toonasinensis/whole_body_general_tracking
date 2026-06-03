from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ..errors import MotionAlignmentError, MotionValidationError

JOINT_NAME_KEYS = ("joint_names", "motion_joint_names", "robot_joint_names", "action_joint_names")
BODY_NAME_KEYS = ("body_names", "motion_body_names", "robot_body_names")
BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


def decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_name_list(raw: np.lib.npyio.NpzFile, keys: Sequence[str]) -> list[str] | None:
    for key in keys:
        if key not in raw.files:
            continue
        arr = np.asarray(raw[key])
        if arr.shape == ():
            value = arr.item()
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
            else:
                return [decode_name(value)]
        return [decode_name(value) for value in arr.reshape(-1).tolist()]
    return None


def name_indexes(source_names: Sequence[str], target_names: Sequence[str], path: str, kind: str) -> list[int]:
    source_to_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_to_index]
    if missing:
        raise MotionAlignmentError(
            f"{path}: npz {kind}_names is missing required names {missing}. "
            f"Available {kind}_names: {list(source_names)}"
        )
    return [source_to_index[name] for name in target_names]


def align_joint_tensors(
    tensors: dict[str, torch.Tensor],
    raw: np.lib.npyio.NpzFile,
    path: str,
    joint_names: Sequence[str] | None,
) -> tuple[dict[str, torch.Tensor], list[str] | None]:
    source_names = read_name_list(raw, JOINT_NAME_KEYS)
    if joint_names is None:
        return tensors, source_names

    joint_dim = int(tensors["joint_pos"].shape[1])
    expected_dim = len(joint_names)
    if source_names is None:
        if joint_dim != expected_dim:
            raise MotionAlignmentError(
                f"{path}: joint_pos dim {joint_dim} does not match expected robot joint dim {expected_dim}, "
                "and no joint_names metadata was found."
            )
        return tensors, list(joint_names)

    if len(source_names) != joint_dim:
        raise MotionAlignmentError(
            f"{path}: joint_names length {len(source_names)} does not match joint_pos dim {joint_dim}."
        )

    indexes = name_indexes(source_names, joint_names, path, "joint")
    for key in ("joint_pos", "joint_vel"):
        tensors[key] = tensors[key][:, indexes]
    return tensors, list(joint_names)


def align_body_tensors(
    tensors: dict[str, torch.Tensor],
    raw: np.lib.npyio.NpzFile,
    path: str,
    motion_body_names: Sequence[str] | None,
    all_body_names: Sequence[str] | None,
    body_indexes: Sequence[int] | None,
) -> tuple[dict[str, torch.Tensor], list[str] | None]:
    body_dim = int(tensors["body_pos_w"].shape[1])
    for key in BODY_KEYS:
        if int(tensors[key].shape[1]) != body_dim:
            raise MotionValidationError(
                f"{path}: body dimension mismatch. body_pos_w has {body_dim}, "
                f"but {key} has {int(tensors[key].shape[1])}."
            )

    selected_body_count = len(motion_body_names) if motion_body_names is not None else len(body_indexes or [])
    source_names = read_name_list(raw, BODY_NAME_KEYS)

    if source_names is not None:
        if len(source_names) != body_dim:
            raise MotionAlignmentError(
                f"{path}: body_names length {len(source_names)} does not match body array dim {body_dim}."
            )
        if motion_body_names is None:
            indexes = list(body_indexes or [])
            names = [source_names[i] for i in indexes] if indexes else list(source_names)
        else:
            indexes = name_indexes(source_names, motion_body_names, path, "body")
            names = list(motion_body_names)
    elif body_dim == selected_body_count and selected_body_count > 0:
        indexes = list(range(selected_body_count))
        names = list(motion_body_names) if motion_body_names is not None else None
    elif all_body_names is not None and body_dim == len(all_body_names):
        if motion_body_names is None:
            indexes = list(body_indexes or [])
            names = [all_body_names[i] for i in indexes] if indexes else list(all_body_names)
        else:
            indexes = name_indexes(all_body_names, motion_body_names, path, "body")
            names = list(motion_body_names)
    elif body_indexes and max(body_indexes) < body_dim:
        indexes = list(body_indexes)
        names = list(motion_body_names) if motion_body_names is not None else None
    else:
        raise MotionAlignmentError(
            f"{path}: cannot map body array dim {body_dim} to selected body dim {selected_body_count}."
        )

    if indexes:
        for key in BODY_KEYS:
            tensors[key] = tensors[key][:, indexes]
    return tensors, names
