"""Transform helpers for feat_motion_lib."""

__all__ = [
    "BODY_KEYS",
    "BODY_NAME_KEYS",
    "JOINT_NAME_KEYS",
    "PairedMotionData",
    "PairedPath",
    "align_body_tensors",
    "align_joint_tensors",
    "decode_name",
    "name_indexes",
    "pair_motion_paths",
    "read_name_list",
    "resample_motion",
    "resample_robot_motion",
    "trim_paired_clips",
    "yup_to_zup_points",
    "yup_to_zup_root_pose",
]


def __getattr__(name: str):
    if name in {"yup_to_zup_points", "yup_to_zup_root_pose"}:
        from .coordinate import yup_to_zup_points, yup_to_zup_root_pose

        mapping = {
            "yup_to_zup_points": yup_to_zup_points,
            "yup_to_zup_root_pose": yup_to_zup_root_pose,
        }
        return mapping[name]
    if name in {"resample_motion", "resample_robot_motion"}:
        from .resample import resample_motion, resample_robot_motion

        mapping = {
            "resample_motion": resample_motion,
            "resample_robot_motion": resample_robot_motion,
        }
        return mapping[name]
    if name in {"PairedMotionData", "PairedPath", "pair_motion_paths", "trim_paired_clips"}:
        from .pairing import pair_motion_paths, trim_paired_clips
        from ..types import PairedMotionData, PairedPath

        mapping = {
            "PairedMotionData": PairedMotionData,
            "PairedPath": PairedPath,
            "pair_motion_paths": pair_motion_paths,
            "trim_paired_clips": trim_paired_clips,
        }
        return mapping[name]
    if name in {
        "BODY_KEYS",
        "BODY_NAME_KEYS",
        "JOINT_NAME_KEYS",
        "align_body_tensors",
        "align_joint_tensors",
        "decode_name",
        "name_indexes",
        "read_name_list",
    }:
        from .alignment import (
            BODY_KEYS,
            BODY_NAME_KEYS,
            JOINT_NAME_KEYS,
            align_body_tensors,
            align_joint_tensors,
            decode_name,
            name_indexes,
            read_name_list,
        )

        mapping = {
            "BODY_KEYS": BODY_KEYS,
            "BODY_NAME_KEYS": BODY_NAME_KEYS,
            "JOINT_NAME_KEYS": JOINT_NAME_KEYS,
            "align_body_tensors": align_body_tensors,
            "align_joint_tensors": align_joint_tensors,
            "decode_name": decode_name,
            "name_indexes": name_indexes,
            "read_name_list": read_name_list,
        }
        return mapping[name]
    raise AttributeError(name)
