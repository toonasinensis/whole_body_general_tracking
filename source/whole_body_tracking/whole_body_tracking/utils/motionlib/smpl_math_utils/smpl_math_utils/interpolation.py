"""Interpolation utilities ported from gear_sonic/trl/utils/math.py.

All functions operate on torch.Tensor. Batched slerp avoids per-joint Python loops.
"""

import torch

from .rotation import angle_axis_to_quaternion, quaternion_to_angle_axis


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between pairs of quaternions (any order).

    Args:
        q0: (N, 4) start quaternions
        q1: (N, 4) end quaternions
        t:  (N,) blend factors in [0, 1]

    Returns:
        (N, 4) interpolated quaternions
    """
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = (cos_half_theta < 0).unsqueeze(-1).expand_as(q1)
    q1 = torch.where(neg_mask, -q1, q1)
    cos_half_theta = torch.abs(cos_half_theta).unsqueeze(-1)  # (N, 1)

    half_theta = torch.acos(cos_half_theta.clamp(-1.0, 1.0))
    sin_half_theta = torch.sqrt((1.0 - cos_half_theta * cos_half_theta).clamp_min(0.0))

    ratioA = torch.sin((1.0 - t[:, None]) * half_theta) / sin_half_theta
    ratioB = torch.sin(t[:, None] * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1
    # fallback to linear when angle is near 0 or 180
    new_q = torch.where(sin_half_theta.abs() < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(cos_half_theta.abs() >= 1.0, q0, new_q)
    return new_q


def _compute_frame_indices(T: int, source_fps: float, target_fps: float, device: torch.device):
    """Compute frame index arrays for resampling a T-frame sequence."""
    duration = (T - 1) / source_fps
    times = torch.arange(0, duration + 1e-6, 1.0 / target_fps, dtype=torch.float32, device=device)
    times = times[times <= duration]

    frame_indices = times * source_fps
    idx0 = torch.floor(frame_indices).long().clamp(0, T - 1)
    idx1 = (idx0 + 1).clamp(0, T - 1)
    blend = frame_indices - idx0.float()
    return idx0, idx1, blend


def interpolate_linear(data: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Linearly resample a tensor along its first (time) axis.

    Args:
        data: (T, ...) input tensor
        source_fps: source frame rate
        target_fps: target frame rate

    Returns:
        (T_new, ...) resampled tensor
    """
    T = data.shape[0]
    idx0, idx1, blend = _compute_frame_indices(T, source_fps, target_fps, data.device)

    # broadcast blend to match remaining dims
    for _ in range(data.ndim - 1):
        blend = blend.unsqueeze(-1)

    return (1.0 - blend) * data[idx0] + blend * data[idx1]


def interpolate_pose(
    pose_aa: torch.Tensor,
    source_fps: float,
    target_fps: float,
    interpolation_type: str = "slerp",
) -> torch.Tensor:
    """Resample a pose sequence (axis-angle) from source_fps to target_fps.

    Uses fully batched slerp — no Python loop over joints.

    Args:
        pose_aa: (T, N*3) or (T, N, 3) axis-angle pose
        source_fps: source frame rate
        target_fps: target frame rate
        interpolation_type: "slerp" (default) or "linear"

    Returns:
        (T_new, ...) resampled pose, same trailing shape as input
    """
    orig_trailing = pose_aa.shape[1:]
    T = pose_aa.shape[0]

    if interpolation_type == "linear":
        return interpolate_linear(pose_aa, source_fps, target_fps)

    if interpolation_type != "slerp":
        raise ValueError(f"Unknown interpolation_type: {interpolation_type!r}")

    # reshape to (T, N, 3)
    pose_flat = pose_aa.reshape(T, -1, 3)
    N = pose_flat.shape[1]

    # batch convert all T*N angle-axes to quaternions at once
    pose_quat = angle_axis_to_quaternion(pose_flat.reshape(T * N, 3))  # (T*N, 4)
    pose_quat = pose_quat.reshape(T, N, 4)

    idx0, idx1, blend = _compute_frame_indices(T, source_fps, target_fps, pose_aa.device)
    T_new = len(blend)

    # gather frames: (T_new, N, 4)
    q0 = pose_quat[idx0]  # (T_new, N, 4)
    q1 = pose_quat[idx1]

    # flatten joints for vectorised slerp: (T_new*N, 4)
    blend_expanded = blend.repeat_interleave(N)  # (T_new*N,)
    q_interp = slerp(q0.reshape(T_new * N, 4), q1.reshape(T_new * N, 4), blend_expanded)

    # convert back to axis-angle: (T_new*N, 3)
    aa_interp = quaternion_to_angle_axis(q_interp)
    return aa_interp.reshape(T_new, *orig_trailing)


def compute_resample_times(n_frames: int, src_fps: float, tgt_fps: float):
    """Compute resampling timestamps as numpy arrays.

    Uses arange (not linspace) to match motion_lib_base.py and avoid off-by-one errors.

    Returns:
        (times_in, times_out) as float64 numpy arrays
    """

    times_in = torch.arange(n_frames, dtype=torch.float64) / src_fps
    duration = float(times_in[-1])
    # +1e-6 matches motion_lib_base.py to avoid floating-point off-by-one on the last frame
    times_out = torch.arange(0.0, duration + 1e-6, 1.0 / tgt_fps, dtype=torch.float64)
    times_out = times_out[times_out <= duration]
    return times_in.numpy(), times_out.numpy()
