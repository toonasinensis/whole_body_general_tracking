"""Rotation utilities ported from gear_sonic/trl/utils/kornia_transform.py.

All functions operate on torch.Tensor with batch dimensions.
Quaternion convention: wxyz throughout (w first).
"""

import torch
import torch.nn.functional as F


def _compute_rotation_matrix(angle_axis: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    k_one = 1.0
    theta = torch.sqrt(theta2.clamp_min(eps))
    wxyz = angle_axis / (theta + eps)
    wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    r00 = cos_theta + wx * wx * (k_one - cos_theta)
    r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
    r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
    r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
    r11 = cos_theta + wy * wy * (k_one - cos_theta)
    r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
    r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
    r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
    r22 = cos_theta + wz * wz * (k_one - cos_theta)
    rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
    return rotation_matrix.view(-1, 3, 3)


def _compute_rotation_matrix_taylor(angle_axis: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
    k_one = torch.ones_like(rx)
    rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
    return rotation_matrix.view(-1, 3, 3)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix.

    Args:
        angle_axis: (*, 3)

    Returns:
        (*, 3, 3) rotation matrices
    """
    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input size must be (*, 3). Got {angle_axis.shape}")

    orig_shape = angle_axis.shape
    angle_axis_flat = angle_axis.reshape(-1, 3)

    _aa = torch.unsqueeze(angle_axis_flat, dim=1)
    theta2 = torch.matmul(_aa, _aa.transpose(1, 2)).squeeze(1)

    rotation_matrix_normal = _compute_rotation_matrix(angle_axis_flat, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis_flat)

    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = mask.type_as(theta2)
    mask_neg = (~mask).type_as(theta2)

    batch_size = angle_axis_flat.shape[0]
    rotation_matrix = torch.eye(3, device=angle_axis.device, dtype=angle_axis.dtype)
    rotation_matrix = rotation_matrix.view(1, 3, 3).repeat(batch_size, 1, 1)
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix.view(orig_shape[:-1] + (3, 3))


def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denominator = torch.where(denominator.abs() < eps, denominator + eps, denominator)
    return numerator / denominator


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize quaternion(s) to unit length. Input: (*, 4)."""
    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be (*, 4). Got {quaternion.shape}")
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert rotation matrix to quaternion (wxyz).

    Args:
        rotation_matrix: (*, 3, 3)

    Returns:
        (*, 4) quaternions in wxyz order
    """
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input must be (*, 3, 3). Got {rotation_matrix.shape}")

    m00 = rotation_matrix[..., 0, 0]
    m01 = rotation_matrix[..., 0, 1]
    m02 = rotation_matrix[..., 0, 2]
    m10 = rotation_matrix[..., 1, 0]
    m11 = rotation_matrix[..., 1, 1]
    m12 = rotation_matrix[..., 1, 2]
    m20 = rotation_matrix[..., 2, 0]
    m21 = rotation_matrix[..., 2, 1]
    m22 = rotation_matrix[..., 2, 2]

    trace = m00 + m11 + m22

    sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0
    qw = 0.25 * sq
    qx = safe_zero_division(m21 - m12, sq)
    qy = safe_zero_division(m02 - m20, sq)
    qz = safe_zero_division(m10 - m01, sq)
    trace_positive_cond = torch.stack((qw, qx, qy, qz), dim=-1)

    sq = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)) * 2.0
    qw = safe_zero_division(m21 - m12, sq)
    qx = 0.25 * sq
    qy = safe_zero_division(m01 + m10, sq)
    qz = safe_zero_division(m02 + m20, sq)
    cond_1 = torch.stack((qw, qx, qy, qz), dim=-1)

    sq = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(eps)) * 2.0
    qw = safe_zero_division(m02 - m20, sq)
    qx = safe_zero_division(m01 + m10, sq)
    qy = 0.25 * sq
    qz = safe_zero_division(m12 + m21, sq)
    cond_2 = torch.stack((qw, qx, qy, qz), dim=-1)

    sq = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(eps)) * 2.0
    qw = safe_zero_division(m10 - m01, sq)
    qx = safe_zero_division(m02 + m20, sq)
    qy = safe_zero_division(m12 + m21, sq)
    qz = 0.25 * sq
    cond_3 = torch.stack((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where((m11 > m22).unsqueeze(-1), cond_2, cond_3)
    where_1 = torch.where(((m00 > m11) & (m00 > m22)).unsqueeze(-1), cond_1, where_2)
    quaternion = torch.where((trace > 0.0).unsqueeze(-1), trace_positive_cond, where_1)
    return quaternion


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (wxyz) to rotation matrix.

    Args:
        quaternion: (*, 4) wxyz

    Returns:
        (*, 3, 3) rotation matrices
    """
    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be (*, 4). Got {quaternion.shape}")

    quaternion_norm = normalize_quaternion(quaternion)
    w, x, y, z = (
        quaternion_norm[..., 0],
        quaternion_norm[..., 1],
        quaternion_norm[..., 2],
        quaternion_norm[..., 3],
    )

    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0, device=quaternion.device, dtype=quaternion.dtype)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(quaternion.shape[:-1] + (3, 3))
    return matrix


def quaternion_to_angle_axis(quaternion: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert quaternion (wxyz) to axis-angle.

    Args:
        quaternion: (*, 4) wxyz

    Returns:
        (*, 3) axis-angle
    """
    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be (*, 4). Got {quaternion.shape}")

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta = torch.sqrt(sin_squared_theta.clamp_min(eps))

    def _safe_atan2(y, x, a_eps=1e-6):
        y = y.clone()
        y[(y.abs() < a_eps) & (x.abs() < a_eps)] += a_eps
        return torch.atan2(y, x)

    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        _safe_atan2(-sin_theta, -cos_theta),
        _safe_atan2(sin_theta, cos_theta),
    )

    k_pos = safe_zero_division(two_theta, sin_theta, eps)
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] = q1 * k
    angle_axis[..., 1] = q2 * k
    angle_axis[..., 2] = q3 * k
    return angle_axis


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to axis-angle.

    Args:
        rotation_matrix: (*, 3, 3)

    Returns:
        (*, 3) axis-angle
    """
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input must be (*, 3, 3). Got {rotation_matrix.shape}")
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def angle_axis_to_quaternion(angle_axis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert axis-angle to quaternion (wxyz).

    Args:
        angle_axis: (*, 3)

    Returns:
        (*, 4) quaternions in wxyz order
    """
    if not angle_axis.shape[-1] == 3:
        raise ValueError(f"Input must be (*, 3). Got {angle_axis.shape}")

    a0 = angle_axis[..., 0:1]
    a1 = angle_axis[..., 1:2]
    a2 = angle_axis[..., 2:3]
    theta_squared = a0 * a0 + a1 * a1 + a2 * a2
    theta = torch.sqrt(theta_squared.clamp_min(eps))
    half_theta = theta * 0.5

    mask = theta_squared > 0.0
    ones = torch.ones_like(half_theta)
    k = torch.where(mask, safe_zero_division(torch.sin(half_theta), theta, eps), 0.5 * ones)
    w = torch.where(mask, torch.cos(half_theta), ones)

    quaternion = torch.zeros(angle_axis.shape[:-1] + (4,), dtype=angle_axis.dtype, device=angle_axis.device)
    # wxyz order: index 0 = w
    quaternion[..., 0:1] = w
    quaternion[..., 1:2] = a0 * k
    quaternion[..., 2:3] = a1 * k
    quaternion[..., 3:4] = a2 * k
    return quaternion
