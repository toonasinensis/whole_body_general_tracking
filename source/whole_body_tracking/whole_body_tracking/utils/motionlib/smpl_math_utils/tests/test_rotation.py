import torch

from smpl_math_utils import (
    angle_axis_to_quaternion,
    angle_axis_to_rotation_matrix,
    normalize_quaternion,
    quaternion_to_angle_axis,
    rotation_matrix_to_angle_axis,
)


def test_zero_aa_is_identity():
    aa = torch.zeros(4, 3)
    mat = angle_axis_to_rotation_matrix(aa)
    expected = torch.eye(3).unsqueeze(0).repeat(4, 1, 1)
    assert torch.allclose(mat, expected, atol=1e-6)


def test_aa_roundtrip():
    torch.manual_seed(0)
    aa = torch.randn(16, 3) * 0.5
    mat = angle_axis_to_rotation_matrix(aa)
    aa_back = rotation_matrix_to_angle_axis(mat)
    assert torch.allclose(aa, aa_back, atol=1e-5)


def test_aa_to_quat_unit_length():
    torch.manual_seed(1)
    aa = torch.randn(32, 3)
    q = angle_axis_to_quaternion(aa)
    norms = q.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(32), atol=1e-6)


def test_quat_to_aa_roundtrip():
    torch.manual_seed(2)
    aa = torch.randn(16, 3) * 0.8
    q = angle_axis_to_quaternion(aa)
    aa_back = quaternion_to_angle_axis(q)
    assert torch.allclose(aa, aa_back, atol=1e-5)


def test_aa_quat_batched_shapes():
    aa = torch.randn(8, 24, 3)
    q = angle_axis_to_quaternion(aa)
    assert q.shape == (8, 24, 4)
    aa_back = quaternion_to_angle_axis(q)
    assert aa_back.shape == (8, 24, 3)


def test_normalize_quaternion():
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    qn = normalize_quaternion(q)
    assert torch.allclose(qn.norm(dim=-1), torch.ones(1), atol=1e-6)
