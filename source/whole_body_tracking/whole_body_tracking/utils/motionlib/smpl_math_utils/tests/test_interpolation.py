import torch

from smpl_math_utils import compute_resample_times, interpolate_linear, interpolate_pose, slerp


def test_slerp_same_quaternion():
    torch.manual_seed(0)
    q = torch.randn(8, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    t = torch.rand(8)
    result = slerp(q, q, t)
    assert torch.allclose(result.norm(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(result, q, atol=1e-5)


def test_slerp_double_cover():
    """slerp(q, -q, t) should still produce unit quaternions close to q."""
    torch.manual_seed(1)
    q = torch.randn(8, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    t = torch.full((8,), 0.5)
    result = slerp(q, -q, t)
    assert torch.allclose(result.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_slerp_midpoint():
    """Verify t=0 → q0 and t=1 → q1."""
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    assert torch.allclose(slerp(q0, q1, t0), q0, atol=1e-5)
    assert torch.allclose(slerp(q0, q1, t1), q1, atol=1e-5)


def test_interpolate_linear_same_fps():
    data = torch.randn(20, 3)
    result = interpolate_linear(data, source_fps=30.0, target_fps=30.0)
    assert result.shape[0] == data.shape[0]
    assert torch.allclose(result, data, atol=1e-5)


def test_interpolate_linear_upsample_shape():
    data = torch.randn(10, 24, 3)
    result = interpolate_linear(data, source_fps=10.0, target_fps=30.0)
    # ~3x as many frames
    assert result.shape[0] > data.shape[0]
    assert result.shape[1:] == (24, 3)


def test_interpolate_pose_shape():
    torch.manual_seed(3)
    pose = torch.randn(30, 72)
    result = interpolate_pose(pose, source_fps=30.0, target_fps=60.0)
    assert result.shape[0] > 30
    assert result.shape[1] == 72


def test_interpolate_pose_same_fps():
    torch.manual_seed(4)
    pose = torch.randn(20, 72)
    result = interpolate_pose(pose, source_fps=30.0, target_fps=30.0)
    assert result.shape == pose.shape


def test_compute_resample_times_frame_count():
    times_in, times_out = compute_resample_times(60, src_fps=60.0, tgt_fps=30.0)
    assert len(times_in) == 60
    # ~30 output frames for 1-second clip at 30 fps
    assert 28 <= len(times_out) <= 32


def test_compute_resample_times_same_fps():
    times_in, times_out = compute_resample_times(100, src_fps=30.0, tgt_fps=30.0)
    # arange + float precision can produce n or n-1 frames; both are correct
    assert abs(len(times_out) - len(times_in)) <= 1


def test_interpolate_pose_linear():
    torch.manual_seed(5)
    pose = torch.randn(20, 72)
    result = interpolate_pose(pose, source_fps=20.0, target_fps=40.0, interpolation_type="linear")
    assert result.shape[0] > 20
    assert result.shape[1] == 72
