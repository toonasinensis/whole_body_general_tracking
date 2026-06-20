from __future__ import annotations

import torch

from feat_motion_lib.transform.resample import _interpolate_angular


def test_interpolate_angular_uses_slerp_and_normalizes() -> None:
    quat = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )

    resampled = _interpolate_angular(quat, source_fps=1.0, target_fps=2.0)

    assert resampled.shape == (3, 1, 4)
    torch.testing.assert_close(torch.linalg.norm(resampled, dim=-1), torch.ones(3, 1))
    torch.testing.assert_close(
        resampled[1, 0],
        torch.tensor([0.70710677, 0.0, 0.0, 0.70710677]),
        atol=1e-6,
        rtol=1e-6,
    )
