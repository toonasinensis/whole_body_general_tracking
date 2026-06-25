from __future__ import annotations

import numpy as np

from sim2sim_g1.math_utils import resize_or_zero


def test_resize_or_zero_pads_and_truncates() -> None:
    assert np.array_equal(
        resize_or_zero(np.array([[1.0, 2.0]], dtype=np.float32), 4), np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    )
    assert np.array_equal(
        resize_or_zero(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), 2), np.array([[1.0, 2.0]], dtype=np.float32)
    )
