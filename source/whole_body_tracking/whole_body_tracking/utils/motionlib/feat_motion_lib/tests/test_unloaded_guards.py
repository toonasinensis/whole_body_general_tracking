from __future__ import annotations

import pytest
import torch

from feat_motion_lib.errors import MotionNotLoadedError
from feat_motion_lib.facade.unified_motion_lib import UnifiedMotionLib


def test_unified_motion_lib_body_access_raises_when_unloaded() -> None:
    lib = UnifiedMotionLib(body_indexes=[0], motion_anchor_body_index=0, device="cpu")
    with pytest.raises(MotionNotLoadedError):
        _ = lib.body_pos_w


def test_unified_motion_lib_timestamp_lookup_raises_when_unloaded() -> None:
    lib = UnifiedMotionLib(body_indexes=[0], motion_anchor_body_index=0, device="cpu")
    with pytest.raises(MotionNotLoadedError):
        lib.motion_ids_from_timestamps(torch.tensor([0]))
