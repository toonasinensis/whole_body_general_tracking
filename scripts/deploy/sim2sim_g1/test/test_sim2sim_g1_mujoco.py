from __future__ import annotations

import numpy as np
import sys
import torch
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from sim2sim_g1.mujoco_robot import action_to_target
from sim2sim_g1.observations import ImuReader, TermMajorHistory, prop_terms_from_metadata
from sim2sim_g1.onnx_policy import onnx_input_names, validate_grouped_onnx_contract
from sim2sim_g1.viewer import ReferenceMotionPlayer

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source" / "whole_body_tracking"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
from whole_body_tracking.utils.exporter import append_onnx_metadata  # noqa: E402


def test_legacy_import_surface_smoke() -> None:
    assert ImuReader is not None
    assert TermMajorHistory is not None
    assert ReferenceMotionPlayer is not None


def test_legacy_action_to_target() -> None:
    raw_action = np.array([[0.5, -2.0]], dtype=np.float32)
    action_scale = np.array([2.0, 3.0], dtype=np.float64)
    action_offset = np.array([1.0, -1.0], dtype=np.float64)

    assert np.allclose(action_to_target(raw_action, action_scale, action_offset), np.array([2.0, -7.0]))


def test_legacy_term_major_history_matches_isaaclab_layout() -> None:
    history = TermMajorHistory([("a", 2, 3), ("b", 1, 3)])
    history.update({"a": np.array([[1.0, 2.0]]), "b": np.array([[10.0]])})
    history.update({"a": np.array([[3.0, 4.0]]), "b": np.array([[20.0]])})
    third = history.update({"a": np.array([[5.0, 6.0]]), "b": np.array([[30.0]])})

    assert np.array_equal(third, np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0]], dtype=np.float32))


def test_legacy_prop_terms_fallback_for_old_metadata() -> None:
    meta = {"observation_shapes": {"prop": [840]}}

    assert prop_terms_from_metadata(meta, num_joints=26) == [
        ("projected_gravity", 3, 10),
        ("base_ang_vel", 3, 10),
        ("joint_pos", 26, 10),
        ("joint_vel", 26, 10),
        ("actions", 26, 10),
    ]


def test_legacy_onnx_contract_accepts_required_inputs() -> None:
    meta = {
        "input_names": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"],
        "observation_shapes": {"prop": [10], "rbt_cmd_mf": [20], "smpl_cmd_mf": [30]},
    }

    validate_grouped_onnx_contract(["prop", "rbt_cmd_mf", "smpl_cmd_mf"], meta)


def test_legacy_onnx_input_names_ignores_initializers(tmp_path) -> None:
    x = helper.make_tensor_value_info("prop", TensorProto.FLOAT, [1, 3])
    w = helper.make_tensor("weight", TensorProto.FLOAT, [3, 2], [1.0] * 6)
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("MatMul", ["prop", "weight"], ["actions"])
    graph = helper.make_graph([node], "test", [x], [y], [w])
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)

    assert onnx_input_names(str(path)) == ["prop"]


def test_legacy_append_onnx_metadata_serializes_numpy_and_torch_values(tmp_path) -> None:
    x = helper.make_tensor_value_info("prop", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("Identity", ["prop"], ["actions"])
    graph = helper.make_graph([node], "test", [x], [y])
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)

    append_onnx_metadata(
        str(path),
        {
            "np_int": np.int64(3),
            "np_float": np.float32(1.5),
            "torch_tensor": torch.tensor([1, 2], dtype=torch.int64),
            "torch_size": torch.Size([4, 5]),
            "nested": {"shape": (np.int64(6), torch.tensor(7))},
        },
    )

    loaded = onnx.load(path)
    meta = {entry.key: entry.value for entry in loaded.metadata_props}
    assert meta["np_int"] == "3"
    assert meta["np_float"] == "1.5"
    assert meta["torch_tensor"] == "[1, 2]"
    assert meta["torch_size"] == "[4, 5]"
    assert meta["nested"] == '{"shape": [6, 7]}'
