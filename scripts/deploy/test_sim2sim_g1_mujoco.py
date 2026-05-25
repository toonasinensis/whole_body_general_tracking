from __future__ import annotations

import numpy as np

import onnx
from onnx import TensorProto, helper
from sim2sim_g1_mujoco import TermMajorHistory, _action_to_target, _onnx_input_names, _prop_terms_from_metadata


def test_term_major_history_matches_isaaclab_layout() -> None:
    history = TermMajorHistory([("a", 2, 3), ("b", 1, 3)])

    first = history.update({"a": np.array([[1.0, 2.0]]), "b": np.array([[10.0]])})
    assert np.array_equal(first, np.array([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 10.0, 10.0, 10.0]], dtype=np.float32))

    history.update({"a": np.array([[3.0, 4.0]]), "b": np.array([[20.0]])})
    third = history.update({"a": np.array([[5.0, 6.0]]), "b": np.array([[30.0]])})

    expected_term_major = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0]], dtype=np.float32)
    frame_major = np.array([[1.0, 2.0, 10.0, 3.0, 4.0, 20.0, 5.0, 6.0, 30.0]], dtype=np.float32)
    assert np.array_equal(third, expected_term_major)
    assert not np.array_equal(third, frame_major)


def test_prop_terms_from_new_metadata() -> None:
    meta = {
        "observation_shapes": {"prop": [48]},
        "observation_terms": {
            "prop": {
                "terms": [
                    {
                        "name": "projected_gravity",
                        "shape": [6],
                        "base_shape": [3],
                        "history_length": 2,
                        "flatten_history_dim": True,
                    },
                    {
                        "name": "joint_pos",
                        "shape": [42],
                        "base_shape": [21],
                        "history_length": 2,
                        "flatten_history_dim": True,
                    },
                ]
            }
        },
    }

    assert _prop_terms_from_metadata(meta, num_joints=21) == [("projected_gravity", 3, 2), ("joint_pos", 21, 2)]


def test_prop_terms_fallback_for_old_metadata() -> None:
    meta = {"observation_shapes": {"prop": [840]}}

    assert _prop_terms_from_metadata(meta, num_joints=26) == [
        ("projected_gravity", 3, 10),
        ("base_ang_vel", 3, 10),
        ("joint_pos", 26, 10),
        ("joint_vel", 26, 10),
        ("actions", 26, 10),
    ]


def test_action_to_target_uses_policy_output() -> None:
    raw_action = np.array([[0.5, -2.0]], dtype=np.float32)
    action_scale = np.array([2.0, 3.0], dtype=np.float64)
    action_offset = np.array([1.0, -1.0], dtype=np.float64)

    assert np.allclose(_action_to_target(raw_action, action_scale, action_offset), np.array([2.0, -7.0]))


def test_onnx_input_names_ignores_initializers(tmp_path) -> None:
    x = helper.make_tensor_value_info("prop", TensorProto.FLOAT, [1, 3])
    w = helper.make_tensor("weight", TensorProto.FLOAT, [3, 2], [1.0] * 6)
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("MatMul", ["prop", "weight"], ["actions"])
    graph = helper.make_graph([node], "test", [x], [y], [w])
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)

    assert _onnx_input_names(str(path)) == ["prop"]
