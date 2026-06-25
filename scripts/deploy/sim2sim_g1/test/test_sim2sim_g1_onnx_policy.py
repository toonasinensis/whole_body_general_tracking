from __future__ import annotations

import onnx
import pytest
from onnx import TensorProto, helper
from sim2sim_g1.onnx_policy import onnx_input_names, validate_grouped_onnx_contract


def test_onnx_input_names_ignores_initializers(tmp_path) -> None:
    x = helper.make_tensor_value_info("prop", TensorProto.FLOAT, [1, 3])
    w = helper.make_tensor("weight", TensorProto.FLOAT, [3, 2], [1.0] * 6)
    y = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    node = helper.make_node("MatMul", ["prop", "weight"], ["actions"])
    graph = helper.make_graph([node], "test", [x], [y], [w])
    model = helper.make_model(graph)
    path = tmp_path / "policy.onnx"
    onnx.save(model, path)

    assert onnx_input_names(str(path)) == ["prop"]


def test_grouped_onnx_contract_rejects_old_single_input() -> None:
    with pytest.raises(ValueError, match="old single-input ONNX"):
        validate_grouped_onnx_contract(["obs"], {})


def test_grouped_onnx_contract_accepts_required_inputs() -> None:
    meta = {
        "input_names": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"],
        "observation_shapes": {"prop": [10], "rbt_cmd_mf": [20], "smpl_cmd_mf": [30]},
    }

    validate_grouped_onnx_contract(["prop", "rbt_cmd_mf", "smpl_cmd_mf"], meta)
