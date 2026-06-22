from __future__ import annotations

import json
import numpy as np

import onnx

G1_SUPPORTED_INPUTS = ("prop", "rbt_cmd_mf", "smpl_cmd_mf", "z")


def load_metadata(path: str) -> dict:
    model = onnx.load(path)
    out = {}
    for entry in model.metadata_props:
        try:
            out[entry.key] = json.loads(entry.value)
        except json.JSONDecodeError:
            out[entry.key] = entry.value
    return out


def onnx_input_names(path: str) -> list[str]:
    model = onnx.load(path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return [inp.name for inp in model.graph.input if inp.name not in initializer_names]


def validate_grouped_onnx_contract(input_names: list[str], meta: dict) -> None:
    if input_names == ["obs"]:
        raise ValueError(
            "This sim2sim script expects a grouped ONNX with inputs "
            f"drawn from {list(G1_SUPPORTED_INPUTS)}, but got old single-input ONNX ['obs']. "
            "Re-export with: play.py --export_onnx --encoder_mode=robot --export_only"
        )

    unsupported = [name for name in input_names if name not in G1_SUPPORTED_INPUTS]
    if unsupported:
        raise ValueError(
            f"This sim2sim script supports G1 grouped ONNX inputs {list(G1_SUPPORTED_INPUTS)}, "
            f"but ONNX inputs are {input_names}; unsupported {unsupported}."
        )
    if "prop" not in input_names:
        raise ValueError(f"This sim2sim script requires ONNX input 'prop', but inputs are {input_names}.")

    metadata_inputs = meta.get("input_names") or meta.get("input_groups") or meta.get("observation_groups")
    if metadata_inputs is not None:
        metadata_inputs = list(metadata_inputs)
        if metadata_inputs != input_names:
            raise ValueError(f"ONNX metadata input order {metadata_inputs} does not match graph inputs {input_names}.")

    shapes = meta.get("observation_shapes", {})
    missing_shapes = [name for name in input_names if name not in shapes]
    if missing_shapes:
        raise ValueError(
            f"ONNX metadata is missing observation_shapes for {missing_shapes}. "
            "Re-export with the grouped play.py ONNX exporter."
        )


class OnnxPolicy:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        available_providers = ort.get_available_providers()
        providers = [name for name in ("CUDAExecutionProvider", "CPUExecutionProvider") if name in available_providers]
        if not providers:
            providers = available_providers
        print(f"[INFO] ONNXRuntime providers: using={providers}, available={available_providers}")
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

    def run(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ort_inputs = {name: obs[name] for name in self.input_names}
        return self.session.run([self.output_name], ort_inputs)[0].astype(np.float32)
