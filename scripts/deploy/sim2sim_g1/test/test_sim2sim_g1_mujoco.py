from __future__ import annotations

import importlib.util
import numpy as np
import sys
import torch
from pathlib import Path
from tensordict import TensorDict

import onnx
from onnx import TensorProto, helper
from sim2sim_g1.mujoco_robot import action_to_target
from sim2sim_g1.observations import ImuReader, TermMajorHistory, prop_terms_from_metadata
from sim2sim_g1.onnx_policy import onnx_input_names, validate_grouped_onnx_contract
from sim2sim_g1.viewer import ReferenceMotionPlayer
from sim2sim_g1_mujoco import LatentInputSampler

REPO_ROOT = Path(__file__).resolve().parents[4]
RSL_RL_DIR = REPO_ROOT.parent / "rsl_rl"
if str(RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_DIR))
from rsl_rl.models import LatentVIBModel  # noqa: E402

EXPORTER_PATH = REPO_ROOT / "source" / "whole_body_tracking" / "whole_body_tracking" / "utils" / "exporter.py"
SPEC = importlib.util.spec_from_file_location("wbt_exporter", EXPORTER_PATH)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(exporter)
append_onnx_metadata = exporter.append_onnx_metadata
collect_observation_terms_metadata = exporter.collect_observation_terms_metadata
resolve_policy_input_shapes = exporter.resolve_policy_input_shapes
wrap_latent_vib_export_policy = exporter.wrap_latent_vib_export_policy


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


def test_collect_observation_terms_metadata_records_term_order_and_shapes() -> None:
    class TermCfg:
        def __init__(self, history_length=0, flatten_history_dim=True):
            self.history_length = history_length
            self.flatten_history_dim = flatten_history_dim

    class ObsManager:
        active_terms = {
            "rbt_cmd_mf": [
                "motion_joint_pos_multi_future",
                "motion_joint_vel_multi_future",
                "motion_anchor_ori_b_multi_future",
                "motion_anchor_z_multi_future",
            ],
            "prop": ["joint_pos"],
        }
        group_obs_term_dim = {
            "rbt_cmd_mf": [(319,), (319,), (66,), (11,)],
            "prop": [(290,)],
        }
        group_obs_concatenate = {"rbt_cmd_mf": True, "prop": True}
        _group_obs_term_cfgs = {
            "rbt_cmd_mf": [TermCfg(), TermCfg(), TermCfg(), TermCfg()],
            "prop": [TermCfg(history_length=10, flatten_history_dim=True)],
        }

    class Env:
        observation_manager = ObsManager()

    meta = collect_observation_terms_metadata(Env(), ["rbt_cmd_mf", "prop"])

    assert [term["name"] for term in meta["rbt_cmd_mf"]["terms"]] == [
        "motion_joint_pos_multi_future",
        "motion_joint_vel_multi_future",
        "motion_anchor_ori_b_multi_future",
        "motion_anchor_z_multi_future",
    ]
    assert meta["rbt_cmd_mf"]["terms"][-1]["shape"] == [11]
    assert meta["prop"]["terms"][0]["shape"] == [290]
    assert meta["prop"]["terms"][0]["base_shape"] == [29]
    assert meta["prop"]["terms"][0]["history_length"] == 10


def test_latent_vib_export_wrappers_expose_dynamic_input_groups() -> None:
    obs = TensorDict(
        {
            "prop": torch.randn(2, 5),
            "rbt_cmd_mf": torch.randn(2, 7),
        },
        batch_size=[2],
    )
    model = LatentVIBModel(
        obs,
        {"student": ["prop", "rbt_cmd_mf"]},
        "student",
        output_dim=4,
        latent_dim=3,
        posterior_hidden_dims=[8],
        prior_hidden_dims=[8],
        decoder_hidden_dims=[8],
        activation="elu",
        obs_normalization=True,
    ).eval()

    prior_policy = wrap_latent_vib_export_policy(model, "prior_prop")
    prior_obs = TensorDict({"prop": obs["prop"]}, batch_size=[2])
    prior_out = prior_policy(prior_obs)

    task_policy = wrap_latent_vib_export_policy(model, "task_posterior")
    task_out = task_policy(obs)

    prior_sample_policy = wrap_latent_vib_export_policy(model, "prior_sample")
    prior_sample_obs = TensorDict({"prop": obs["prop"], "z": torch.randn(2, 3)}, batch_size=[2])
    prior_sample_out = prior_sample_policy(prior_sample_obs)
    prior_sample_shapes = resolve_policy_input_shapes(
        prior_sample_policy, TensorDict({"prop": obs["prop"]}, batch_size=[2])
    )

    latent_input_policy = wrap_latent_vib_export_policy(model, "latent_input")
    latent_input_out = latent_input_policy(prior_sample_obs)

    assert prior_policy.obs_groups == ["prop"]
    assert task_policy.obs_groups == ["prop", "rbt_cmd_mf"]
    assert prior_sample_policy.obs_groups == ["prop", "z"]
    assert prior_sample_policy.extra_input_shapes == {"z": (3,)}
    assert prior_sample_shapes == {"prop": (5,), "z": (3,)}
    assert latent_input_policy.obs_groups == ["prop", "z"]
    assert prior_out["actions"].shape == (2, 4)
    assert task_out["actions"].shape == (2, 4)
    assert prior_sample_out["actions"].shape == (2, 4)
    assert latent_input_out["actions"].shape == (2, 4)


def test_latent_input_sampler_resamples_and_holds() -> None:
    sampler = LatentInputSampler(
        ["prop", "z"],
        {"observation_shapes": {"z": [3]}},
        mean=0.0,
        std=1.0,
        seed=123,
        resample_interval=2,
        sample_mode="normal",
    )
    obs = {"prop": np.zeros((1, 5), dtype=np.float32)}

    sampler.add_to_obs(obs, step=0)
    z0 = obs["z"].copy()
    sampler.add_to_obs(obs, step=1)
    z1 = obs["z"].copy()
    sampler.add_to_obs(obs, step=2)
    z2 = obs["z"].copy()

    assert z0.shape == (1, 3)
    assert np.array_equal(z0, z1)
    assert not np.array_equal(z1, z2)


def test_latent_input_sampler_zero_mode() -> None:
    sampler = LatentInputSampler(
        ["prop", "z"],
        {"observation_shapes": {"z": [3]}},
        mean=1.0,
        std=1.0,
        seed=123,
        resample_interval=1,
        sample_mode="zero",
    )
    obs = {"prop": np.zeros((1, 5), dtype=np.float32)}

    sampler.add_to_obs(obs, step=0)

    assert np.array_equal(obs["z"], np.zeros((1, 3), dtype=np.float32))
