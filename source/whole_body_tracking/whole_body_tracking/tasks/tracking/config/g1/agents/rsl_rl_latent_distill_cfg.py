from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import ActorShellCfg

ACTION_LOSS_TERM = {
    "name": "action",
    "kind": "supervised",
    "loss": "mse",
    "pred_key": "actions",
    "target_key": "privileged_actions",
    "weight": 1.0,
}


@configclass
class LatentVIBStudentCfg:
    class_name: str = "LatentVIBModel"
    state_groups: list[str] = ["prop"]
    target_groups: list[str] = ["rbt_cmd_mf"]
    latent_dim: int = 64
    posterior_hidden_dims: list[int] = [1024, 512, 256]
    prior_hidden_dims: list[int] = [512, 256]
    decoder_hidden_dims: list[int] = [1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    sample_latent_train: bool = True
    sample_latent_eval: bool = False


@configclass
class MLPStudentCfg:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = [1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict | None = None


@configclass
class ConfigurableDistillationAlgorithmCfg:
    class_name: str = "ConfigurableDistillation"
    num_learning_epochs: int = 5
    gradient_length: int = 15
    learning_rate: float = 1.0e-3
    max_grad_norm: float = 1.0
    loss_terms: list[dict] = [ACTION_LOSS_TERM]


@configclass
class LatentVIBDistillationAlgorithmCfg(ConfigurableDistillationAlgorithmCfg):
    loss_terms: list[dict] = [
        ACTION_LOSS_TERM,
        {
            "name": "kl",
            "kind": "output_mean",
            "pred_key": "kl",
            "weight": 1.0e-3,
            "schedule": {"type": "linear_warmup", "steps": 1000},
        },
    ]


@configclass
class G1FlatLatentDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 50
    max_iterations: int = 30000
    save_interval: int = 500
    experiment_name: str = "g1_latent_distill"
    empirical_normalization: bool = True
    logger: str = "tensorboard"
    seed: int = 42
    run_name: str = "latent_vib_student"

    algorithm = LatentVIBDistillationAlgorithmCfg()
    student = LatentVIBStudentCfg()
    teacher = ActorShellCfg()

    obs_groups = {
        "student": ["prop", "rbt_cmd_mf"],
        "teacher": ["prop", "rbt_cmd_mf", "smpl_cmd_mf"],
    }


@configclass
class G1FlatMLPDistillRunnerCfg(G1FlatLatentDistillRunnerCfg):
    experiment_name: str = "g1_mlp_distill"
    run_name: str = "mlp_student"

    algorithm = ConfigurableDistillationAlgorithmCfg()
    student = MLPStudentCfg()
