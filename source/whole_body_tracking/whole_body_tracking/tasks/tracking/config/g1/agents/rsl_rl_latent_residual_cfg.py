from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_latent_distill_cfg import LatentVIBStudentCfg


@configclass
class LatentResidualActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 0.22,
        "std_type": "scalar",
        "learnable_std": False,
    }
    backbone: dict = {
        "class_name": "MLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class LatentResidualCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list[int] = [1024, 512, 256]
    activation: str = "swish"
    obs_normalization: bool = True
    distribution_cfg: dict | None = None


@configclass
class LatentResidualPPOAlgorithmCfg:
    class_name: str = "LatentResidualPPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.005
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    min_policy_std: list[float] | float | None = None
    primitive_checkpoint_path: str | None = None
    primitive_load_strict: bool = True
    plugins: list = []


@configclass
class LatentPrimitiveCfg(LatentVIBStudentCfg):
    pass


@configclass
class G1FlatLatentResidualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env: int = 24
    max_iterations: int = 300000
    save_interval: int = 500
    experiment_name: str = "g1_latent_residual_ppo"
    empirical_normalization: bool = True
    logger: str = "tensorboard"
    seed: int = 42
    run_name: str = "latent_residual"

    algorithm = LatentResidualPPOAlgorithmCfg()
    actor = LatentResidualActorCfg()
    critic = LatentResidualCriticCfg()
    primitive = LatentPrimitiveCfg()

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf"],
        "critic": ["critic"],
        "primitive": ["prop", "rbt_cmd_mf"],
    }
