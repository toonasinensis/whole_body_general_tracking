from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DeparseActorCfg:
    class_name: str = "StochasticWrapper"
    backbone: dict = {
        "class_name": "BackboneMLP",
        "hidden_dims": [4096, 2048, 1024, 512, 256],
        "activation": "elu",
        "obs_normalization": True,
    }
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }


@configclass
class DeparseCriticCfg:
    class_name: str = "BackboneMLP"
    hidden_dims: list[int] = [4096, 2048, 1024, 512, 256]
    activation: str = "elu"
    obs_normalization: bool = True


@configclass
class DeparsePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "DeparsePPO"
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
    optimizer: str = "adam"
    normalize_advantage_per_mini_batch: bool = False
    share_cnn_encoders: bool = False
    rnd_cfg: dict | None = None
    symmetry_cfg: dict | None = None


@configclass
class RobanS22FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config matching `rsl_rl.runners.deparse_policy_runner.OnPolicyRunner`."""

    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "roban_flat"
    empirical_normalization = True

    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }
    actor = DeparseActorCfg()
    critic = DeparseCriticCfg()
    algorithm = DeparsePpoAlgorithmCfg()
