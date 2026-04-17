from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ActorCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512, 256]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


@configclass
class CriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512, 256]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class RobanS22FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 100
    experiment_name = "roban_flat"
    empirical_normalization = True

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    actor = ActorCfg()

    critic = CriticCfg()

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
