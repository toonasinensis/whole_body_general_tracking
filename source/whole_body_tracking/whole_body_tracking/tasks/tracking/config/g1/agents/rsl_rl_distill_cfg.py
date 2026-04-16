from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class ActorCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


@configclass
class CriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class RslRlDistillAlgorithmCfg:
    class_name: str = "Distillation"
    num_learning_epochs = (5,)
    learning_rate = (1.0e-3,)
    max_grad_norm = (1.0,)


# def __init__(
#         self,
#         student: MLPModel,
#         teacher: MLPModel,
#         storage: RolloutStorage,
#         num_learning_epochs: int = 1,
#         gradient_length: int = 15,
#         learning_rate: float = 1e-3,
#         max_grad_norm: float | None = None,
#         loss_type: str = "mse",
#         optimizer: str = "adam",
#         device: str = "cpu",
#         # Distributed training parameters
#         multi_gpu_cfg: dict | None = None,
#         **kwargs: dict,  # handle unused config parameters
#     ) -> None:


@configclass
class RslRlDistillRunnerCfg:
    num_steps_per_env = 50
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True

    algorithm = RslRlDistillAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
    )
    logger = "tensorboard"
    seed = 42
    run_name = "distill_student"
    student = ActorCfg()

    teacher = ActorCfg()

    obs_groups = {
        "student": ["policy"],
        "teacher": ["policy"],
    }
