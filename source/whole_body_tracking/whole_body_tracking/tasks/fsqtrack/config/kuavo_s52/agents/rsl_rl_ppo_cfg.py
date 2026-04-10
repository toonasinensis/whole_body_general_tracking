from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

# Command (s^g) dimension for S52 flat env: command_horizon * 2 * num_joints = 10 * 2 * 27 = 540.
# The runner builds the policy with eval("ActorCriticFSQVAE") from rsl_rl.modules.
ACTOR_SG_DIM_S52 = 540


@configclass
class RslRlFsqVaeActorCriticCfg:
    """Policy config for ActorCriticFSQVAE (FSQ-VAE actor + MLP critic)."""

    class_name: str = "ActorCriticFSQVAE"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: list = [512, 256, 128]
    critic_hidden_dims: list = [512, 256, 128]
    activation: str = "elu"
    actor_sg_dim: int = ACTOR_SG_DIM_S52

    # NOTE the hyper params for FSQ-VAE are not tuned yet
    fsqvae_latent_dim: int = 32
    fsq_levels: list = [8, 5, 5, 5]
    num_codebooks: int = 1
    robot_encoder_hidden_dims: list = [512, 256, 128]
    recover_decoder_hidden_dims: list = [128, 256, 512]


@configclass
class RslRlFsqVaePpoAlgorithmCfg:
    """Algorithm config for FSQVAE_PPO (PPO + reconstruction loss)."""

    class_name: str = "FSQVAE_PPO"
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
    reconstruction_loss_coef: float = 1.0


@configclass
class FsqtrackS52FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "fsqtrackS52_flat"
    empirical_normalization = True

    policy = RslRlFsqVaeActorCriticCfg()
    algorithm = RslRlFsqVaePpoAlgorithmCfg()

LOW_FREQ_SCALE = 0.5


@configclass
class FsqtrackS52FlatLowFreqPPORunnerCfg(FsqtrackS52FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
