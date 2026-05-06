from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RobanS22FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for Isaac Lab + rsl_rl `ActorCritic` (Isaac Sim 4.x style).

    Uses `RslRlPpoActorCriticCfg` so `to_dict()` yields `policy.class_name == "ActorCritic"`.
    The newer split `actor`/`critic` + `MLPModel` layout is for a different rsl_rl stack.
    """

    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "roban_flat"
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[4096, 2048, 1024, 512, 256],
        critic_hidden_dims=[4096, 2048, 1024, 512, 256],
        activation="elu",
        noise_std_type="scalar",
    )
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
