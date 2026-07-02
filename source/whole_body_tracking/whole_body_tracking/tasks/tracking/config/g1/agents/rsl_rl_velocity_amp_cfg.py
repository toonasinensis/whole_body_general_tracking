from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import MyPpoAlgorithmCfg
from whole_body_tracking.tasks.tracking.config.g1.velocity_amp_env_cfg import G1_AMP_ANCHOR_BODY_NAME, G1_AMP_BODY_NAMES

_WBT_ROOT = Path(__file__).resolve().parents[8]
G1_AMP_MOTION_DIR = str(_WBT_ROOT / "data/mesh_flatwbc_model151000_z10cm/tracking_npz_data_amp")
# Keep the legacy constant name because mixed/heading AMP configs import it.


@configclass
class VelocityModalTransformerActorCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MyMLPModel",
        "hidden_dims": [1024, 512, 256],
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class VelocityModalCriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [1024, 512]
    activation: str = "relu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class G1VelocityFlatAMPModalRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000
    save_interval = 200
    experiment_name = "g1_velocity_flat_amp_modal"
    run_name = "isaaclab_flat_velocity_amp_modal"
    empirical_normalization = True

    actor = VelocityModalTransformerActorCfg()
    critic = VelocityModalCriticCfg()

    obs_groups = {
        "actor": ["prop", "base_velocity_cmd"],
        "critic": ["policy"],
    }

    algorithm = MyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        min_policy_std=0.055,
        plugins=[
            {
                "class_name": "AMPPlugin",
                "amp_reward_coef": 0.05,
                "amp_motion_files": G1_AMP_MOTION_DIR,
                "amp_task_reward_lerp": 0.9,
                "amp_discr_hidden_dims": [1024, 512],
                "amp_replay_buffer_size": 100000,
                "amp_body_names": G1_AMP_BODY_NAMES,
                "amp_anchor_name": G1_AMP_ANCHOR_BODY_NAME,
                "amp_use_height_scan": False,
            }
        ],
    )
