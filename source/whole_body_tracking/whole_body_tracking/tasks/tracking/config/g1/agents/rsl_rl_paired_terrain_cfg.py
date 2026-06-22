from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from .rsl_rl_ppo_cfg import CriticCfg, MyPpoAlgorithmCfg


@configclass
class PairedTerrainActorShellCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MyModel",
        "main_encoder": "encoder_g1",
        "encoder": {
            "encoder_g1": {
                "encoder_groups": ["rbt_cmd_mf"],
                "hidden_dims": [2048, 1024, 512, 512],
                "activation": "swish",
            },
        },
        "loss": {
            "recon": 0.010,
        },
        "fsq": {"num_fsq_levels": 32, "fsq_level_list": 32, "max_num_tokens": 2},
        "latent_dim": 128,
        "decoder": {
            "action_decoder": {
                "decoder_groups": ["prop", "terrain"],
                "hidden_dims": [2048, 1024, 512, 512],
                "activation": "swish",
                "detach_latent": True,
                "outputs": ["actions"],
            },
            "g1_kin_decoder": {
                "decoder_groups": [],
                "hidden_dims": [4096, 2048, 1024, 512, 256],
                "activation": "swish",
                "detach_latent": False,
                "outputs": ["rbt_cmd_mf"],
            },
        },
        "activation": "swish",
        "obs_normalization": True,
    }


@configclass
class G1PairedTerrainHeightScanRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_paired_terrain_height_scan"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()
    actor = PairedTerrainActorShellCfg()
    critic = CriticCfg()

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf", "smpl_cmd_mf", "terrain"],
        "critic": ["critic", "terrain"],
    }
