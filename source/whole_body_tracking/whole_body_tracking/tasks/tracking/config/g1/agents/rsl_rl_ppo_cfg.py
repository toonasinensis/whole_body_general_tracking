from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class ActorCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [4096, 2048, 1024, 512, 256]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}


@configclass
class CriticCfg:
    class_name: str = "MLPModel"
    hidden_dims: list = [4096, 2048, 1024, 512, 256]
    activation: str = "elu"
    obs_normalization: bool = True
    distribution_cfg: dict = None


@configclass
class ActorShellCfg:
    class_name: str = "ActorModel"
    distribution_cfg: dict = {
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
    }
    backbone: dict = {
        "class_name": "MLPModel",
        "hidden_dims": [4096, 2048, 1024, 512, 256],
        "activation": "elu",
        "obs_normalization": True,
    }


# @configclass
# class ActorShellCfg:
#     class_name: str = "ActorModel"
#     distribution_cfg: dict = {
#         "class_name": "GaussianDistribution",
#         "init_std": 1.0,
#         "std_type": "scalar",
#     }


#     backbone: dict = {
#            "class_name": "rsl_rl.models:SubEncoderMLPModel",

#             "encoder" : {
#                 "encoder_g1_cfg": {
#                 "encoder_groups": ["rbt_cmd_mf"],  #  输入的 encoder_groups
#                 "encoder_hidden_dims":   [2048, 1024, 512, 512],         # ecoder MLP 隐层
#                 "activation":        "elu",
#                 },

#                 "encoder_smpl_cfg": {
#                 "encoder_groups": ["rbt_cmd_mf"],  # 输入的 encoder_groups
#                 "encoder_hidden_dims":   [2048, 1024, 512, 512],         # ecoder MLP 隐层
#                 "activation":        "elu",
#                 },
#             },

#             "latent_dim" : 32,  # 所有的 encoder 和 decoder 共用
#             "token_loss_weight": 0.0,  # token loss: MSE(z_g1, z_smpl)，0.0 = 关闭

#             "decoder":{
#                 "action_decoder_cfg": {   # action decoder只负责输出action,输入latent和prop
#                 "input_prop_groups": ["prop"],  # decoder 输入中哪些是 passthrough 的原始 obs（默认为 []，即全 latent 输入）
#                 "hidden_dims":   [2048, 1024, 512, 512],         # decoder MLP 隐层
#                 "detach_latent": True,             # True = decoder 梯度不回传 encoder
#                 },

#                 "g1_kin_decoder_cfg": {   # g1_kin_decoder 负责重建 rbt_cmd_mf
#                 "input_prop_groups": [],  # decoder 输入中哪些是 passthrough 的原始 obs（默认为 []，即全 latent 输入）
#                 "hidden_dims":   [2048, 1024, 512, 512],         # decoder MLP 隐层
#                 "detach_latent": True,             # True = decoder 梯度不回传 encoder
#                 "recon_weight":  1.0,              # 重建 loss 权重: MSE(decoder(z), rbt_cmd_mf) * recon_weight
#                 "cycle_loss_weight": 1.0,          # 循环 loss 权重: MSE(encoder_g1(recon.detach()), z_g1.detach()) * weight，0.0 = 关闭
#                 },
#             },
#         }


@configclass
class MyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
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


@configclass
class G1FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_flat"
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


@configclass
class G1FlatFMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000000000000000
    save_interval = 500
    experiment_name = "g1_flat"
    empirical_normalization = True

    algorithm = MyPpoAlgorithmCfg()

    actor = ActorShellCfg()

    critic = CriticCfg()

    obs_groups = {
        "actor": ["prop", "rbt_cmd_mf"],
        "critic": ["critic"],
    }
