import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        policy_path = path.split("model")[0]
        filename = policy_path.split("/")[-2] + ".onnx"
        # export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
        # run_path = os.path.basename(os.path.normpath(policy_path))
        # attach_onnx_metadata(self.env.unwrapped, run_path, path=policy_path, filename=filename)


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    # def save(self, path: str, infos=None):
    #     """Save the model and training information."""
    #     super().save(path, infos)
    #     policy_path = path.split("model")[0]
    #     filename = policy_path.split("/")[-2] + ".onnx"
    #     export_motion_policy_as_onnx(
    #         self.env.unwrapped, self.alg.get_policy(), normalizer=self.obs_normalizer, path=policy_path, filename=filename
    #     )
    #     run_path = os.path.basename(os.path.normpath(policy_path))
    #     attach_onnx_metadata(self.env.unwrapped, run_path, path=policy_path, filename=filename)
