import torch
from isaaclab.markers import VisualizationMarkers


#负责可视化，未人工复核
class MotionCommandDebugVisualizer:
    """Own all debug-visualization markers and rendering details for MotionCommand."""

    def __init__(self, cfg, command, device: str):
        self.cfg = cfg
        self.command = command
        self.device = device

        self.current_anchor_visualizer: VisualizationMarkers = None
        self.goal_anchor_visualizer: VisualizationMarkers = None
        self.future_anchor_visualizer: VisualizationMarkers = None
        self.current_anchor_lin_vel_visualizer: VisualizationMarkers = None
        self.goal_anchor_lin_vel_visualizer: VisualizationMarkers = None
        self.current_body_visualizers: list[VisualizationMarkers] = []
        self.goal_body_visualizers: list[VisualizationMarkers] = []

        """
        负责训练和验证过程中的可视化
        不改变任何 command 中的变量
        
        TODO 
        后续应加入 self.cfg 对应的Config类 而非直接复用 CommandCfg 
        """

    def set_enabled(self, debug_vis: bool) -> None:
        if debug_vis:
            self._ensure_initialized()
            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            self.future_anchor_visualizer.set_visibility(True)
            if self.cfg.debug_anchor_speed:
                self.current_anchor_lin_vel_visualizer.set_visibility(True)
                self.goal_anchor_lin_vel_visualizer.set_visibility(True)
            for visualizer in self.current_body_visualizers:
                visualizer.set_visibility(True)
            for visualizer in self.goal_body_visualizers:
                visualizer.set_visibility(True)
            return

        if self.current_anchor_visualizer is None:
            return

        self.current_anchor_visualizer.set_visibility(False)
        self.goal_anchor_visualizer.set_visibility(False)
        self.future_anchor_visualizer.set_visibility(False)
        if self.cfg.debug_anchor_speed:
            self.current_anchor_lin_vel_visualizer.set_visibility(False)
            self.goal_anchor_lin_vel_visualizer.set_visibility(False)
        for visualizer in self.current_body_visualizers:
            visualizer.set_visibility(False)
        for visualizer in self.goal_body_visualizers:
            visualizer.set_visibility(False)

    def render(self) -> None:
        if not self.command.robot.is_initialized or self.future_anchor_visualizer is None:
            return

        self.current_anchor_visualizer.visualize(
            self.command.robot_anchor_pos_w,
            self.command.robot_anchor_quat_w,
        )
        self.goal_anchor_visualizer.visualize(
            self.command.anchor_pos_w,
            self.command.anchor_quat_w,
        )
        self.future_anchor_visualizer.visualize(
            self.command.anchor_pos_w_future.view(-1, 3),
            self.command.anchor_quat_w_future.view(-1, 4),
        )

        if self.cfg.debug_anchor_speed:
            current_lin_vel_scale, current_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.command.robot_anchor_lin_vel_w,
                self.current_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )
            goal_lin_vel_scale, goal_lin_vel_quat = self._resolve_velocity_to_arrow(
                self.command.anchor_lin_vel_w,
                self.goal_anchor_lin_vel_visualizer.cfg.markers["arrow"].scale,
                self.cfg.debug_anchor_speed_scale,
            )

            self.current_anchor_lin_vel_visualizer.visualize(
                self.command.robot_anchor_pos_w,
                current_lin_vel_quat,
                current_lin_vel_scale,
            )
            self.goal_anchor_lin_vel_visualizer.visualize(
                self.command.anchor_pos_w,
                goal_lin_vel_quat,
                goal_lin_vel_scale,
            )

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(
                self.command.robot_body_pos_w[:, i],
                self.command.robot_body_quat_w[:, i],
            )
            self.goal_body_visualizers[i].visualize(
                self.command.body_pos_relative_w[:, i],
                self.command.body_quat_relative_w[:, i],
            )

    def _ensure_initialized(self) -> None:
        if self.current_anchor_visualizer is not None:
            return

        self.current_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
        )
        self.goal_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
        )
        self.future_anchor_visualizer = VisualizationMarkers(
            self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/future/anchor")
        )

        if self.cfg.debug_anchor_speed:
            self.current_anchor_lin_vel_visualizer = VisualizationMarkers(
                self.cfg.current_anchor_lin_vel_visualizer_cfg
            )
            self.goal_anchor_lin_vel_visualizer = VisualizationMarkers(
                self.cfg.goal_anchor_lin_vel_visualizer_cfg
            )

        self.current_body_visualizers = []
        self.goal_body_visualizers = []
        for name in self.cfg.body_names:
            self.current_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                )
            )
            self.goal_body_visualizers.append(
                VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                )
            )

    def _resolve_velocity_to_arrow(
        self,
        velocity_w: torch.Tensor,
        default_scale: tuple[float, float, float],
        speed_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 3D world velocity to arrow scale and world quaternion."""
        speed = torch.linalg.norm(velocity_w, dim=1)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= speed * speed_scale

        eps = 1.0e-8
        direction = velocity_w / speed.unsqueeze(-1).clamp(min=eps)
        x_axis = torch.zeros_like(direction)
        x_axis[:, 0] = 1.0

        cross = torch.cross(x_axis, direction, dim=1)
        dot = torch.sum(x_axis * direction, dim=1).clamp(-1.0, 1.0)

        w = torch.sqrt(((1.0 + dot).clamp(min=0.0)) * 0.5)
        xyz = cross / (2.0 * w.unsqueeze(-1).clamp(min=eps))
        arrow_quat_w = torch.cat([w.unsqueeze(-1), xyz], dim=1)

        opposite = dot < (-1.0 + 1.0e-6)
        if torch.any(opposite):
            arrow_quat_w[opposite] = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

        stationary = speed < 1.0e-6
        if torch.any(stationary):
            arrow_quat_w[stationary] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        arrow_quat_w = torch.nn.functional.normalize(arrow_quat_w, dim=1)
        return arrow_scale, arrow_quat_w
