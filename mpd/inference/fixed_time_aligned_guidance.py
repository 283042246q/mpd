"""Phase-4 aligned fixed-time dynamic guidance without changing diffusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mpd.inference.dynamic_risk import mean_cvar_dynamic_risk
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
)


@dataclass(frozen=True)
class FixedTimeAlignedGuidanceSettings:
    dynamic_collision_alpha: float = 0.5
    dynamic_collision_cvar_fraction: float = 0.10
    dynamic_max_grad_norm: float = 1.0
    dynamic_scale: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.dynamic_collision_alpha <= 1.0:
            raise ValueError("dynamic_collision_alpha must lie in [0, 1]")
        if not 0.0 < self.dynamic_collision_cvar_fraction <= 1.0:
            raise ValueError(
                "dynamic_collision_cvar_fraction must lie in (0, 1]"
            )
        if self.dynamic_max_grad_norm <= 0.0:
            raise ValueError("dynamic_max_grad_norm must be positive")
        if self.dynamic_scale < 0.0:
            raise ValueError("dynamic_scale must be non-negative")


def _clip_per_candidate(gradient: torch.Tensor, max_norm: float):
    flat = gradient.flatten(start_dim=1)
    norm = torch.linalg.norm(flat, dim=-1)
    scale = (float(max_norm) / norm.clamp_min(torch.finfo(norm.dtype).eps)).clamp(
        max=1.0
    )
    return gradient * scale.reshape(-1, *([1] * (gradient.ndim - 1))), norm


class FixedTimeDynamicRiskEvaluator:
    def __init__(
        self,
        planning_task,
        dataset,
        dynamic_world,
        dynamic_field,
        settings: FixedTimeAlignedGuidanceSettings,
    ) -> None:
        self.planning_task = planning_task
        self.dataset = dataset
        self.dynamic_world = dynamic_world
        self.dynamic_field = dynamic_field
        self.settings = settings

    def evaluate_q(self, q: torch.Tensor):
        batch, horizon, _ = q.shape
        poses = self.planning_task.robot.fk_collision_spheres(
            q.reshape(batch * horizon, -1)
        )
        poses = torch.stack(poses).transpose(0, 1).reshape(
            batch, horizon, -1, 3, 4
        )
        positions = link_pos_from_link_tensor(poses)[..., :3]
        minimum_distance = self.dynamic_world.minimum_signed_distance(positions)
        margins = (
            self.dynamic_field.collision_margins.to(
                dtype=q.dtype,
                device=q.device,
            )
            + self.dynamic_field.cutoff_margin
        )
        penetration = torch.relu(margins - minimum_distance)
        times = torch.linspace(
            0.0,
            self.dynamic_world.trajectory_duration_s,
            horizon,
            dtype=q.dtype,
            device=q.device,
        ).expand(batch, -1)
        return mean_cvar_dynamic_risk(
            penetration,
            times,
            alpha=self.settings.dynamic_collision_alpha,
            cvar_fraction=self.settings.dynamic_collision_cvar_fraction,
        )

    def evaluate_control_points(self, control_points_normalized: torch.Tensor):
        control_points = self.dataset.unnormalize_control_points(
            control_points_normalized
        )
        trajectory = self.planning_task.parametric_trajectory.get_q_trajectory(
            control_points,
            None,
            None,
            get_type=("pos",),
            get_time_representation=False,
        )
        return self.evaluate_q(trajectory["pos"])


class FixedTimeAlignedGuide:
    """Keep static guidance intact and replace only its dynamic aggregation."""

    def __init__(
        self,
        spatial_guide,
        risk_evaluator: FixedTimeDynamicRiskEvaluator,
        dynamic_field,
        *,
        dynamic_collision_weight: float,
        settings: FixedTimeAlignedGuidanceSettings,
    ) -> None:
        self.spatial_guide = spatial_guide
        self.risk_evaluator = risk_evaluator
        self.dynamic_field = dynamic_field
        self.dynamic_collision_weight = float(dynamic_collision_weight)
        self.settings = settings
        self.statistics: list[dict[str, float]] = []

    @property
    def guidance_profiler(self):
        return self.spatial_guide.guidance_profiler

    def use_all_collision_objects(self):
        return self.spatial_guide.use_all_collision_objects()

    def warmup(self, shape_x):
        return self.spatial_guide.warmup(shape_x)

    def _static_spatial_descent(self, control_points_normalized, **kwargs):
        collision_entry = self.spatial_guide.costs.get(
            "CostTaskSpaceCollisionObjects"
        )
        original_field = None
        if collision_entry is not None:
            original_field = collision_entry.cost.collision_objects_field
            collision_entry.cost.collision_objects_field = self.dynamic_field.static_field
        try:
            return self.spatial_guide(control_points_normalized, **kwargs)
        finally:
            if collision_entry is not None:
                collision_entry.cost.collision_objects_field = original_field

    def __call__(
        self,
        control_points_normalized,
        return_cost=False,
        warmup=False,
        **kwargs,
    ):
        static_output = self._static_spatial_descent(
            control_points_normalized,
            return_cost=return_cost,
            warmup=warmup,
            **kwargs,
        )
        if return_cost:
            static_cost, static_descent = static_output
        else:
            static_cost = None
            static_descent = static_output
        with torch.enable_grad():
            spatial = control_points_normalized.detach().requires_grad_(True)
            dynamic_risk, breakdown = self.risk_evaluator.evaluate_control_points(
                spatial
            )
            if dynamic_risk.requires_grad:
                dynamic_gradient = torch.autograd.grad(
                    self.dynamic_collision_weight * dynamic_risk.sum(),
                    spatial,
                )[0]
            else:
                dynamic_gradient = torch.zeros_like(spatial)
        dynamic_gradient, gradient_norm = _clip_per_candidate(
            dynamic_gradient,
            self.settings.dynamic_max_grad_norm,
        )
        descent = static_descent - self.settings.dynamic_scale * dynamic_gradient
        if not warmup:
            self.statistics.append(
                {
                    "dynamic_collision_mean": float(
                        breakdown["mean"].mean().detach().cpu()
                    ),
                    "dynamic_collision_cvar": float(
                        breakdown["cvar"].mean().detach().cpu()
                    ),
                    "dynamic_gradient_norm_mean": float(
                        gradient_norm.mean().detach().cpu()
                    ),
                }
            )
        if return_cost:
            total_cost = (
                static_cost
                + self.dynamic_collision_weight * dynamic_risk.detach()
            )
            return total_cost, descent
        return descent
