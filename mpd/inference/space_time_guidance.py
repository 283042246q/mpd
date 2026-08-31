"""Inference-only joint space-time guidance for Phase 5.

The diffusion model continues to update normalized spatial control points.
Timing control points are low-dimensional optimizer state owned by this guide.
Dynamic collision cost is evaluated once and differentiated with respect to
both variables from the same autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from mpd.inference.dynamic_risk import mean_cvar_dynamic_risk
from mpd.parametric_trajectory.timing_spline import (
    TimingSpline,
    TimingSplineEvaluation,
)
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
)


SPACE_TIME_MODES = {
    "phase5_scalar_duration",
    "phase5_timing_only",
    "phase5_joint",
}

DURATION_COST_NORMALIZER_S = 10.0

FIXED_TIME_KINEMATIC_COSTS = (
    "CostJointSpaceVelocity",
    "CostJointSpaceAcceleration",
)


@dataclass(frozen=True)
class SpaceTimeGuidanceSettings:
    mode: str = "phase5_joint"
    num_timing_control_points: int = 8
    timing_degree: int = 3
    u_min: float = 0.05
    duration_min: float = 6.0
    duration_max: float = 14.0
    nominal_duration: float = 10.0
    timing_learning_rate: float = 0.08
    timing_beta1: float = 0.9
    timing_beta2: float = 0.999
    timing_epsilon: float = 1e-8
    timing_max_grad_norm: float = 1.0
    spatial_dynamic_max_grad_norm: float = 2.0
    spatial_dynamic_scale: float = 1.0
    dynamic_collision_weight: float = 10.0
    dynamic_collision_alpha: float = 0.5
    dynamic_collision_cvar_fraction: float = 0.10
    velocity_weight: float = 0.02
    acceleration_weight: float = 0.005
    duration_weight: float = 0.2
    timing_smoothness_weight: float = 0.02
    collision_power: float = 2.0

    @classmethod
    def from_mapping(cls, values: Any, *, mode: str | None = None):
        mapping = dict(values or {})
        if mode is not None:
            mapping["mode"] = mode
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(f"unknown space-time guidance settings: {unknown}")
        settings = cls(**mapping)
        if settings.mode not in SPACE_TIME_MODES:
            raise ValueError(f"unsupported space-time guidance mode {settings.mode!r}")
        if not settings.duration_min < settings.nominal_duration < settings.duration_max:
            raise ValueError("nominal duration must lie strictly inside duration bounds")
        if settings.timing_learning_rate <= 0.0:
            raise ValueError("timing_learning_rate must be positive")
        if settings.spatial_dynamic_max_grad_norm <= 0.0:
            raise ValueError("spatial_dynamic_max_grad_norm must be positive")
        if not 0.0 <= settings.dynamic_collision_alpha <= 1.0:
            raise ValueError("dynamic collision alpha must lie in [0, 1]")
        if not 0.0 < settings.dynamic_collision_cvar_fraction <= 1.0:
            raise ValueError("dynamic collision CVaR fraction must lie in (0, 1]")
        return settings


@dataclass(frozen=True)
class SpaceTimeTrajectoryState:
    """One candidate batch's shared spatial and timing evaluation state."""

    q: torch.Tensor
    q_s: torch.Tensor
    q_ss: torch.Tensor
    timing: TimingSplineEvaluation
    collision_sphere_positions: torch.Tensor | None = None
    collision_sphere_poses: torch.Tensor | None = None
    collision_sphere_jacobians: torch.Tensor | None = None


def _clip_per_candidate(gradient: torch.Tensor, max_norm: float):
    flat = gradient.flatten(start_dim=1)
    norm = torch.linalg.norm(flat, dim=-1)
    scale = (float(max_norm) / norm.clamp_min(torch.finfo(norm.dtype).eps)).clamp(max=1.0)
    clipped = gradient * scale.reshape(-1, *([1] * (gradient.ndim - 1)))
    return clipped, norm, scale < 1.0


def _gradient_cosine_per_candidate(
    static_gradient: torch.Tensor,
    dynamic_gradient: torch.Tensor,
):
    """Return gradient cosine and a mask excluding undefined zero directions."""
    static_flat = static_gradient.flatten(start_dim=1)
    dynamic_flat = dynamic_gradient.flatten(start_dim=1)
    static_norm = torch.linalg.norm(static_flat, dim=-1)
    dynamic_norm = torch.linalg.norm(dynamic_flat, dim=-1)
    epsilon = torch.finfo(static_flat.dtype).eps
    valid = (static_norm > epsilon) & (dynamic_norm > epsilon)
    denominator = (static_norm * dynamic_norm).clamp_min(epsilon)
    cosine = (static_flat * dynamic_flat).sum(dim=-1) / denominator
    return cosine.clamp(min=-1.0, max=1.0), valid


class SpaceTimeCostEvaluator:
    """Differentiable dynamic, kinematic and timing costs."""

    def __init__(
        self,
        timing_spline: TimingSpline,
        dynamic_world,
        *,
        collision_margins: torch.Tensor,
        cutoff_margin: float,
        velocity_limits: torch.Tensor | None,
        acceleration_limits: torch.Tensor | None,
        settings: SpaceTimeGuidanceSettings,
    ) -> None:
        self.timing_spline = timing_spline
        self.dynamic_world = dynamic_world
        self.collision_margins = collision_margins
        self.cutoff_margin = float(cutoff_margin)
        self.velocity_limits = velocity_limits
        self.acceleration_limits = acceleration_limits
        self.settings = settings

    def __call__(
        self,
        timing_control_points: torch.Tensor | None = None,
        *,
        q: torch.Tensor | None = None,
        q_s: torch.Tensor | None = None,
        q_ss: torch.Tensor | None = None,
        collision_sphere_positions: torch.Tensor | None = None,
        trajectory_state: SpaceTimeTrajectoryState | None = None,
    ):
        if trajectory_state is None:
            if timing_control_points is None:
                raise ValueError("timing_control_points or trajectory_state is required")
            evaluation = self.timing_spline.evaluate(
                timing_control_points,
                q=q,
                q_s=q_s,
                q_ss=q_ss,
                require_fixed_endpoint_derivatives=True,
            )
        else:
            if any(value is not None for value in (timing_control_points, q, q_s, q_ss)):
                raise ValueError(
                    "trajectory_state cannot be combined with timing or spatial inputs"
                )
            evaluation = trajectory_state.timing
            q = trajectory_state.q
            collision_sphere_positions = trajectory_state.collision_sphere_positions
        if collision_sphere_positions is None:
            raise ValueError("collision_sphere_positions are required")
        minimum_distance = self.dynamic_world.minimum_signed_distance(
            collision_sphere_positions,
            trajectory_times=evaluation.time_from_start,
        )
        margins = self.collision_margins.to(
            dtype=q.dtype, device=q.device
        ) + self.cutoff_margin
        penetration = torch.relu(margins - minimum_distance)
        if self.settings.collision_power != 2.0:
            raise ValueError("mean/CVaR dynamic risk currently requires collision_power=2")
        candidate_duration = evaluation.duration
        inverse_duration = candidate_duration.clamp_min(
            torch.finfo(candidate_duration.dtype).eps
        ).reciprocal()
        # Compare the time-distributed penalty using its physical-time mean
        # and worst-time tail. Candidate makespan is charged exactly once by
        # the explicit duration term.
        dynamic_collision, dynamic_breakdown = mean_cvar_dynamic_risk(
            penetration,
            evaluation.time_from_start,
            alpha=self.settings.dynamic_collision_alpha,
            cvar_fraction=self.settings.dynamic_collision_cvar_fraction,
        )
        dynamic_collision_mean = dynamic_breakdown["mean"]
        dynamic_collision_cvar = dynamic_breakdown["cvar"]

        velocity = torch.zeros_like(dynamic_collision)
        if self.velocity_limits is not None:
            utilization = torch.abs(evaluation.dq) / self.velocity_limits
            velocity_density = torch.relu(utilization - 1.0).square().sum(dim=-1)
            velocity = torch.trapezoid(
                velocity_density * evaluation.u, evaluation.phase, dim=-1
            ) * inverse_duration
        acceleration = torch.zeros_like(dynamic_collision)
        if self.acceleration_limits is not None:
            utilization = torch.abs(evaluation.ddq) / self.acceleration_limits
            acceleration_density = torch.relu(utilization - 1.0).square().sum(dim=-1)
            acceleration = torch.trapezoid(
                acceleration_density * evaluation.u, evaluation.phase, dim=-1
            ) * inverse_duration

        duration_cost = candidate_duration / DURATION_COST_NORMALIZER_S
        timing_smoothness = torch.trapezoid(
            (evaluation.u_s / evaluation.u).square(), evaluation.phase, dim=-1
        )
        breakdown = {
            "dynamic_collision": dynamic_collision,
            "dynamic_collision_mean": dynamic_collision_mean,
            "dynamic_collision_cvar": dynamic_collision_cvar,
            "velocity": velocity,
            "acceleration": acceleration,
            "duration": duration_cost,
            "timing_smoothness": timing_smoothness,
        }
        total = (
            self.settings.dynamic_collision_weight * dynamic_collision
            + self.settings.velocity_weight * velocity
            + self.settings.acceleration_weight * acceleration
            + self.settings.duration_weight * duration_cost
            + self.settings.timing_smoothness_weight * timing_smoothness
        )
        return total, breakdown, evaluation


class InferenceOnlySpaceTimeGuide:
    """Wrap the existing spatial guide and own the timing optimizer state."""

    def __init__(
        self,
        spatial_guide,
        planning_task,
        dataset,
        dynamic_field,
        settings: SpaceTimeGuidanceSettings,
        tensor_args,
        reuse_spatial_kinematics_enabled: bool = True,
    ) -> None:
        trajectory = planning_task.parametric_trajectory
        if not hasattr(trajectory, "bspline"):
            raise ValueError("Phase-5 timing guidance requires a spatial B-spline trajectory")
        self.spatial_guide = spatial_guide
        self.planning_task = planning_task
        self.dataset = dataset
        self.dynamic_field = dynamic_field
        self.settings = settings
        self.tensor_args = dict(tensor_args)
        self.reuse_spatial_kinematics_enabled = bool(
            reuse_spatial_kinematics_enabled
        )
        self.timing_spline = TimingSpline(
            num_control_points=settings.num_timing_control_points,
            degree=settings.timing_degree,
            num_phase_points=int(trajectory.num_T_pts),
            u_min=settings.u_min,
            duration_min=settings.duration_min,
            duration_max=settings.duration_max,
            tensor_args=self.tensor_args,
        )
        robot = planning_task.robot
        self.cost_evaluator = SpaceTimeCostEvaluator(
            self.timing_spline,
            dynamic_field.dynamic_world,
            collision_margins=dynamic_field.collision_margins,
            cutoff_margin=dynamic_field.cutoff_margin,
            velocity_limits=robot.dq_max,
            acceleration_limits=robot.ddq_max,
            settings=settings,
        )
        self.timing_control_points = None
        self._timing_momentum = None
        self._timing_variance = None
        self._optimizer_step = 0
        self.statistics = []

    @property
    def guidance_profiler(self):
        return self.spatial_guide.guidance_profiler

    def use_all_collision_objects(self):
        return self.spatial_guide.use_all_collision_objects()

    def warmup(self, shape_x):
        # The resident dynamic engine performs an explicit Phase-5 warmup after
        # a world and plan-start context exist.
        return self.spatial_guide.warmup(shape_x)

    def reset(self, candidate_count: int) -> None:
        candidate_count = int(candidate_count)
        base = self.timing_spline.linear_control_points(
            self.settings.nominal_duration, batch_shape=(candidate_count,)
        )
        if self.settings.mode != "phase5_scalar_duration":
            # Fixed population allocation: linear, shorter, longer, slow-zone.
            quarter = max(1, candidate_count // 4)
            base[quarter : 2 * quarter, 2:-2] -= 0.35
            base[2 * quarter : 3 * quarter, 2:-2] += 0.35
            slow = base[3 * quarter :]
            if slow.numel():
                midpoint = (slow.shape[-1] - 4) // 2
                slow[..., 2 + midpoint : -2] += 0.6
            base = self.timing_spline.enforce_endpoint_derivatives(base)
        self.timing_control_points = base.detach()
        self._timing_momentum = torch.zeros_like(base)
        self._timing_variance = torch.zeros_like(base)
        self._optimizer_step = 0
        self.statistics = []

    def _phase_trajectory(self, control_points_normalized, timing_control_points):
        trajectory = self.planning_task.parametric_trajectory
        inner = self.dataset.unnormalize_control_points(control_points_normalized)
        full = trajectory.augment_control_points_fn(inner, None, None).clone()
        timing = self.timing_spline.evaluate(timing_control_points)

        q_start = trajectory.q_pos_start
        q_goal = trajectory.q_pos_goal
        q_vel_start = torch.zeros_like(q_start) if trajectory.q_vel_start is None else trajectory.q_vel_start
        q_vel_goal = torch.zeros_like(q_goal) if trajectory.q_vel_goal is None else trajectory.q_vel_goal
        q_acc_start = torch.zeros_like(q_start) if trajectory.q_acc_start is None else trajectory.q_acc_start
        q_acc_goal = torch.zeros_like(q_goal) if trajectory.q_acc_goal is None else trajectory.q_acc_goal
        targets = (
            (
                0,
                q_start,
                timing.u[..., 0, None] * q_vel_start,
                timing.u[..., 0, None].square() * q_acc_start
                + timing.u_s[..., 0, None] * q_vel_start,
            ),
            (
                -1,
                q_goal,
                timing.u[..., -1, None] * q_vel_goal,
                timing.u[..., -1, None].square() * q_acc_goal
                + timing.u_s[..., -1, None] * q_vel_goal,
            ),
        )
        basis_rows = torch.stack(
            (
                trajectory.bspline.N.squeeze(0),
                trajectory.bspline.dN.squeeze(0),
                trajectory.bspline.ddN.squeeze(0),
            ),
            dim=0,
        )
        for endpoint, position, phase_velocity, phase_acceleration in targets:
            rows = basis_rows[:, endpoint]
            if endpoint == 0:
                indices = torch.arange(3, device=full.device)
            else:
                indices = torch.arange(full.shape[-2] - 3, full.shape[-2], device=full.device)
                if trajectory.keep_last_control_point:
                    position = torch.einsum("n,bnd->bd", rows[0], full)
            fixed = rows[:, indices]
            current = torch.einsum("kn,bnd->bkd", rows, full)
            current_fixed = torch.einsum("kj,bjd->bkd", fixed, full[:, indices])
            rhs = torch.stack(torch.broadcast_tensors(position, phase_velocity, phase_acceleration), dim=-2)
            solved = torch.linalg.solve(fixed, rhs - (current - current_fixed))
            full[:, indices] = solved

        q = torch.einsum("hk,bkd->bhd", trajectory.bspline.N.squeeze(0), full)
        q_s = torch.einsum("hk,bkd->bhd", trajectory.bspline.dN.squeeze(0), full)
        q_ss = torch.einsum("hk,bkd->bhd", trajectory.bspline.ddN.squeeze(0), full)
        timing = self.timing_spline.attach_spatial_derivatives(
            timing,
            q=q,
            q_s=q_s,
            q_ss=q_ss,
        )
        return SpaceTimeTrajectoryState(q=q, q_s=q_s, q_ss=q_ss, timing=timing)

    def _attach_collision_kinematics(
        self,
        state: SpaceTimeTrajectoryState,
        *,
        include_spatial_jacobians: bool,
    ) -> SpaceTimeTrajectoryState:
        dense_batch = None
        if (
            include_spatial_jacobians
            and getattr(self.spatial_guide, "gradient_pruning_enabled", False)
            and getattr(self.spatial_guide, "use_dense_parent_fast_path", False)
        ):
            dense_batch = self.spatial_guide.active_jacobian_computer.compute_dense(
                state.q
            )
            poses = dense_batch.poses
        else:
            batch, horizon, _ = state.q.shape
            poses = self.planning_task.robot.fk_collision_spheres(
                state.q.reshape(batch * horizon, -1)
            )
            poses = torch.stack(poses).transpose(0, 1).reshape(
                batch, horizon, -1, 3, 4
            )
        sphere_positions = link_pos_from_link_tensor(poses)[..., :3]
        return replace(
            state,
            collision_sphere_positions=sphere_positions,
            collision_sphere_poses=poses if dense_batch is not None else None,
            collision_sphere_jacobians=(
                dense_batch.jacobians.detach() if dense_batch is not None else None
            ),
        )

    def evaluate_control_points(
        self,
        control_points_normalized,
        timing_control_points=None,
        *,
        include_spatial_jacobians=False,
        return_state=False,
    ):
        timing_control_points = (
            self.timing_control_points if timing_control_points is None else timing_control_points
        )
        state = self._phase_trajectory(
            control_points_normalized, timing_control_points
        )
        state = self._attach_collision_kinematics(
            state,
            include_spatial_jacobians=include_spatial_jacobians,
        )
        result = self.cost_evaluator(trajectory_state=state)
        return (*result, state) if return_state else result

    def _spatial_descent(self, control_points_normalized, **kwargs):
        if self.settings.mode == "phase5_timing_only":
            return torch.zeros_like(control_points_normalized)
        cost_weight_overrides = dict(kwargs.pop("cost_weight_overrides", None) or {})
        # Phase 5 evaluates velocity and acceleration with the candidate's
        # current timing spline.  The legacy CostGuide variants assume the
        # fixed trajectory duration and would count the same constraints twice.
        cost_weight_overrides.update(
            {cost_name: 0.0 for cost_name in FIXED_TIME_KINEMATIC_COSTS}
        )
        phase5_trajectory_state = kwargs.pop("_phase5_trajectory_state", None)
        if not getattr(self.spatial_guide, "gradient_pruning_enabled", False):
            phase5_trajectory_state = None
        collision_entry = self.spatial_guide.costs.get("CostTaskSpaceCollisionObjects")
        original_field = None
        if collision_entry is not None:
            original_field = collision_entry.cost.collision_objects_field
            collision_entry.cost.collision_objects_field = self.dynamic_field.static_field
        try:
            if phase5_trajectory_state is not None:
                kwargs["_phase5_trajectory_state"] = phase5_trajectory_state
            return self.spatial_guide(
                control_points_normalized,
                cost_weight_overrides=cost_weight_overrides,
                **kwargs,
            )
        finally:
            if collision_entry is not None:
                collision_entry.cost.collision_objects_field = original_field

    def _update_timing(self, control_points, gradient):
        gradient, gradient_norm, clipped = _clip_per_candidate(
            gradient, self.settings.timing_max_grad_norm
        )
        if self.settings.mode == "phase5_scalar_duration":
            scalar_gradient = gradient.sum(dim=-1, keepdim=True) / gradient.shape[-1]
            gradient = scalar_gradient.expand_as(gradient)
        else:
            gradient = gradient * self.timing_spline.optimizable_mask

        self._optimizer_step += 1
        beta1 = self.settings.timing_beta1
        beta2 = self.settings.timing_beta2
        self._timing_momentum = beta1 * self._timing_momentum + (1.0 - beta1) * gradient
        self._timing_variance = beta2 * self._timing_variance + (1.0 - beta2) * gradient.square()
        momentum = self._timing_momentum / (1.0 - beta1**self._optimizer_step)
        variance = self._timing_variance / (1.0 - beta2**self._optimizer_step)
        update = self.settings.timing_learning_rate * momentum / (
            variance.sqrt() + self.settings.timing_epsilon
        )
        proposed = control_points - update
        if self.settings.mode == "phase5_scalar_duration":
            proposed = proposed.mean(dim=-1, keepdim=True).expand_as(proposed).clone()
        else:
            proposed = torch.where(
                self.timing_spline.optimizable_mask,
                proposed,
                control_points,
            )
            proposed = self.timing_spline.enforce_endpoint_derivatives(proposed)

        # Project by backtracking each invalid candidate toward its previous
        # feasible state; no clipping through a detached duration surrogate.
        for _ in range(8):
            duration = self.timing_spline.evaluate(proposed).duration
            invalid = (duration < self.settings.duration_min) | (
                duration > self.settings.duration_max
            )
            if not invalid.any().item():
                break
            proposed = torch.where(
                invalid[:, None], 0.5 * (proposed + control_points), proposed
            )
        duration = self.timing_spline.evaluate(proposed).duration
        invalid = (duration < self.settings.duration_min) | (
            duration > self.settings.duration_max
        )
        proposed = torch.where(invalid[:, None], control_points, proposed)
        return proposed.detach(), gradient_norm, clipped

    def __call__(
        self,
        control_points_normalized,
        return_cost=False,
        warmup=False,
        **kwargs,
    ):
        batch = int(control_points_normalized.shape[0])
        if self.timing_control_points is None or self.timing_control_points.shape[0] != batch:
            self.reset(batch)

        with torch.enable_grad():
            spatial = control_points_normalized.detach().requires_grad_(
                self.settings.mode == "phase5_joint"
            )
            timing = self.timing_control_points.detach().requires_grad_(True)
            reuse_kinematics = (
                self.reuse_spatial_kinematics_enabled
                and self.settings.mode != "phase5_timing_only"
                and getattr(self.spatial_guide, "gradient_pruning_enabled", False)
                and getattr(self.spatial_guide, "use_dense_parent_fast_path", False)
            )
            if reuse_kinematics:
                total, breakdown, evaluation, trajectory_state = (
                    self.evaluate_control_points(
                        spatial,
                        timing,
                        include_spatial_jacobians=True,
                        return_state=True,
                    )
                )
                spatial_descent = self._spatial_descent(
                    control_points_normalized,
                    return_cost=False,
                    warmup=warmup,
                    _phase5_trajectory_state=trajectory_state,
                    **kwargs,
                )
            else:
                spatial_descent = self._spatial_descent(
                    control_points_normalized,
                    return_cost=False,
                    warmup=warmup,
                    **kwargs,
                )
                total, breakdown, evaluation = self.evaluate_control_points(
                    spatial, timing
                )
            variables = [timing]
            if self.settings.mode == "phase5_joint":
                variables.insert(0, spatial)
            gradients = torch.autograd.grad(total.sum(), variables)
            if self.settings.mode == "phase5_joint":
                spatial_gradient, timing_gradient = gradients
                static_dynamic_cosine, cosine_valid = _gradient_cosine_per_candidate(
                    -spatial_descent.detach(), spatial_gradient.detach()
                )
                spatial_dynamic, spatial_norm, spatial_clipped = _clip_per_candidate(
                    spatial_gradient, self.settings.spatial_dynamic_max_grad_norm
                )
                spatial_descent = spatial_descent - (
                    self.settings.spatial_dynamic_scale * spatial_dynamic
                )
            else:
                timing_gradient = gradients[0]
                spatial_norm = torch.zeros(batch, dtype=timing.dtype, device=timing.device)
                spatial_clipped = torch.zeros(batch, dtype=torch.bool, device=timing.device)
                static_dynamic_cosine = torch.zeros_like(spatial_norm)
                cosine_valid = torch.zeros_like(spatial_clipped)

        if not return_cost:
            updated, timing_norm, timing_clipped = self._update_timing(
                self.timing_control_points, timing_gradient.detach()
            )
            self.timing_control_points = updated
        else:
            timing_norm = torch.linalg.norm(timing_gradient.flatten(start_dim=1), dim=-1)
            timing_clipped = torch.zeros(batch, dtype=torch.bool, device=timing.device)

        if not warmup:
            valid_cosine = static_dynamic_cosine[cosine_valid]
            self.statistics.append(
                {
                    "mode": self.settings.mode,
                    "spatial_gradient_norm_mean": float(spatial_norm.mean().detach().cpu()),
                    "timing_gradient_norm_mean": float(timing_norm.mean().detach().cpu()),
                    "spatial_clip_ratio": float(spatial_clipped.float().mean().detach().cpu()),
                    "timing_clip_ratio": float(timing_clipped.float().mean().detach().cpu()),
                    "static_dynamic_gradient_cosine_mean": (
                        float(valid_cosine.mean().detach().cpu())
                        if valid_cosine.numel()
                        else None
                    ),
                    "static_dynamic_gradient_cosine_min": (
                        float(valid_cosine.min().detach().cpu())
                        if valid_cosine.numel()
                        else None
                    ),
                    "static_dynamic_gradient_cosine_max": (
                        float(valid_cosine.max().detach().cpu())
                        if valid_cosine.numel()
                        else None
                    ),
                    "static_dynamic_gradient_conflict_ratio": (
                        float((valid_cosine < 0.0).float().mean().detach().cpu())
                        if valid_cosine.numel()
                        else None
                    ),
                    "static_dynamic_gradient_cosine_valid_ratio": float(
                        cosine_valid.float().mean().detach().cpu()
                    ),
                    "duration_min_s": float(evaluation.duration.min().detach().cpu()),
                    "duration_max_s": float(evaluation.duration.max().detach().cpu()),
                    "cost": {
                        name: float(value.mean().detach().cpu())
                        for name, value in breakdown.items()
                    },
                }
            )
        if return_cost:
            return total.detach(), spatial_descent
        return spatial_descent
