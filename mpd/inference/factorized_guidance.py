"""Stateless physical cost evaluation; only the sampler applies updates."""
import torch

from mpd.inference.space_time_guidance import (
    InferenceOnlySpaceTimeGuide, SpaceTimeTrajectoryState, _clip_per_candidate,
)


class FactorizedCostGuide(InferenceOnlySpaceTimeGuide):
    """Reuse collision/kinematic costs without the Phase-5 timing optimizer.

    timing_control_points is solely a final-result slot required by the existing
    runtime validator. It contains standardized six-vectors until artifact export.
    """
    def __init__(self, *args, codec, factorized_settings, **kwargs):
        super().__init__(*args, **kwargs)
        self.codec = codec
        self.factorized_settings = factorized_settings

    def reset(self, candidate_count):
        self.timing_control_points = None
        self.statistics = []

    def full_path(self, p):
        trajectory = self.planning_task.parametric_trajectory
        # Timing training uses rest-to-rest paths. Reject unsupported boundaries
        # rather than silently violating the request or creating a circular Tmin.
        for name in ("q_vel_start", "q_vel_goal", "q_acc_start", "q_acc_goal"):
            value = getattr(trajectory, name, None)
            if value is not None and torch.any(value != 0):
                raise ValueError("learned timing inference currently requires zero endpoint velocity/acceleration")
        return trajectory.augment_control_points_fn(self.dataset.unnormalize_control_points(p), None, None)

    def condition(self, p):
        return self.codec.normalize_path(self.full_path(p))

    def nominal(self, p):
        return torch.zeros((len(p), 6), device=p.device, dtype=p.dtype)

    def state(self, p, z, *, weak=False, collision_kinematics=True):
        full = self.full_path(p)
        bspline = self.planning_task.parametric_trajectory.bspline
        q, qs, qss = [torch.einsum("hk,bkd->bhd", basis.squeeze(0), full)
                      for basis in (bspline.N, bspline.dN, bspline.ddN)]
        if weak:
            c = self.timing_spline.linear_control_points(self.settings.nominal_duration, batch_shape=(len(p),))
            timing = self.timing_spline.evaluate(c, q=q, q_s=qs, q_ss=qss)
        else:
            timing, _ = self.codec.evaluate(z, q, qs, qss)
        state = SpaceTimeTrajectoryState(q, qs, qss, timing)
        return (self._attach_collision_kinematics(state, include_spatial_jacobians=False)
                if collision_kinematics else state)

    def evaluate_control_points(self, p, timing_control_points=None, *, return_state=False, **kwargs):
        z = self.timing_control_points if timing_control_points is None else timing_control_points
        state = self.state(p, z)
        result = self.cost_evaluator(trajectory_state=state)
        return (*result, state) if return_state else result

    def evaluate(self, p, z, *, weak=False):
        state = self.state(p, z, weak=weak)
        total, breakdown, timing = self.cost_evaluator(trajectory_state=state)
        if weak:
            total = (self.factorized_settings.weak_dynamic_scale * self.settings.dynamic_collision_weight
                     * breakdown["dynamic_collision"])
            if not self.settings.dynamic_guidance_enabled:
                total = total * 0.
        else:
            # A smooth deadline penalty also repairs invalid unguided c or an
            # r shape whose dynamic Tmin exceeds the allowed task deadline.
            bounds = (torch.relu(timing.duration - self.settings.duration_max).square()
                      + torch.relu(self.settings.duration_min - timing.duration).square())
            total = total + 10. * bounds
        return total, breakdown, timing

    @torch.enable_grad()
    def gradients(self, p, z, *, active, weak=False):
        if active not in ("space", "timing", "joint"):
            raise ValueError("active must be space, timing or joint")
        p = p.detach().requires_grad_(True)
        z = z.detach().requires_grad_(True)
        total, _, _ = self.evaluate(p, z, weak=weak)
        gp, gz = (torch.autograd.grad(total.sum(), (p, z), allow_unused=True)
                  if total.requires_grad else (None, None))
        gp = torch.zeros_like(p) if gp is None else gp
        gz = torch.zeros_like(z) if gz is None else gz
        if active in ("space", "joint"):
            # Existing spatial manager returns a DESCENT direction.
            gp = gp - self._spatial_descent(p.detach()).detach()
        else:
            gp = torch.zeros_like(p)
        if active == "space":
            gz = torch.zeros_like(z)
        if not torch.isfinite(gp).all() or not torch.isfinite(gz).all():
            raise ValueError("nonfinite factorized cost gradient")
        return gp.detach(), gz.detach()

    def refine(self, p, z, *, active, weak=False):
        gp, gz = self.gradients(p, z, active=active, weak=weak)
        gp, _, _ = _clip_per_candidate(gp, self.settings.spatial_dynamic_max_grad_norm)
        gz, _, _ = _clip_per_candidate(gz, self.settings.timing_max_grad_norm)
        proposed_p = (p - self.factorized_settings.space_lr * gp).clamp(-1., 1.)
        proposed_z = z - self.factorized_settings.timing_lr * gz
        with torch.no_grad():
            if not weak:
                # Preserve already feasible bounds. Invalid initial predictions
                # may improve their violation instead of being permanently frozen.
                def violation(pp, zz):
                    t = self.state(pp, zz, collision_kinematics=False).timing.duration
                    return torch.relu(t - self.settings.duration_max) + torch.relu(self.settings.duration_min - t)
                baseline = violation(p, z)
                for _ in range(8):
                    v = violation(proposed_p, proposed_z)
                    invalid = ~torch.isfinite(v) | (v > baseline + 1e-6)
                    if not invalid.any():
                        break
                    proposed_p = torch.where(invalid[:, None, None], .5 * (proposed_p + p), proposed_p)
                    proposed_z = torch.where(invalid[:, None], .5 * (proposed_z + z), proposed_z)
                v = violation(proposed_p, proposed_z)
                invalid = ~torch.isfinite(v) | (v > baseline + 1e-6)
                proposed_p = torch.where(invalid[:, None, None], p, proposed_p)
                proposed_z = torch.where(invalid[:, None], z, proposed_z)
        return proposed_p.detach(), proposed_z.detach()

    def __call__(self, *args, **kwargs):
        raise RuntimeError("factorized guidance must be scheduled by FactorizedSampler")
