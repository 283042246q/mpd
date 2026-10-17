"""Differentiable canonical-grid codecs; query resolution never changes T."""
import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
import torch.nn.functional as F

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.parametric_trajectory.timing_spline import TimingSplineEvaluation

CODEC_CONTRACT = dict(version="joint_timing_v1", codec_grid_size=2049, spatial_points=21,
    spatial_degree=5, timing_points=8, timing_degree=3, duration_min=2., duration_max=14.,
    u_min=.05, density_floor=.001, integration="trapezoid_linear_cumulative")


class JointTimingCodec(nn.Module):
    def __init__(self, normalization, representation, velocity_limits, acceleration_limits):
        super().__init__()
        if representation not in ("c", "tau_r"):
            raise ValueError("representation must be c or tau_r")
        self.representation = representation
        for name, value in dict(q_mean=normalization["q_mean"], q_std=normalization["q_std"],
            target_mean=normalization[representation]["mean"], target_std=normalization[representation]["std"],
            velocity_limits=velocity_limits, acceleration_limits=acceleration_limits).items():
            value = torch.as_tensor(value, dtype=torch.float64)
            if not torch.isfinite(value).all() or (name.endswith(("std", "limits")) and (value <= 0).any()):
                raise ValueError("invalid codec normalization or limits")
            self.register_buffer(name, value)
        phase = np.linspace(0, 1, 2049)
        self.register_buffer("phase", torch.tensor(phase))
        self.register_buffer("timing_knots", torch.tensor(open_uniform_knots(8, 3)))
        for name, count, degree in (("p", 21, 5), ("z", 8, 3)):
            basis = BSpline(open_uniform_knots(count, degree), np.eye(count), degree)
            for order in range(3 if name == "p" else 2):
                self.register_buffer(f"{name}_basis_{order}", torch.tensor(basis.derivative(order)(phase)))

    def raw(self, z):
        return z * self.target_std.to(z) + self.target_mean.to(z)

    def full_path(self, p, q_start):
        p = p * self.q_std.to(p) + self.q_mean.to(p)
        qs = q_start.to(p).reshape(-1, 1, 7).expand(len(p), 3, 7)
        return torch.cat((qs, p[:, :-1], p[:, -1:].expand(-1, 3, -1)), dim=1)

    def prepare(self, full_path, z):
        # Double precision canonical integration matches the offline fitted labels.
        p, raw = full_path.double(), self.raw(z.double())
        phase = self.phase.to(p)
        q, qs, qss = [torch.einsum("sh,bhd->bsd", getattr(self, f"p_basis_{i}").to(p), p) for i in range(3)]
        if self.representation == "c":
            coeff = raw[:, [0, 0, 1, 2, 3, 4, 5, 5]]
        else:
            zero = raw[:, :1] * 0
            coeff = torch.cat((zero, zero, raw[:, 1:], raw[:, -1:]), -1)
        g, gs = [coeff @ getattr(self, f"z_basis_{i}").to(p).T for i in range(2)]
        tmin = None
        if self.representation == "c":
            u, us = .05 + F.softplus(g), g.sigmoid() * gs
        else:
            expg = torch.exp(g - g.amax(-1, keepdim=True))
            unit = expg / torch.trapezoid(expg, phase, dim=-1)[:, None]
            rho, rhos = .001 + .999 * unit, .999 * unit * gs
            vel = qs / rho[..., None]
            acc = qss / rho[..., None] ** 2 - qs * rhos[..., None] / rho[..., None] ** 3
            tv = (vel.abs() / self.velocity_limits.to(p)).flatten(1).amax(-1)
            ta = (acc.abs() / self.acceleration_limits.to(p)).flatten(1).amax(-1).clamp_min(1e-12).sqrt()
            tmin = torch.maximum(tv, ta).clamp_min(2.)
            duration = tmin + (14. - tmin).clamp_min(0) * raw[:, 0].sigmoid()
            u, us = duration[:, None] * rho, duration[:, None] * rhos
        increments = .5 * (u[:, :-1] + u[:, 1:]) * torch.diff(phase)
        times = torch.cat((u[:, :1] * 0, increments.cumsum(-1)), -1)
        return dict(p=p, coeff=coeff, u=u, us=us, times=times, tmin=tmin,
                    q=q, qs=qs, qss=qss, duration=times[:, -1])

    @staticmethod
    def interpolate(x, grid, values):
        upper = torch.searchsorted(grid.contiguous(), x.contiguous(), right=True).clamp(1, grid.shape[-1] - 1)
        lower = upper - 1
        lo, hi = torch.gather(grid, -1, lower), torch.gather(grid, -1, upper)
        weight = (x - lo) / (hi - lo).clamp_min(torch.finfo(x.dtype).eps)
        return torch.gather(values, -1, lower) * (1 - weight) + torch.gather(values, -1, upper) * weight

    @staticmethod
    def basis(s, count, degree):
        """Cox-de Boor and analytic derivatives on arbitrary batched phase grids."""
        knots = torch.as_tensor(open_uniform_knots(count, degree), dtype=s.dtype, device=s.device)
        # At s=1 use the left limit, then explicitly set the endpoint basis.
        x = s.clamp(0, 1 - torch.finfo(s.dtype).eps)[..., None]
        levels = [((x >= knots[:-1]) & (x < knots[1:])).to(s.dtype)]
        for d in range(1, degree + 1):
            n = len(knots) - d - 1
            left = knots[d:d+n] - knots[:n]; right = knots[d+1:d+1+n] - knots[1:1+n]
            a = torch.where(left > 0, (x - knots[:n]) / left.clamp_min(1e-30), 0)
            b = torch.where(right > 0, (knots[d+1:d+1+n] - x) / right.clamp_min(1e-30), 0)
            levels.append(a * levels[-1][..., :n] + b * levels[-1][..., 1:n+1])
        def deriv(d, order):
            if order == 0:
                return levels[d]
            n = len(knots) - d - 1
            l = knots[d:d+n] - knots[:n]; r = knots[d+1:d+1+n] - knots[1:1+n]
            prev = deriv(d - 1, order - 1)
            return torch.where(l > 0, d / l.clamp_min(1e-30), 0) * prev[..., :n] - torch.where(
                r > 0, d / r.clamp_min(1e-30), 0) * prev[..., 1:n+1]
        result = [deriv(degree, i) for i in range(3 if degree == 5 else 2)]
        endpoint = torch.zeros_like(result[0]); endpoint[..., -1] = 1
        result[0] = torch.where((s == 1)[..., None], endpoint, result[0])
        return result

    def evaluate(self, context, phase):
        p = context["p"]; s = torch.as_tensor(phase, device=p.device, dtype=p.dtype)
        if s.ndim == 1:
            s = s[None].expand(len(p), -1)
        pb = self.basis(s, 21, 5); zb = self.basis(s, 8, 3)
        q, qs, qss = [torch.einsum("bsh,bhd->bsd", b, p) for b in pb]
        g, gs = [torch.einsum("bsh,bh->bs", b, context["coeff"]) for b in zb]
        if self.representation == "c":
            u, us = .05 + F.softplus(g), g.sigmoid() * gs
        else:
            # Same canonical density normalizer, including its exponential shift.
            canonical_g = context["coeff"] @ self.z_basis_0.to(p).T
            shift = canonical_g.amax(-1, keepdim=True)
            normalizer = torch.trapezoid(torch.exp(canonical_g - shift), self.phase.to(p), dim=-1)[:, None]
            unit = torch.exp(g - shift) / normalizer
            u = context["duration"][:, None] * (.001 + .999 * unit)
            us = context["duration"][:, None] * .999 * unit * gs
        times = self.interpolate(s, self.phase.to(p).expand(len(p), -1), context["times"])
        dq = qs / u[..., None]; ddq = qss / u[..., None] ** 2 - qs * us[..., None] / u[..., None] ** 3
        return TimingSplineEvaluation(s, g, u, us, times, context["duration"], q, dq, ddq)

    def phase_at_times(self, context, times):
        t = torch.as_tensor(times, dtype=context["p"].dtype, device=context["p"].device)
        if t.ndim == 1:
            t = t[None].expand(len(context["p"]), -1)
        return self.interpolate(t, context["times"], self.phase.to(t).expand(len(t), -1))

    def export(self, z):
        raw = self.raw(z)
        result = dict(timing_latents=raw)
        if self.representation == "c":
            result["timing_control_points"] = raw[:, [0, 0, 1, 2, 3, 4, 5, 5]]
        else:
            result.update(timing_tau=raw[:, 0], timing_shape_control_points=raw[:, 1:])
        return result
