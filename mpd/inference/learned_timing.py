"""Checkpoint loading and differentiable timing codecs for factorized inference.

Latents exposed to the sampler are standardized, exactly as during training.
No fitting or legacy training-loader calls occur during decoding.
"""
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
import torch.nn.functional as F

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.timing_training.model import TimingDenoiser, TimingDiffusion
from mpd.parametric_trajectory.timing_spline import TimingSplineEvaluation


def load_timing_checkpoint(path, device="cpu", *, use_ema=True):
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("checkpoint_version") != "timing_diffusion_checkpoint_v1":
        raise ValueError("unsupported timing checkpoint version")
    representation = payload.get("representation")
    if representation not in ("c", "tau_r"):
        raise ValueError("timing checkpoint must use c or tau_r")
    state = payload["ema_model_state_dict" if use_ema else "model_state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("timing checkpoint contains NaN/Inf; select a healthy earlier checkpoint")
    model = TimingDiffusion(TimingDenoiser(**payload["model_config"]), **payload["diffusion_config"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, payload


class LearnedTimingCodec(nn.Module):
    def __init__(self, normalization, *, num_phase_points=128, duration_min=2.,
                 duration_max=14., u_min=.05, density_floor=.001,
                 velocity_limits, acceleration_limits):
        super().__init__()
        if not 0 < u_min < duration_min < duration_max:
            raise ValueError("expected 0 < u_min < duration_min < duration_max")
        if not 0 < density_floor < 1:
            raise ValueError("density_floor must lie in (0,1)")
        if num_phase_points < 3:
            raise ValueError("timing grid needs at least three points")
        self.representation = normalization["representation"]
        if self.representation not in ("c", "tau_r"):
            raise ValueError("unknown timing representation")
        self.duration_min, self.duration_max = float(duration_min), float(duration_max)
        self.u_min, self.density_floor = float(u_min), float(density_floor)
        for key in ("path_mean", "path_std", "target_mean", "target_std"):
            value = torch.as_tensor(normalization[key], dtype=torch.float32)
            if value.ndim != 1 or not torch.isfinite(value).all():
                raise ValueError(f"invalid normalization: {key}")
            if key.endswith("std") and not (value > 0).all():
                raise ValueError(f"nonpositive normalization: {key}")
            self.register_buffer(key, value)
        if self.target_mean.shape != (6,) or self.target_std.shape != (6,):
            raise ValueError("timing normalization must have six dimensions")
        for key, value in (("velocity_limits", velocity_limits), ("acceleration_limits", acceleration_limits)):
            value = torch.as_tensor(value, dtype=torch.float32).flatten()
            if value.shape != self.path_mean.shape or not torch.isfinite(value).all() or not (value > 0).all():
                raise ValueError(f"invalid {key}")
            self.register_buffer(key, value)
        phase = np.linspace(0., 1., num_phase_points)
        spline = BSpline(open_uniform_knots(8, 3), np.eye(8), 3)
        self.register_buffer("phase", torch.tensor(phase, dtype=torch.float32))
        self.register_buffer("basis", torch.tensor(spline(phase), dtype=torch.float32))
        self.register_buffer("basis_d1", torch.tensor(spline.derivative()(phase), dtype=torch.float32))

    def normalize_path(self, full_path):
        return (full_path - self.path_mean) / self.path_std

    def raw(self, latent):
        return latent * self.target_std + self.target_mean

    def normalize_target(self, raw):
        return (raw - self.target_mean) / self.target_std

    def evaluate(self, latent, q, q_s, q_ss):
        """Return evaluation and Tmin; infeasible Tmin>Tmax is never clamped away.

        Such candidates decode to T>=Tmin (and hence above the task deadline),
        receive a deadline penalty and are rejected by final dense validation.
        """
        raw = self.raw(latent)
        if self.representation == "c":
            full = raw[:, [0, 0, 1, 2, 3, 4, 5, 5]]
        else:
            zero = torch.zeros_like(raw[:, :1])
            full = torch.cat((zero, zero, raw[:, 1:], raw[:, -1:]), dim=-1)
        g, gs = full @ self.basis.T, full @ self.basis_d1.T
        if self.representation == "c":
            u, us = self.u_min + F.softplus(g), torch.sigmoid(g) * gs
            tmin = torch.full_like(raw[:, 0], self.duration_min)
        else:
            exp_g = torch.exp(g - g.amax(dim=-1, keepdim=True))
            unit = exp_g / torch.trapezoid(exp_g, self.phase, dim=-1)[:, None]
            density = self.density_floor + (1. - self.density_floor) * unit
            density_s = (1. - self.density_floor) * unit * gs
            dq_unit = q_s / density[..., None]
            ddq_unit = q_ss / density[..., None].square() - q_s * density_s[..., None] / density[..., None].pow(3)
            tv = (dq_unit.abs() / self.velocity_limits).flatten(1).amax(-1)
            ta = (ddq_unit.abs() / self.acceleration_limits).flatten(1).amax(-1).clamp_min(1e-12).sqrt()
            tmin = torch.maximum(tv, ta).clamp_min(self.duration_min)
            duration = tmin + (self.duration_max - tmin).clamp_min(0.) * torch.sigmoid(raw[:, 0])
            u, us = duration[:, None] * density, duration[:, None] * density_s
        increments = .5 * (u[:, :-1] + u[:, 1:]) * torch.diff(self.phase)
        times = torch.cat((torch.zeros_like(u[:, :1]), increments.cumsum(-1)), dim=-1)
        dq = q_s / u[..., None]
        ddq = q_ss / u[..., None].square() - q_s * us[..., None] / u[..., None].pow(3)
        return TimingSplineEvaluation(self.phase, g, u, us, times, times[:, -1], q, dq, ddq), tmin
