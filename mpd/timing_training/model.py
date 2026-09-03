"""Small path-conditioned diffusion model for six-dimensional timing latents.

This module is deliberately independent from the spatial MPD TemporalUnet and
GaussianDiffusionModel.  Both ``c`` and ``[tau, r]`` training use the same
network; only the dataset codec and saved normalization differ.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy import interpolate
import torch
from torch import nn
import torch.nn.functional as F

from mpd.datasets.spacetime_legacy import open_uniform_knots


def cosine_beta_schedule(
    num_steps: int, *, schedule_offset: float = 0.008
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be at least two")
    steps = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    cumulative = torch.cos(
        ((steps / num_steps + schedule_offset) / (1.0 + schedule_offset))
        * math.pi
        * 0.5
    ).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999).to(dtype=torch.float32)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape((timesteps.shape[0],) + (1,) * (target.ndim - 1))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least four")
        self.dimension = int(dimension)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10000.0) / (half - 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.to(dtype=torch.float32)[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualConv1d(nn.Module):
    def __init__(self, channels: int, *, dilation: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class SpatialPathEncoder(nn.Module):
    """Encode ordered phase samples of ``q, q_s, q_ss`` without a robot runtime."""

    def __init__(
        self,
        *,
        dof: int,
        num_control_points: int,
        degree: int,
        num_phase_points: int = 64,
        width: int = 128,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if dof <= 0 or num_phase_points < 8 or width <= 0 or embedding_dim <= 0:
            raise ValueError("invalid path encoder dimensions")
        knots = open_uniform_knots(num_control_points, degree)
        basis_spline = interpolate.BSpline(
            knots, np.eye(num_control_points), degree, axis=0
        )
        phase = np.linspace(0.0, 1.0, num_phase_points)
        self.register_buffer(
            "basis", torch.as_tensor(basis_spline(phase), dtype=torch.float32)
        )
        self.register_buffer(
            "basis_d1",
            torch.as_tensor(basis_spline.derivative(1)(phase), dtype=torch.float32),
        )
        self.register_buffer(
            "basis_d2",
            torch.as_tensor(basis_spline.derivative(2)(phase), dtype=torch.float32),
        )
        self.register_buffer(
            "phase", torch.as_tensor(phase, dtype=torch.float32).reshape(1, 1, -1)
        )
        self.dof = int(dof)
        self.num_control_points = int(num_control_points)
        self.degree = int(degree)
        self.num_phase_points = int(num_phase_points)
        input_channels = 3 * dof + 1
        self.input_projection = nn.Conv1d(input_channels, width, kernel_size=1)
        self.residual = nn.Sequential(
            ResidualConv1d(width, dilation=1),
            ResidualConv1d(width, dilation=2),
            ResidualConv1d(width, dilation=4),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )
        reduced_points = num_phase_points // 8
        if reduced_points < 1:
            raise ValueError("num_phase_points is too small for path encoder")
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * reduced_points, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, control_points: torch.Tensor) -> torch.Tensor:
        expected = (self.num_control_points, self.dof)
        if control_points.ndim != 3 or tuple(control_points.shape[1:]) != expected:
            raise ValueError(
                f"path must have shape [batch,{expected[0]},{expected[1]}], "
                f"got {tuple(control_points.shape)}"
            )
        q = torch.einsum("sh,bhd->bsd", self.basis, control_points)
        q_s = torch.einsum("sh,bhd->bsd", self.basis_d1, control_points)
        q_ss = torch.einsum("sh,bhd->bsd", self.basis_d2, control_points)
        phase = self.phase.expand(control_points.shape[0], -1, -1)
        features = torch.cat(
            (q.transpose(1, 2), q_s.transpose(1, 2), q_ss.transpose(1, 2), phase),
            dim=1,
        )
        return self.output(self.downsample(self.residual(self.input_projection(features))))


class FiLMResidualMLP(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(hidden_dim)
        self.condition = nn.Linear(condition_dim, 2 * hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.condition(condition).chunk(2, dim=-1)
        hidden = self.normalization(value) * (1.0 + scale) + shift
        return value + self.layers(hidden)


class TimingDenoiser(nn.Module):
    """Predict noise in a standardized six-dimensional timing latent."""

    def __init__(
        self,
        *,
        dof: int,
        num_spatial_control_points: int,
        spatial_degree: int,
        latent_dim: int = 6,
        num_phase_points: int = 64,
        path_width: int = 128,
        path_embedding_dim: int = 256,
        hidden_dim: int = 256,
        time_embedding_dim: int = 128,
        num_residual_blocks: int = 6,
    ) -> None:
        super().__init__()
        if latent_dim != 6:
            raise ValueError("timing representations currently require latent_dim=6")
        if num_residual_blocks < 1:
            raise ValueError("num_residual_blocks must be positive")
        self.latent_dim = int(latent_dim)
        self.path_encoder = SpatialPathEncoder(
            dof=dof,
            num_control_points=num_spatial_control_points,
            degree=spatial_degree,
            num_phase_points=num_phase_points,
            width=path_width,
            embedding_dim=path_embedding_dim,
        )
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        condition_dim = path_embedding_dim + time_embedding_dim
        self.input = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMResidualMLP(hidden_dim, condition_dim) for _ in range(num_residual_blocks)]
        )
        self.output_normalization = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, noisy_timing: torch.Tensor, timesteps: torch.Tensor, path: torch.Tensor
    ) -> torch.Tensor:
        if noisy_timing.ndim != 2 or noisy_timing.shape[-1] != self.latent_dim:
            raise ValueError("noisy_timing must have shape [batch, 6]")
        if timesteps.shape != (noisy_timing.shape[0],):
            raise ValueError("timesteps must have shape [batch]")
        path_condition = self.path_encoder(path)
        time_condition = self.time_embedding(timesteps)
        condition = torch.cat((path_condition, time_condition), dim=-1)
        hidden = self.input(noisy_timing)
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(F.silu(self.output_normalization(hidden)))


class TimingDiffusion(nn.Module):
    """DDPM objective and ancestral sampler for conditional timing generation."""

    def __init__(
        self,
        denoiser: TimingDenoiser,
        *,
        num_diffusion_steps: int = 100,
        schedule_offset: float = 0.008,
        clip_clean: Optional[float] = 5.0,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.clip_clean = clip_clean
        betas = cosine_beta_schedule(
            self.num_diffusion_steps, schedule_offset=schedule_offset
        )
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        cumulative_previous = torch.cat((torch.ones(1), cumulative[:-1]))
        posterior_variance = betas * (1.0 - cumulative_previous) / (1.0 - cumulative)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", cumulative)
        self.register_buffer("sqrt_alphas_cumprod", cumulative.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - cumulative).sqrt())
        self.register_buffer("sqrt_recip_alphas_cumprod", (1.0 / cumulative).sqrt())
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", (1.0 / cumulative - 1.0).sqrt()
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance",
            posterior_variance.clamp(min=1e-20).log(),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * cumulative_previous.sqrt() / (1.0 - cumulative),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - cumulative_previous) * alphas.sqrt() / (1.0 - cumulative),
        )

    def q_sample(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        return _extract(self.sqrt_alphas_cumprod, timesteps, clean) * clean + _extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, clean
        ) * noise

    def predict_clean_from_noise(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        clean = _extract(self.sqrt_recip_alphas_cumprod, timesteps, noisy) * noisy - _extract(
            self.sqrt_recipm1_alphas_cumprod, timesteps, noisy
        ) * noise
        if self.clip_clean is not None:
            clean = clean.clamp(-float(self.clip_clean), float(self.clip_clean))
        return clean

    def training_loss(
        self,
        path: torch.Tensor,
        clean_timing: torch.Tensor,
        *,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = clean_timing.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                0, self.num_diffusion_steps, (batch,), device=clean_timing.device
            )
        noise = torch.randn_like(clean_timing) if noise is None else noise
        noisy = self.q_sample(clean_timing, timesteps, noise)
        predicted = self.denoiser(noisy, timesteps, path)
        per_dimension = torch.mean(torch.square(predicted - noise), dim=0)
        loss = torch.mean(per_dimension)
        return loss, {
            "noise_mse": loss.detach(),
            "noise_mse_per_dimension": per_dimension.detach(),
            "mean_timestep": timesteps.to(dtype=torch.float32).mean().detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        path: torch.Tensor,
        *,
        num_candidates: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        batch = path.shape[0]
        conditioned_path = path.repeat_interleave(num_candidates, dim=0)
        value = torch.randn(
            (batch * num_candidates, self.denoiser.latent_dim),
            device=path.device,
            dtype=path.dtype,
            generator=generator,
        )
        for step in reversed(range(self.num_diffusion_steps)):
            timesteps = torch.full(
                (value.shape[0],), step, device=value.device, dtype=torch.long
            )
            predicted_noise = self.denoiser(value, timesteps, conditioned_path)
            clean = self.predict_clean_from_noise(value, timesteps, predicted_noise)
            mean = _extract(self.posterior_mean_coef1, timesteps, value) * clean + _extract(
                self.posterior_mean_coef2, timesteps, value
            ) * value
            if step:
                noise = torch.randn(
                    value.shape,
                    device=value.device,
                    dtype=value.dtype,
                    generator=generator,
                )
                value = mean + torch.exp(
                    0.5 * _extract(self.posterior_log_variance, timesteps, value)
                ) * noise
            else:
                value = mean
        return value.reshape(batch, num_candidates, self.denoiser.latent_dim)
