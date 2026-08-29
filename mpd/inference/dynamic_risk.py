"""Shared differentiable aggregation for fixed- and variable-time dynamic risk."""

from __future__ import annotations

import torch


def time_weighted_cvar(
    values: torch.Tensor,
    times: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Return the worst-fraction mean using trapezoidal physical-time weights."""

    if values.shape != times.shape or values.ndim < 1:
        raise ValueError("values and times must have the same non-empty shape")
    if values.shape[-1] < 2:
        raise ValueError("CVaR requires at least two time samples")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("CVaR fraction must lie in (0, 1]")
    intervals = torch.diff(times, dim=-1)
    weights = torch.zeros_like(values)
    weights[..., 0] = 0.5 * intervals[..., 0]
    weights[..., -1] = 0.5 * intervals[..., -1]
    if values.shape[-1] > 2:
        weights[..., 1:-1] = 0.5 * (
            intervals[..., :-1] + intervals[..., 1:]
        )
    sorted_values, order = torch.sort(values, dim=-1, descending=True)
    sorted_weights = torch.gather(weights, -1, order)
    target_weight = float(fraction) * weights.sum(dim=-1, keepdim=True)
    cumulative_before = torch.cumsum(sorted_weights, dim=-1) - sorted_weights
    included = torch.minimum(
        sorted_weights,
        torch.clamp(target_weight - cumulative_before, min=0.0),
    )
    return (sorted_values * included).sum(dim=-1) / target_weight.squeeze(
        -1
    ).clamp_min(torch.finfo(values.dtype).eps)


def mean_cvar_dynamic_risk(
    penetration: torch.Tensor,
    times: torch.Tensor,
    *,
    alpha: float = 0.5,
    cvar_fraction: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate squared sphere penetration over physical time."""

    if penetration.shape[:-1] != times.shape:
        raise ValueError("penetration must have shape [batch,time,spheres]")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("dynamic risk alpha must lie in [0, 1]")
    density = penetration.square().sum(dim=-1)
    duration = (times[..., -1] - times[..., 0]).clamp_min(
        torch.finfo(times.dtype).eps
    )
    mean = torch.trapezoid(density, times, dim=-1) / duration
    cvar = time_weighted_cvar(density, times, cvar_fraction)
    risk = float(alpha) * mean + (1.0 - float(alpha)) * cvar
    return risk, {"mean": mean, "cvar": cvar}
