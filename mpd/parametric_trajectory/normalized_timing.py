"""Duration/shape-decoupled timing representation for Space-Time MPD.

The normalized density ``pi(s)`` integrates to one.  A trajectory timing is
therefore reconstructed as ``dt/ds = T * pi(s)`` where ``T`` is total duration
and the five-vector ``r`` controls only the relative allocation of time along
the path.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import interpolate

from mpd.datasets.spacetime_legacy import open_uniform_knots


NORMALIZED_TIMING_REPRESENTATION = "duration_normalized_log_density_v1"


def decode_shape_latent(latent: np.ndarray) -> np.ndarray:
    """Expand five gauge-fixed values to eight clamped spline coefficients."""

    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape[-1] != 5:
        raise ValueError(f"shape latent must end in 5 values, got {latent.shape}")
    zeros = np.zeros_like(latent[..., 0])
    return np.stack(
        (
            zeros,
            zeros,
            latent[..., 0],
            latent[..., 1],
            latent[..., 2],
            latent[..., 3],
            latent[..., 4],
            latent[..., 4],
        ),
        axis=-1,
    )


@dataclass(frozen=True)
class NormalizedTimingEvaluation:
    phase: np.ndarray
    log_density: np.ndarray
    density: np.ndarray
    density_s: np.ndarray
    normalized_time: np.ndarray


@dataclass(frozen=True)
class NormalizedTimingFit:
    shape_control_points: np.ndarray
    normalized_time: np.ndarray
    rmse: float
    relative_density_clip_fraction: float


@dataclass(frozen=True)
class MinimumDuration:
    velocity: float
    acceleration: float
    floor: float
    value: float


class NormalizedTimingSplineNumpy:
    """CPU implementation of ``dt/ds = T * pi(s; r)``.

    The underlying cubic spline has the same eight-control-point endpoint
    contract as the v1 TimingSpline.  Removing its additive log-density gauge
    leaves five non-redundant shape values.
    """

    def __init__(
        self,
        *,
        num_control_points: int = 8,
        degree: int = 3,
        num_phase_points: int = 128,
        density_floor: float = 1e-3,
    ) -> None:
        if num_control_points != 8:
            raise ValueError("v1 normalized timing codec requires exactly 8 control points")
        if not 0.0 <= density_floor < 1.0:
            raise ValueError("density_floor must be in [0, 1)")
        self.num_control_points = int(num_control_points)
        self.num_shape_control_points = 5
        self.degree = int(degree)
        self.num_phase_points = int(num_phase_points)
        self.density_floor = float(density_floor)
        self.phase = np.linspace(0.0, 1.0, self.num_phase_points)
        knots = open_uniform_knots(self.num_control_points, self.degree)
        basis_spline = interpolate.BSpline(
            knots, np.eye(self.num_control_points), self.degree, axis=0
        )
        self.basis = basis_spline(self.phase)
        self.basis_derivative = basis_spline.derivative(1)(self.phase)
        self._latent_expansion = decode_shape_latent(
            np.eye(self.num_shape_control_points, dtype=np.float64)
        ).T
        self._latent_basis = self.basis @ self._latent_expansion

    def evaluate(self, shape_control_points: np.ndarray) -> NormalizedTimingEvaluation:
        shape_control_points = np.asarray(shape_control_points, dtype=np.float64)
        if shape_control_points.shape != (self.num_shape_control_points,):
            raise ValueError(
                "shape control points must have shape "
                f"{(self.num_shape_control_points,)}, got {shape_control_points.shape}"
            )
        if not np.all(np.isfinite(shape_control_points)):
            raise ValueError("shape control points contain NaN or Inf")
        full = decode_shape_latent(shape_control_points)
        log_density = self.basis @ full
        log_density_s = self.basis_derivative @ full
        # Subtracting a constant is the removed gauge and avoids exp overflow.
        exponential = np.exp(log_density - np.max(log_density))
        normalizer = float(np.trapz(exponential, self.phase))
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError("normalized timing density has invalid integral")
        unit_density = exponential / normalizer
        density = self.density_floor + (1.0 - self.density_floor) * unit_density
        density_s = (1.0 - self.density_floor) * unit_density * log_density_s
        increments = 0.5 * (density[:-1] + density[1:]) * np.diff(self.phase)
        normalized_time = np.concatenate(([0.0], np.cumsum(increments)))
        return NormalizedTimingEvaluation(
            phase=self.phase,
            log_density=log_density,
            density=density,
            density_s=density_s,
            normalized_time=normalized_time,
        )

    def fit_normalized_time(self, normalized_time: np.ndarray) -> NormalizedTimingFit:
        """Fit a monotone zero-to-one curve using one linear least-squares solve."""

        normalized_time = np.asarray(normalized_time, dtype=np.float64)
        expected = (self.num_phase_points,)
        if normalized_time.shape != expected:
            raise ValueError(f"normalized_time must have shape {expected}, got {normalized_time.shape}")
        if not np.all(np.isfinite(normalized_time)):
            raise ValueError("normalized_time contains NaN or Inf")
        normalized_time = normalized_time - normalized_time[0]
        if normalized_time[-1] <= 0.0 or not np.all(np.diff(normalized_time) > 0.0):
            raise ValueError("normalized_time must be strictly increasing")
        normalized_time = normalized_time / normalized_time[-1]
        edge_order = 2 if self.num_phase_points >= 3 else 1
        target_density = np.gradient(normalized_time, self.phase, edge_order=edge_order)
        threshold = self.density_floor + np.finfo(np.float64).eps
        clipped = target_density <= threshold
        unit_density = np.maximum(
            (target_density - self.density_floor) / (1.0 - self.density_floor),
            np.finfo(np.float64).tiny,
        )
        target_log_density = np.log(unit_density)
        # The extra constant absorbs the log-density gauge discarded by r.
        design = np.column_stack((self._latent_basis, np.ones(self.num_phase_points)))
        solution, *_ = np.linalg.lstsq(design, target_log_density, rcond=None)
        shape = solution[: self.num_shape_control_points]
        evaluation = self.evaluate(shape)
        rmse = float(
            np.sqrt(np.mean(np.square(evaluation.normalized_time - normalized_time)))
        )
        return NormalizedTimingFit(
            shape_control_points=shape,
            normalized_time=evaluation.normalized_time,
            rmse=rmse,
            relative_density_clip_fraction=float(np.mean(clipped)),
        )


def minimum_duration_for_path(
    spatial_spline: interpolate.BSpline,
    timing: NormalizedTimingEvaluation,
    *,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    duration_floor: float = 0.0,
    safety_factor: float = 1.0,
) -> MinimumDuration:
    """Compute the sampled velocity/acceleration lower bound for ``T``."""

    dq_max = np.asarray(dq_max, dtype=np.float64)
    ddq_max = np.asarray(ddq_max, dtype=np.float64)
    if dq_max.ndim != 1 or ddq_max.shape != dq_max.shape:
        raise ValueError("dq_max and ddq_max must be equal-length vectors")
    if np.any(dq_max <= 0.0) or np.any(ddq_max <= 0.0):
        raise ValueError("dynamic limits must be positive")
    if duration_floor < 0.0:
        raise ValueError("duration_floor must be non-negative")
    if safety_factor < 1.0:
        raise ValueError("safety_factor must be at least one")
    q_s = np.asarray(spatial_spline.derivative(1)(timing.phase), dtype=np.float64)
    q_ss = np.asarray(spatial_spline.derivative(2)(timing.phase), dtype=np.float64)
    if q_s.shape != (timing.phase.size, dq_max.size) or q_ss.shape != q_s.shape:
        raise ValueError("spatial spline output does not match dynamic limit dimensions")
    density = timing.density[:, None]
    unit_velocity = q_s / density
    unit_acceleration = q_ss / np.square(density) - (
        q_s * timing.density_s[:, None] / np.power(density, 3)
    )
    velocity = float(np.max(np.abs(unit_velocity) / dq_max[None, :]))
    acceleration = float(
        np.sqrt(np.max(np.abs(unit_acceleration) / ddq_max[None, :]))
    )
    velocity *= safety_factor
    acceleration *= safety_factor
    value = max(float(duration_floor), velocity, acceleration)
    return MinimumDuration(
        velocity=velocity,
        acceleration=acceleration,
        floor=float(duration_floor),
        value=value,
    )


def duration_to_logit(
    duration: float,
    duration_min: float,
    duration_max: float,
    *,
    clip: float = 1e-3,
) -> tuple[float, float, bool]:
    """Return ``(tau, raw_fraction, clipped)`` for bounded duration training."""

    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be in (0, 0.5)")
    if not duration_max > duration_min:
        return float("nan"), float("nan"), False
    fraction = (float(duration) - float(duration_min)) / (
        float(duration_max) - float(duration_min)
    )
    bounded = float(np.clip(fraction, clip, 1.0 - clip))
    tau = float(np.log(bounded) - np.log1p(-bounded))
    return tau, float(fraction), bool(bounded != fraction)
