"""NumPy/SciPy fitting for the runtime ``TimingSpline`` representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import interpolate, optimize

from mpd.datasets.spacetime_legacy import open_uniform_knots


def _softplus(value: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, value)


def _inverse_softplus(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, np.finfo(np.float64).tiny)
    return value + np.log(-np.expm1(-value))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def decode_timing_latent(latent: np.ndarray) -> np.ndarray:
    """Map the non-redundant six-vector to the canonical eight-vector."""

    latent = np.asarray(latent, dtype=np.float64)
    if latent.shape[-1] != 6:
        raise ValueError(f"timing latent must end in 6 values, got {latent.shape}")
    return np.stack(
        (
            latent[..., 0],
            latent[..., 0],
            latent[..., 1],
            latent[..., 2],
            latent[..., 3],
            latent[..., 4],
            latent[..., 5],
            latent[..., 5],
        ),
        axis=-1,
    )


def encode_timing_control_points(control_points: np.ndarray) -> np.ndarray:
    control_points = np.asarray(control_points, dtype=np.float64)
    if control_points.shape[-1] != 8:
        raise ValueError(f"timing control points must end in 8 values, got {control_points.shape}")
    if not np.allclose(control_points[..., 0], control_points[..., 1], atol=1e-8, rtol=0.0):
        raise ValueError("first timing control-point pair is not equal")
    if not np.allclose(control_points[..., -2], control_points[..., -1], atol=1e-8, rtol=0.0):
        raise ValueError("last timing control-point pair is not equal")
    return control_points[..., [0, 2, 3, 4, 5, 7]]


@dataclass(frozen=True)
class TimingSplineNumpyEvaluation:
    phase: np.ndarray
    log_density: np.ndarray
    u: np.ndarray
    u_s: np.ndarray
    time_from_start: np.ndarray
    duration: float


class TimingSplineNumpy:
    """Numerically identical CPU representation of ``TimingSpline``."""

    def __init__(
        self,
        *,
        num_control_points: int = 8,
        degree: int = 3,
        num_phase_points: int = 128,
        u_min: float = 0.05,
    ) -> None:
        if num_control_points != 8:
            raise ValueError("v1 timing latent codec currently requires exactly 8 control points")
        self.num_control_points = int(num_control_points)
        self.degree = int(degree)
        self.num_phase_points = int(num_phase_points)
        self.u_min = float(u_min)
        self.phase = np.linspace(0.0, 1.0, self.num_phase_points)
        knots = open_uniform_knots(self.num_control_points, self.degree)
        basis_spline = interpolate.BSpline(
            knots, np.eye(self.num_control_points), self.degree, axis=0
        )
        self.basis = basis_spline(self.phase)
        self.basis_derivative = basis_spline.derivative(1)(self.phase)

    def evaluate(self, control_points: np.ndarray) -> TimingSplineNumpyEvaluation:
        control_points = np.asarray(control_points, dtype=np.float64)
        if control_points.shape != (self.num_control_points,):
            raise ValueError(
                f"timing control points must have shape {(self.num_control_points,)}, "
                f"got {control_points.shape}"
            )
        if not np.all(np.isfinite(control_points)):
            raise ValueError("timing control points contain NaN or Inf")
        encode_timing_control_points(control_points)
        log_density = self.basis @ control_points
        log_density_s = self.basis_derivative @ control_points
        u = self.u_min + _softplus(log_density)
        u_s = _sigmoid(log_density) * log_density_s
        increments = 0.5 * (u[:-1] + u[1:]) * np.diff(self.phase)
        time_from_start = np.concatenate(([0.0], np.cumsum(increments)))
        return TimingSplineNumpyEvaluation(
            phase=self.phase,
            log_density=log_density,
            u=u,
            u_s=u_s,
            time_from_start=time_from_start,
            duration=float(time_from_start[-1]),
        )


@dataclass(frozen=True)
class TimingFitResult:
    control_points: np.ndarray
    time_from_start: np.ndarray
    duration: float
    rmse: float
    relative_rmse: float
    success: bool
    message: str
    nfev: int


def _reference_density(phase: np.ndarray, reference_time: np.ndarray, u_min: float) -> np.ndarray:
    edge_order = 2 if phase.size >= 3 else 1
    density = np.gradient(reference_time, phase, edge_order=edge_order)
    density = np.maximum(density, u_min + 1e-6)
    return density


def fit_timing_reference(
    reference_time: np.ndarray,
    *,
    timing_spline: Optional[TimingSplineNumpy] = None,
    smoothness_weight: float = 0.0,
    max_nfev: int = 200,
) -> TimingFitResult:
    """Fit monotone ``t(s)`` samples to the canonical six-DoF timing latent."""

    timing_spline = timing_spline or TimingSplineNumpy()
    reference_time = np.asarray(reference_time, dtype=np.float64)
    if reference_time.shape != (timing_spline.num_phase_points,):
        raise ValueError(
            f"reference_time must have shape {(timing_spline.num_phase_points,)}, "
            f"got {reference_time.shape}"
        )
    if not np.all(np.isfinite(reference_time)):
        raise ValueError("reference_time contains NaN or Inf")
    reference_time = reference_time - reference_time[0]
    if reference_time[-1] <= 0.0 or not np.all(np.diff(reference_time) > 0.0):
        raise ValueError("reference_time must be strictly increasing")

    reference_u = _reference_density(
        timing_spline.phase, reference_time, timing_spline.u_min
    )
    target_log_density = _inverse_softplus(reference_u - timing_spline.u_min)
    full_initial, *_ = np.linalg.lstsq(
        timing_spline.basis, target_log_density, rcond=None
    )
    full_initial[1] = full_initial[0]
    full_initial[-2] = full_initial[-1]
    initial = np.clip(encode_timing_control_points(full_initial), -19.9, 39.9)
    duration = reference_time[-1]

    def residual(latent: np.ndarray) -> np.ndarray:
        evaluation = timing_spline.evaluate(decode_timing_latent(latent))
        time_residual = (evaluation.time_from_start - reference_time) / duration
        if smoothness_weight <= 0.0:
            return time_residual
        relative_density_slope = evaluation.u_s / evaluation.u
        smooth_residual = np.sqrt(smoothness_weight) * relative_density_slope
        return np.concatenate((time_residual, smooth_residual))

    solution = optimize.least_squares(
        residual,
        initial,
        bounds=(-20.0, 40.0),
        max_nfev=max_nfev,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    control_points = decode_timing_latent(solution.x)
    evaluation = timing_spline.evaluate(control_points)
    rmse = float(
        np.sqrt(np.mean(np.square(evaluation.time_from_start - reference_time)))
    )
    return TimingFitResult(
        control_points=control_points,
        time_from_start=evaluation.time_from_start,
        duration=evaluation.duration,
        rmse=rmse,
        relative_rmse=rmse / float(duration),
        success=bool(solution.success),
        message=str(solution.message),
        nfev=int(solution.nfev),
    )


def attach_spatial_derivatives(
    spatial_spline: interpolate.BSpline,
    timing: TimingSplineNumpyEvaluation,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = spatial_spline(timing.phase)
    q_s = spatial_spline.derivative(1)(timing.phase)
    q_ss = spatial_spline.derivative(2)(timing.phase)
    dq = q_s / timing.u[:, None]
    ddq = q_ss / np.square(timing.u[:, None]) - (
        q_s * timing.u_s[:, None] / np.power(timing.u[:, None], 3)
    )
    return q, dq, ddq


@dataclass(frozen=True)
class ValidatedTimingFit:
    fit: TimingFitResult
    reference_time: np.ndarray
    reference_scale: float
    v_ratio_max: float
    a_ratio_max: float
    q_within_limits: bool
    accepted: bool
    reject_reason: Optional[str]


def fit_and_validate_timing_reference(
    reference_time: np.ndarray,
    *,
    spatial_spline: interpolate.BSpline,
    q_min: np.ndarray,
    q_max: np.ndarray,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    timing_spline: Optional[TimingSplineNumpy] = None,
    duration_min: float = 2.0,
    duration_max: float = 15.0,
    ratio_tolerance: float = 1e-3,
    safety_margin: float = 1.02,
    max_safety_iterations: int = 5,
    max_relative_rmse: float = 0.05,
    smoothness_weight: float = 0.0,
) -> ValidatedTimingFit:
    """Fit a reference and globally slow it until the fitted spline is feasible."""

    timing_spline = timing_spline or TimingSplineNumpy()
    reference_time = np.asarray(reference_time, dtype=np.float64)
    reference_time = reference_time - reference_time[0]
    reference_scale = max(1.0, duration_min / float(reference_time[-1]))
    scaled_reference = reference_time * reference_scale
    fit: Optional[TimingFitResult] = None
    v_ratio_max = np.inf
    a_ratio_max = np.inf
    q_within_limits = False
    reject_reason: Optional[str] = None

    for _ in range(max_safety_iterations + 1):
        fit = fit_timing_reference(
            scaled_reference,
            timing_spline=timing_spline,
            smoothness_weight=smoothness_weight,
        )
        evaluation = timing_spline.evaluate(fit.control_points)
        q, dq, ddq = attach_spatial_derivatives(spatial_spline, evaluation)
        q_within_limits = bool(
            np.all(q >= np.asarray(q_min)[None, :] - 1e-6)
            and np.all(q <= np.asarray(q_max)[None, :] + 1e-6)
        )
        v_ratio_max = float(np.max(np.abs(dq) / np.asarray(dq_max)[None, :]))
        a_ratio_max = float(np.max(np.abs(ddq) / np.asarray(ddq_max)[None, :]))
        if not np.isfinite(v_ratio_max) or not np.isfinite(a_ratio_max):
            reject_reason = "non_finite_derivatives"
            break
        duration_factor = max(1.0, duration_min / fit.duration)
        feasibility_factor = max(1.0, v_ratio_max, np.sqrt(a_ratio_max))
        if duration_factor <= 1.0 and feasibility_factor <= 1.0 + ratio_tolerance:
            break
        step_scale = max(duration_factor, feasibility_factor * safety_margin)
        reference_scale *= step_scale
        scaled_reference = reference_time * reference_scale
        if scaled_reference[-1] > duration_max * 1.1:
            reject_reason = "duration_exceeds_max_during_safety_scaling"
            break

    assert fit is not None
    if reject_reason is None and not fit.success:
        reject_reason = "timing_fit_failed"
    if reject_reason is None and not q_within_limits:
        reject_reason = "spatial_joint_position_limit"
    if reject_reason is None and fit.duration < duration_min:
        reject_reason = "duration_below_min"
    if reject_reason is None and fit.duration > duration_max:
        reject_reason = "duration_above_max"
    if reject_reason is None and v_ratio_max > 1.0 + ratio_tolerance:
        reject_reason = "velocity_limit"
    if reject_reason is None and a_ratio_max > 1.0 + ratio_tolerance:
        reject_reason = "acceleration_limit"
    if reject_reason is None and fit.relative_rmse > max_relative_rmse:
        reject_reason = "timing_fit_relative_rmse"
    return ValidatedTimingFit(
        fit=fit,
        reference_time=scaled_reference,
        reference_scale=float(reference_scale),
        v_ratio_max=v_ratio_max,
        a_ratio_max=a_ratio_max,
        q_within_limits=q_within_limits,
        accepted=reject_reason is None,
        reject_reason=reject_reason,
    )
