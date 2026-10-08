"""Fit raw q-t teachers, preserving time, and revalidate the decoded trajectory."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.parametric_trajectory.timing_fitting import TimingSplineNumpy, fit_and_validate_timing_reference
from mpd.parametric_trajectory.normalized_timing import (
    NormalizedTimingSplineNumpy, minimum_duration_for_path, duration_to_logit,
)
from scripts.spacetime_data.warehouse_world import interaction_audit


def fit_spatial_time_path(path, *, control_points=29, degree=5, eta=0.05):
    """Fixed phase gauge; positive time contribution preserves internal holds."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 8 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("raw path must be finite [N,7+time]")
    if abs(path[0, -1]) > 1e-6 or np.any(np.diff(path[:, -1]) <= 0):
        raise ValueError("raw time must start at zero and increase strictly")
    if not 0 < eta < 1 or degree < 3 or control_points < 2 * degree + 1:
        raise ValueError("invalid spatial codec")
    chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :7], axis=0), axis=1))]
    if chord[-1] < 1e-6:
        raise ValueError("stationary-only task has no spatial training trajectory")
    phase = (1 - eta) * chord / chord[-1] + eta * path[:, -1] / path[-1, -1]
    sample = np.linspace(0, 1, 512)
    reference = np.stack([np.interp(sample, phase, path[:, axis]) for axis in range(7)], axis=1)
    knots = open_uniform_knots(control_points, degree)
    basis = BSpline(knots, np.eye(control_points), degree)(sample)
    coefficients = np.zeros((control_points, 7))
    coefficients[:3], coefficients[-3:] = path[0, :7], path[-1, :7]
    free = slice(3, control_points - 3)
    fixed = basis @ coefficients
    curvature = np.diff(np.eye(control_points), n=2, axis=0)
    regularization = 0.01
    design = np.r_[basis[:, free], regularization * curvature[:, free]]
    target = np.r_[reference - fixed, -regularization * curvature @ coefficients]
    coefficients[free], *_ = np.linalg.lstsq(design, target, rcond=None)
    spline = BSpline(knots, coefficients, degree)
    error = float(np.sqrt(np.mean((spline(sample) - reference)**2)))
    return spline, phase, error


def physical_samples(spline, phase, density, density_s):
    q = spline(phase)
    dq = spline.derivative(1)(phase) / density[:, None]
    ddq = (spline.derivative(2)(phase) / density[:, None]**2
           - spline.derivative(1)(phase) * density_s[:, None] / density[:, None]**3)
    return q, dq, ddq


def audit_decoded(spline, phase, times, density, density_s, collision, query, forecast,
                  *, validation_dt=0.01):
    """Independent dense phase *and* time samples; never trust the raw solution."""
    robot = collision.robot
    q, dq, ddq = physical_samples(spline, phase, density, density_s)
    velocity = robot.dq_max.cpu().numpy()
    acceleration = robot.ddq_max.cpu().numpy()
    duration = float(times[-1])
    v_ratio = float(np.max(np.abs(dq) / velocity))
    a_ratio = float(np.max(np.abs(ddq) / acceleration))
    if not np.isfinite(q).all() or not np.isfinite(v_ratio + a_ratio):
        return None, "non_finite"
    if v_ratio > 1.001 or a_ratio > 1.001:
        return None, "kinematic_limits"
    if duration <= 0 or duration > forecast.horizon + 1e-6:
        return None, "horizon"
    interaction = interaction_audit(forecast, duration)
    if not interaction["all_objects_interact_during_execution"]:
        return None, "interaction_outside_execution"
    dt = min(validation_dt, 0.02 / max(0.1, float(np.abs(dq).max())))
    uniform_times = np.linspace(0, duration, int(np.ceil(duration / dt)) + 1)
    dense_phase = np.unique(np.r_[phase, np.interp(uniform_times, times, phase)])
    dense_times = np.interp(dense_phase, phase, times)
    dense_q = spline(dense_phase)
    clearances = collision.clearances(dense_q, dense_times, np.full(len(dense_q), query)).cpu().numpy()
    minima = clearances.min(axis=0)
    if not np.isfinite(minima).all() or np.any(minima <= collision.margin):
        reason = ["static_collision", "self_collision", "dynamic_collision"][int(np.argmin(minima))]
        return None, reason
    return {"duration": duration, "v_ratio_max": v_ratio, "a_ratio_max": a_ratio,
            "clearance_min": minima, "validation_dt_max": dt,
            "validation_samples": len(dense_phase), **interaction}, None


def fit_teacher(path, collision, query, forecast, *, spatial_rmse_max=0.08,
                timing_rmse_max=0.08, duration_floor=2., require_tau_r=False):
    """Return a safe canonical c trajectory plus independently checked tau-r.

    Any global slowdown changes the interaction with the world and is rechecked.
    Rejected raw teachers are kept by the caller as rejected, never as positives.
    """
    robot = collision.robot
    try:
        spline, raw_phase, spatial_error = fit_spatial_time_path(path)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, "spatial_fit:" + str(exc)
    # Clamp only free control points to the convex position-limit hull, then
    # recompute error and collision. This is an explicit fit modification.
    spline.c[3:-3] = np.clip(spline.c[3:-3], robot.q_pos_min.cpu().numpy(), robot.q_pos_max.cpu().numpy())
    sample = np.linspace(0, 1, 512)
    raw_reference = np.stack([np.interp(sample, raw_phase, path[:, axis]) for axis in range(7)], axis=1)
    spatial_error = float(np.sqrt(np.mean((spline(sample) - raw_reference)**2)))
    if spatial_error > spatial_rmse_max:
        return None, "spatial_fit_rmse"
    codec = TimingSplineNumpy(num_phase_points=257)
    dense_codec = TimingSplineNumpy(num_phase_points=2049)
    reference_time = np.interp(codec.phase, raw_phase, path[:, -1])
    kwargs = dict(q_min=robot.q_pos_min.cpu().numpy(), q_max=robot.q_pos_max.cpu().numpy(),
                  dq_max=robot.dq_max.cpu().numpy(), ddq_max=robot.ddq_max.cpu().numpy(),
                  duration_min=duration_floor, duration_max=forecast.horizon,
                  max_relative_rmse=timing_rmse_max, safety_margin=1.04)
    reason = "no_valid_fit"
    for extra_scale in (1., 1.15, 1.35):
        validated = fit_and_validate_timing_reference(reference_time * extra_scale, spatial_spline=spline,
                                                     timing_spline=codec, **kwargs)
        if not validated.accepted:
            reason = validated.reject_reason
            continue
        c = validated.fit.control_points
        evaluation = dense_codec.evaluate(c)
        quality, reason = audit_decoded(spline, evaluation.phase, evaluation.time_from_start,
                                         evaluation.u, evaluation.u_s, collision, query, forecast)
        if quality is None:
            continue
        normalized = NormalizedTimingSplineNumpy(num_phase_points=2049)
        shape_fit = normalized.fit_normalized_time(evaluation.time_from_start / evaluation.duration)
        normalized_eval = normalized.evaluate(shape_fit.shape_control_points)
        t_min = minimum_duration_for_path(spline, normalized_eval, dq_max=kwargs["dq_max"],
                                          ddq_max=kwargs["ddq_max"], duration_floor=duration_floor).value
        tau, fraction, clipped = duration_to_logit(evaluation.duration, t_min, forecast.horizon)
        tau_quality, tau_reason = None, "duration_bounds"
        if np.isfinite(tau) and not clipped and 0 < fraction < 1 and shape_fit.rmse <= timing_rmse_max:
            tau_quality, tau_reason = audit_decoded(
                spline, normalized.phase, normalized_eval.normalized_time * evaluation.duration,
                normalized_eval.density * evaluation.duration, normalized_eval.density_s * evaluation.duration,
                collision, query, forecast)
        if require_tau_r and tau_quality is None:
            reason = "tau_r:" + str(tau_reason)
            continue
        # Saved executable reference uses a time-uniform grid; raw waits remain
        # in the separate raw q-t dataset even if this codec approximates them.
        times = np.linspace(0, evaluation.duration, int(np.ceil(evaluation.duration / 0.01)) + 1)
        phase = np.interp(times, evaluation.time_from_start, evaluation.phase)
        q = spline(phase)
        basis = BSpline(open_uniform_knots(8, 3), np.eye(8), 3)
        log_u = basis(phase) @ c
        u = 0.05 + np.logaddexp(0, log_u)
        u_s = (1 / (1 + np.exp(-np.clip(log_u, -700, 700)))) * (basis.derivative(1)(phase) @ c)
        _, dq, ddq = physical_samples(spline, phase, u, u_s)
        return {"P": spline.c, "knots": spline.t, "degree": spline.k, "c": c,
                "reference_phase": evaluation.phase, "reference_time": evaluation.time_from_start,
                "execution_q": q, "execution_dq": dq, "execution_ddq": ddq, "execution_time": times,
                "quality": {**quality, "spatial_fit_rmse": spatial_error,
                            "timing_relative_rmse": validated.fit.relative_rmse},
                "tau_r": np.r_[tau, shape_fit.shape_control_points], "tau_r_valid": tau_quality is not None,
                "tau_r_quality": tau_quality, "tau_r_reject_reason": tau_reason,
                "T_min": t_min, "phase_eta": 0.05}, None
    return None, reason
