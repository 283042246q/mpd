"""TOPP-RA teacher and multimodal retiming references for fixed spatial paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy import interpolate

from mpd.datasets.spacetime_schema import TimingSource


DEFAULT_RETIMING_VARIANTS = (
    "fast_anchor",
    "duration_1.5",
    "duration_2.0",
    "duration_2.5",
    "local_slowdown",
    "near_wait",
)


@dataclass(frozen=True)
class RetimingReference:
    name: str
    source: TimingSource
    time_from_start: np.ndarray
    mode_id: int
    metadata: dict


@dataclass(frozen=True)
class ToppraResult:
    phase: np.ndarray
    time_from_start: np.ndarray
    path_velocity: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.time_from_start[-1])


class ScipySplineGeometricPath:
    """Minimal TOPP-RA path adapter that preserves the canonical B-spline."""

    def __init__(self, spline: interpolate.BSpline) -> None:
        self.spline = spline
        value = np.asarray(spline(0.0))
        if value.ndim != 1:
            raise ValueError("spatial spline must evaluate to a joint vector")
        self._dof = int(value.shape[0])

    def __call__(self, path_positions, order: int = 0):
        if order < 0 or order > 2:
            raise ValueError("TOPP-RA only requests derivative orders 0, 1, and 2")
        return self.spline.derivative(order)(path_positions) if order else self.spline(path_positions)

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def path_interval(self) -> np.ndarray:
        return np.asarray([0.0, 1.0])

    @property
    def waypoints(self):
        return None

    def eval(self, path_positions):
        return self(path_positions, 0)

    def evald(self, path_positions):
        return self(path_positions, 1)

    def evaldd(self, path_positions):
        return self(path_positions, 2)


def run_toppra(
    spatial_spline: interpolate.BSpline,
    *,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    num_gridpoints: int = 256,
    velocity_scale: float = 1.0,
    acceleration_scale: float = 1.0,
) -> ToppraResult:
    """Compute a rest-to-rest fixed-path parameterization using true limits."""

    try:
        import toppra.algorithm as algo
        import toppra.constraint as constraint
    except ImportError as error:
        raise RuntimeError(
            "TOPP-RA is required; run this generator in the mpd-splines-public environment"
        ) from error

    dq_max = np.asarray(dq_max, dtype=np.float64) * float(velocity_scale)
    ddq_max = np.asarray(ddq_max, dtype=np.float64) * float(acceleration_scale)
    if dq_max.ndim != 1 or ddq_max.shape != dq_max.shape:
        raise ValueError("dq_max and ddq_max must be same-shaped joint vectors")
    if not np.all(dq_max > 0.0) or not np.all(ddq_max > 0.0):
        raise ValueError("scaled velocity and acceleration limits must remain positive")
    gridpoints = np.linspace(0.0, 1.0, int(num_gridpoints))
    path = ScipySplineGeometricPath(spatial_spline)
    velocity_constraint = constraint.JointVelocityConstraint(
        np.column_stack((-dq_max, dq_max))
    )
    acceleration_constraint = constraint.JointAccelerationConstraint(
        np.column_stack((-ddq_max, ddq_max)),
        discretization_scheme=constraint.DiscretizationType.Interpolation,
    )
    instance = algo.TOPPRA(
        [velocity_constraint, acceleration_constraint],
        path,
        gridpoints=gridpoints,
        parametrizer="ParametrizeConstAccel",
    )
    _, path_velocity, _ = instance.compute_parameterization(0.0, 0.0)
    if path_velocity is None or not np.all(np.isfinite(path_velocity)):
        raise RuntimeError("TOPP-RA could not parameterize the spatial path")
    denominator = path_velocity[:-1] + path_velocity[1:]
    if np.any(denominator <= np.finfo(np.float64).eps):
        raise RuntimeError("TOPP-RA returned a zero-speed phase interval")
    delta_time = 2.0 * np.diff(gridpoints) / denominator
    time_from_start = np.concatenate(([0.0], np.cumsum(delta_time)))
    if not np.all(np.diff(time_from_start) > 0.0):
        raise RuntimeError("TOPP-RA reference time is not strictly increasing")
    return ToppraResult(gridpoints, time_from_start, path_velocity)


def resample_reference(reference: ToppraResult, phase: np.ndarray) -> np.ndarray:
    result = np.interp(phase, reference.phase, reference.time_from_start)
    result[0] = 0.0
    if not np.all(np.diff(result) > 0.0):
        raise RuntimeError("resampled TOPP-RA reference is not strictly increasing")
    return result


def _integrate_density(phase: np.ndarray, density: np.ndarray) -> np.ndarray:
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(phase)
    return np.concatenate(([0.0], np.cumsum(increments)))


def build_toppra_anchor_reference(
    spatial_spline: interpolate.BSpline,
    *,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    phase: np.ndarray,
    num_toppra_gridpoints: int = 256,
) -> RetimingReference:
    """Build the raw TOPP-RA reference that must be made feasible after fitting."""

    anchor = run_toppra(
        spatial_spline,
        dq_max=dq_max,
        ddq_max=ddq_max,
        num_gridpoints=num_toppra_gridpoints,
    )
    return RetimingReference(
        "fast_anchor",
        TimingSource.TOPPRA,
        resample_reference(anchor, phase),
        0,
        {"duration_scale": 1.0, "stage": "raw_toppra"},
    )


def build_retiming_references_from_feasible_anchor(
    feasible_anchor_time: np.ndarray,
    *,
    phase: np.ndarray,
    rng: np.random.Generator,
    variant_names: Optional[Sequence[str]] = None,
    duration_max: Optional[float] = None,
    duration_margin: float = 0.1,
) -> List[RetimingReference]:
    """Create timing modes from an already fitted and validated anchor.

    Applying duration scales here, rather than to raw TOPP-RA output, prevents
    the scaled modes from collapsing back onto the same feasibility boundary.
    """

    selected = set(variant_names or DEFAULT_RETIMING_VARIANTS)
    unknown = sorted(selected - set(DEFAULT_RETIMING_VARIANTS))
    if unknown:
        raise ValueError(f"unknown retiming variants: {unknown}")
    phase = np.asarray(phase, dtype=np.float64)
    anchor_time = np.asarray(feasible_anchor_time, dtype=np.float64)
    if anchor_time.shape != phase.shape:
        raise ValueError("feasible_anchor_time and phase must have the same shape")
    anchor_time = anchor_time - anchor_time[0]
    if anchor_time[-1] <= 0.0 or not np.all(np.diff(anchor_time) > 0.0):
        raise ValueError("feasible_anchor_time must be strictly increasing")
    if duration_max is not None:
        if duration_margin < 0.0 or duration_max <= duration_margin:
            raise ValueError("duration margin must be non-negative and below duration_max")
        if anchor_time[-1] > duration_max:
            raise ValueError("feasible anchor exceeds duration_max")

    def bounded_amplitude(requested: float, bump: np.ndarray) -> float:
        if duration_max is None:
            return requested
        added_duration_per_unit = float(np.trapz(anchor_density * bump, phase))
        available_duration = max(0.0, duration_max - duration_margin - anchor_time[-1])
        if added_duration_per_unit <= 0.0:
            return 0.0
        return min(requested, available_duration / added_duration_per_unit)

    references: List[RetimingReference] = []
    if "fast_anchor" in selected:
        references.append(
            RetimingReference(
                "fast_anchor",
                TimingSource.TOPPRA,
                anchor_time.copy(),
                0,
                {"duration_scale": 1.0, "stage": "feasible_anchor"},
            )
        )
    for mode_id, scale in ((1, 1.5), (2, 2.0), (3, 2.5)):
        name = f"duration_{scale:.1f}"
        if name in selected:
            references.append(
                RetimingReference(
                    name,
                    TimingSource.DURATION_SCALED,
                    anchor_time * scale,
                    mode_id,
                    {"duration_scale": scale, "anchor": "feasible_anchor"},
                )
            )

    anchor_density = np.gradient(anchor_time, phase, edge_order=2)
    if "local_slowdown" in selected:
        center = float(rng.uniform(0.25, 0.75))
        sigma = float(rng.uniform(0.08, 0.14))
        requested_amplitude = float(rng.uniform(0.6, 1.2))
        bump = np.exp(-0.5 * np.square((phase - center) / sigma))
        amplitude = bounded_amplitude(requested_amplitude, bump)
        density = anchor_density * (1.0 + amplitude * bump)
        references.append(
            RetimingReference(
                "local_slowdown",
                TimingSource.LOCAL_SLOWDOWN,
                _integrate_density(phase, density),
                4,
                {
                    "center": center,
                    "sigma": sigma,
                    "amplitude": amplitude,
                    "requested_amplitude": requested_amplitude,
                },
            )
        )
    if "near_wait" in selected:
        center = float(rng.uniform(0.3, 0.7))
        # Eight c coefficients and five gauge-free r coefficients cannot
        # faithfully represent an impulse-like stop. Keep the bump narrow
        # enough to resemble waiting, but wide enough to survive both fits.
        sigma = float(rng.uniform(0.06, 0.09))
        requested_amplitude = float(rng.uniform(2.5, 4.0))
        bump = np.exp(-0.5 * np.square((phase - center) / sigma))
        amplitude = bounded_amplitude(requested_amplitude, bump)
        density = anchor_density * (1.0 + amplitude * bump)
        references.append(
            RetimingReference(
                "near_wait",
                TimingSource.NEAR_WAIT,
                _integrate_density(phase, density),
                5,
                {
                    "center": center,
                    "sigma": sigma,
                    "amplitude": amplitude,
                    "requested_amplitude": requested_amplitude,
                },
            )
        )
    return references


def build_multimodal_retiming_references(
    spatial_spline: interpolate.BSpline,
    *,
    dq_max: np.ndarray,
    ddq_max: np.ndarray,
    phase: np.ndarray,
    rng: np.random.Generator,
    num_toppra_gridpoints: int = 256,
    variant_names: Optional[Sequence[str]] = None,
) -> List[RetimingReference]:
    """Compatibility helper using raw TOPP-RA as the anchor.

    Dataset generation should validate the result of
    :func:`build_toppra_anchor_reference` first and then call
    :func:`build_retiming_references_from_feasible_anchor`.
    """

    anchor = build_toppra_anchor_reference(
        spatial_spline,
        dq_max=dq_max,
        ddq_max=ddq_max,
        phase=phase,
        num_toppra_gridpoints=num_toppra_gridpoints,
    )
    return build_retiming_references_from_feasible_anchor(
        anchor.time_from_start,
        phase=phase,
        rng=rng,
        variant_names=variant_names,
    )
