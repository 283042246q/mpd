import numpy as np
import pytest
from scipy import interpolate

from mpd.datasets.spacetime_legacy import open_uniform_knots
from mpd.parametric_trajectory.toppra_retiming import (
    build_multimodal_retiming_references,
    run_toppra,
)


toppra = pytest.importorskip("toppra")


def _rest_spatial_spline():
    control_points = np.zeros((12, 2), dtype=np.float64)
    control_points[3:-3, 0] = np.linspace(0.1, 0.9, 6)
    control_points[3:-3, 1] = np.sin(np.linspace(0.0, np.pi, 6)) * 0.2
    control_points[-3:, 0] = 1.0
    return interpolate.BSpline(open_uniform_knots(12, 5), control_points, 5, axis=0)


def test_toppra_retimes_exact_canonical_spline_with_limits():
    spline = _rest_spatial_spline()
    result = run_toppra(
        spline,
        dq_max=np.asarray([1.0, 1.0]),
        ddq_max=np.asarray([2.0, 2.0]),
        num_gridpoints=128,
    )

    assert result.duration > 0.0
    assert np.all(np.diff(result.time_from_start) > 0.0)
    assert result.path_velocity[0] == 0.0
    assert result.path_velocity[-1] == 0.0


def test_multimodal_retiming_contains_global_and_local_modes():
    spline = _rest_spatial_spline()
    phase = np.linspace(0.0, 1.0, 128)
    references = build_multimodal_retiming_references(
        spline,
        dq_max=np.asarray([1.0, 1.0]),
        ddq_max=np.asarray([2.0, 2.0]),
        phase=phase,
        rng=np.random.default_rng(3),
        num_toppra_gridpoints=128,
    )

    assert [reference.name for reference in references] == [
        "toppra",
        "duration_1.2",
        "duration_1.5",
        "limits_0.85_0.80",
        "limits_0.65_0.70",
        "local_slowdown",
        "near_wait",
    ]
    assert all(np.all(np.diff(reference.time_from_start) > 0.0) for reference in references)
    assert references[1].time_from_start[-1] > references[0].time_from_start[-1]
