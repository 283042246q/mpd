import numpy as np
import torch

from mpd.parametric_trajectory.timing_fitting import (
    TimingSplineNumpy,
    decode_timing_latent,
    encode_timing_control_points,
    fit_and_validate_timing_reference,
    fit_timing_reference,
)
from mpd.datasets.spacetime_legacy import open_uniform_knots
from scipy import interpolate
from mpd.parametric_trajectory.timing_spline import TimingSpline


def test_numpy_timing_spline_matches_runtime_torch_implementation():
    numpy_spline = TimingSplineNumpy()
    control_points = decode_timing_latent(np.asarray([1.2, 1.5, 1.9, 1.1, 2.0, 1.4]))
    numpy_result = numpy_spline.evaluate(control_points)

    torch_spline = TimingSpline(tensor_args={"device": "cpu", "dtype": torch.float64})
    torch_result = torch_spline.evaluate(torch.as_tensor(control_points, dtype=torch.float64))

    np.testing.assert_allclose(numpy_result.u, torch_result.u.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        numpy_result.time_from_start,
        torch_result.time_from_start.numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_timing_latent_codec_roundtrip():
    latent = np.arange(6, dtype=np.float64)
    np.testing.assert_array_equal(encode_timing_control_points(decode_timing_latent(latent)), latent)


def test_fit_timing_reference_recovers_representable_curve():
    spline = TimingSplineNumpy()
    expected = decode_timing_latent(np.asarray([1.0, 1.5, 2.0, 2.3, 1.7, 1.2]))
    reference = spline.evaluate(expected).time_from_start

    fitted = fit_timing_reference(reference, timing_spline=spline, smoothness_weight=0.0)

    assert fitted.success
    assert fitted.relative_rmse < 1e-7
    assert np.all(np.diff(fitted.time_from_start) > 0.0)


def test_validation_slows_infeasible_fitted_timing_until_limits_hold():
    control_points = np.zeros((12, 2), dtype=np.float64)
    control_points[3:-3, 0] = np.linspace(0.1, 0.9, 6)
    control_points[-3:, 0] = 1.0
    spatial = interpolate.BSpline(open_uniform_knots(12, 5), control_points, 5, axis=0)
    timing_spline = TimingSplineNumpy()
    fast_reference = timing_spline.phase * 0.4

    result = fit_and_validate_timing_reference(
        fast_reference,
        spatial_spline=spatial,
        q_min=np.asarray([-2.0, -2.0]),
        q_max=np.asarray([2.0, 2.0]),
        dq_max=np.asarray([0.5, 0.5]),
        ddq_max=np.asarray([1.0, 1.0]),
        timing_spline=timing_spline,
        duration_min=0.2,
        duration_max=20.0,
    )

    assert result.accepted
    assert result.reference_scale > 1.0
    assert result.v_ratio_max <= 1.001
    assert result.a_ratio_max <= 1.001
