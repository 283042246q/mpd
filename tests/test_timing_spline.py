import pytest
import torch

from mpd.parametric_trajectory.timing_spline import TimingSpline
from mpd.parametric_trajectory.trajectory_bspline import BSpline


TENSOR_ARGS = {"device": torch.device("cpu"), "dtype": torch.float64}


def _timing(**kwargs):
    return TimingSpline(
        num_control_points=8,
        degree=3,
        num_phase_points=65,
        u_min=0.05,
        duration_min=2.0,
        duration_max=15.0,
        tensor_args=TENSOR_ARGS,
        **kwargs,
    )


def test_linear_ten_second_timing_matches_fixed_phase_grid():
    timing = _timing()
    control_points = timing.linear_control_points(10.0, batch_shape=(3,))
    result = timing.evaluate(control_points, require_duration_bounds=True)

    torch.testing.assert_close(result.u, torch.full((3, 65), 10.0, **TENSOR_ARGS))
    torch.testing.assert_close(result.u_s, torch.zeros((3, 65), **TENSOR_ARGS), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        result.time_from_start,
        torch.linspace(0.0, 10.0, 65, **TENSOR_ARGS).expand(3, -1),
    )
    torch.testing.assert_close(result.duration, torch.full((3,), 10.0, **TENSOR_ARGS))


def test_endpoint_projection_is_monotone_and_fixes_density_derivative():
    timing = _timing()
    raw = timing.linear_control_points(7.0, batch_shape=(2,))
    raw[:, 2:-2] += torch.tensor([[0.4, -0.2, 0.3, -0.1], [-0.5, 0.1, 0.2, 0.4]], **TENSOR_ARGS)
    control_points = timing.enforce_endpoint_derivatives(raw)
    result = timing.evaluate(control_points, require_duration_bounds=True)

    assert (result.u > timing.u_min).all().item()
    assert (torch.diff(result.time_from_start, dim=-1) > 0.0).all().item()
    torch.testing.assert_close(result.u_s[..., (0, -1)], torch.zeros((2, 2), **TENSOR_ARGS), atol=1e-12, rtol=0)
    assert timing.optimizable_mask.tolist() == [False, False, True, True, True, True, False, False]


def test_duration_bounds_and_invalid_values_fail_fast():
    timing = _timing()
    with pytest.raises(ValueError, match="duration"):
        timing.evaluate(timing.linear_control_points(1.0), require_duration_bounds=True)
    with pytest.raises(ValueError, match="duration"):
        timing.evaluate(timing.linear_control_points(16.0), require_duration_bounds=True)
    invalid = timing.linear_control_points(10.0)
    invalid[3] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        timing.evaluate(invalid)
    invalid = timing.linear_control_points(10.0)
    invalid[1] += 0.1
    with pytest.raises(ValueError, match="endpoint"):
        timing.evaluate(invalid)


def test_space_time_chain_rule_matches_central_finite_difference():
    timing = _timing()
    spatial = BSpline(num_pts=7, degree=3, num_T_pts=65, **TENSOR_ARGS)
    spatial_control_points = torch.tensor(
        [[[0.0], [0.1], [0.35], [0.8], [1.1], [1.25], [1.4]]], **TENSOR_ARGS
    )
    timing_control_points = timing.linear_control_points(6.0, batch_shape=(1,))
    timing_control_points[:, 2:-2] += torch.tensor([[0.3, 0.1, -0.2, 0.2]], **TENSOR_ARGS)
    result = timing.evaluate_spatial_control_points(
        timing_control_points,
        spatial_control_points,
        position_basis=spatial.N.squeeze(0),
        velocity_basis=spatial.dN.squeeze(0),
        acceleration_basis=spatial.ddN.squeeze(0),
    )

    q = result.q[0, :, 0]
    t = result.time_from_start[0]
    central_dq = (q[2:] - q[:-2]) / (t[2:] - t[:-2])
    central_ddq = (result.dq[0, 2:, 0] - result.dq[0, :-2, 0]) / (t[2:] - t[:-2])
    torch.testing.assert_close(result.dq[0, 1:-1, 0], central_dq, atol=4e-3, rtol=2e-2)
    torch.testing.assert_close(result.ddq[0, 1:-1, 0], central_ddq, atol=3e-2, rtol=5e-2)


def test_duration_autograd_matches_control_point_central_difference():
    timing = _timing()
    control_points = timing.linear_control_points(8.0)
    control_points[2:-2] += torch.tensor([0.2, -0.1, 0.3, -0.2], **TENSOR_ARGS)
    control_points.requires_grad_(True)
    duration = timing.evaluate(control_points).duration
    gradient = torch.autograd.grad(duration, control_points)[0]

    epsilon = 1e-5
    finite_difference = torch.zeros_like(control_points)
    for index in range(control_points.numel()):
        plus = control_points.detach().clone()
        minus = control_points.detach().clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (
            timing.evaluate(plus, require_fixed_endpoint_derivatives=False).duration
            - timing.evaluate(minus, require_fixed_endpoint_derivatives=False).duration
        ) / (2.0 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, atol=1e-8, rtol=1e-7)
