from types import SimpleNamespace

import numpy as np
import torch

from scripts.spacetime_data.warehouse_trajectory_fit import fit_spatial_time_path, fit_teacher
from scripts.spacetime_data.warehouse_world import Forecast


class CollisionFixture:
    margin = 0.005

    def __init__(self, blocked=False):
        self.blocked = blocked
        self.calls = []
        self.robot = SimpleNamespace(q_pos_min=torch.full((7,), -2.), q_pos_max=torch.full((7,), 2.),
                                     dq_max=torch.ones(7), ddq_max=torch.full((7,), 3.))

    def clearances(self, q, times, query_ids):
        self.calls.append(np.asarray(times))
        result = torch.ones((len(q), 3))
        if self.blocked:
            mask = (np.asarray(times) > 1.) & (np.asarray(times) < 2.)
            result[mask, 2] = -0.1
        return result


def forecast_fixture():
    return Forecast(np.array([0., 1.5, 14.]), np.tile([[[0.4, 0., 0.5]]], (3, 1, 1)),
                     np.array([0]), np.array([[0.05, 0., 0.]]), np.array([1.5]), ["linear"], np.array([[0.4, 0., 0.5]]))


def raw_line():
    q = np.zeros((8, 7))
    q[:, 0] = np.linspace(-0.5, 0.5, 8)
    return np.c_[q, np.linspace(0, 5., 8)]


def test_phase_keeps_wait_time_and_rest_boundary():
    raw = raw_line()
    raw[1, :7] = raw[0, :7]
    spline, phase, error = fit_spatial_time_path(raw)
    assert phase[1] > phase[0]
    assert error < 0.08
    np.testing.assert_allclose(spline([0, 1]), raw[[0, -1], :7], atol=1e-12)
    np.testing.assert_allclose(spline.derivative(1)([0, 1]), 0, atol=1e-12)
    np.testing.assert_allclose(spline.derivative(2)([0, 1]), 0, atol=1e-12)


def test_safe_fit_checks_physics_and_tau_r_independently():
    fixture = CollisionFixture()
    result, reason = fit_teacher(raw_line(), fixture, 0, forecast_fixture(), require_tau_r=True)
    assert result is not None, reason
    assert result["tau_r_valid"]
    assert result["quality"]["v_ratio_max"] <= 1.001
    assert result["quality"]["a_ratio_max"] <= 1.001
    assert len(fixture.calls) >= 2
    np.testing.assert_allclose(result["execution_dq"][[0, -1]], 0, atol=1e-8)
    np.testing.assert_allclose(result["execution_ddq"][[0, -1]], 0, atol=1e-8)


def test_fit_never_accepts_dynamic_collision_after_retiming():
    fixture = CollisionFixture(blocked=True)
    result, reason = fit_teacher(raw_line(), fixture, 0, forecast_fixture())
    assert result is None and reason == "dynamic_collision"
    assert fixture.calls and all((times > 1).any() for times in fixture.calls)
