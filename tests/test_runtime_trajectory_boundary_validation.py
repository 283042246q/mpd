from types import SimpleNamespace

import pytest
import torch

from scripts.runtime.infer_once import ResultValidationError, _validate_best_trajectory


class _PlanningTask:
    margin_for_dense_collision_checking = 0.0
    robot = SimpleNamespace(
        q_pos_min=torch.full((7,), -2.0),
        q_pos_max=torch.full((7,), 2.0),
        dq_max=torch.full((7,), 2.0),
        ddq_max=torch.full((7,), 4.0),
    )

    @staticmethod
    def compute_collision(positions, margin):
        del margin
        return torch.zeros(positions.shape[0], dtype=torch.bool)


def _results():
    timesteps = torch.linspace(0.0, 1.0, 4)
    return SimpleNamespace(
        q_trajs_pos_best=torch.zeros((4, 7)),
        q_trajs_vel_best=torch.zeros((4, 7)),
        q_trajs_acc_best=torch.zeros((4, 7)),
        timesteps=timesteps,
    )


def _validate(results):
    return _validate_best_trajectory(
        results,
        _PlanningTask(),
        torch.zeros(7),
        torch.zeros(7),
        torch.zeros(7),
        expected_horizon=4,
        expected_duration=1.0,
    )


def test_runtime_explicitly_validates_all_three_start_boundaries():
    validation = _validate(_results())

    assert validation["start_max_abs_error_rad"] == 0.0
    assert validation["start_velocity_max_abs_error_rad_s"] == 0.0
    assert validation["start_acceleration_max_abs_error_rad_s2"] == 0.0


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("q_trajs_vel_best", "q_vel_start"),
        ("q_trajs_acc_best", "q_acc_start"),
    ),
)
def test_runtime_rejects_derivative_start_mismatch(field, message):
    results = _results()
    getattr(results, field)[0, 2] = 1e-3

    with pytest.raises(ResultValidationError, match=message):
        _validate(results)
