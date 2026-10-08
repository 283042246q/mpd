import numpy as np
import torch

from scripts.spacetime_data.batch_spacetime_rrt import TorchSpaceTimeForest, edge_collision_mask
from scripts.spacetime_data.warehouse_world import Forecast, ForecastBatch, MOTION_MODES, make_forecast, interaction_audit


def test_wait_is_checked_in_time_and_query_worlds_do_not_mix():
    starts = torch.tensor([[[0., 0.]], [[0., 0.]]])
    goals = torch.tensor([[[0., 2.]], [[0., 2.]]])
    ids = torch.tensor([[0], [1]])

    def collision(q, t, query):
        return (query == 0) & (t > 0.9) & (t < 1.1)

    blocked, checked = edge_collision_mask(starts, goals, ids, collision, time_step=0.05)
    assert blocked.tolist() == [[True], [False]]
    assert checked == 82
    backward, _ = edge_collision_mask(goals, starts, ids, collision, time_step=0.05)
    assert torch.equal(backward, blocked)


def test_forecast_interpolation_ragged_objects_and_horizon_rejection():
    first = Forecast(np.array([0., 2.]), np.array([[[0., 0., 0.]], [[2., 0., 0.]]]),
                     np.array([0]), np.array([[0.1, 0., 0.]]), np.array([1.]), ["linear"], np.array([[1., 0., 0.]]))
    second = Forecast(np.array([0., 1., 2.]), np.array([[[5., 0., 0.]], [[6., 0., 0.]], [[7., 0., 0.]]]),
                      np.array([0]), np.array([[0.1, 0., 0.]]), np.array([1.]), ["linear"], np.array([[6., 0., 0.]]))
    batch = ForecastBatch([first, second])
    ids, times = torch.tensor([0, 1, 0]), torch.tensor([1., 1., 3.])
    assert torch.allclose(batch.centers(times, ids)[:, 0, 0], torch.tensor([1., 6., 2.]))
    clearance = batch.clearance(torch.tensor([[[1., 0., 0.]]] * 3), torch.tensor([0.1]), times, ids)
    assert clearance[0] < 0 and clearance[1] > 0 and torch.isneginf(clearance[2])


def test_all_motion_modes_hit_their_saved_anchor_during_execution():
    centers = np.tile(np.array([0.45, 0., 0.5]), (64, 10, 1))
    forecast = make_forecast(centers, 8., 14., np.random.default_rng(12),
                             object_count=len(MOTION_MODES), modes=MOTION_MODES)
    for obj, event in enumerate(forecast.event_times):
        index = np.flatnonzero(forecast.times == event)[0]
        np.testing.assert_allclose(forecast.positions[index, obj], forecast.anchors[obj], atol=1e-12)
    assert interaction_audit(forecast, 8.)["all_objects_interact_during_execution"]
    assert not interaction_audit(forecast, 0.5)["all_objects_interact_during_execution"]
    assert forecast.metadata()["obstacle_static_collision"] == "ignored"


def test_forest_returns_velocity_valid_monotone_paths_and_isolates_failure():
    def collision(q, t, query):
        return query == 1  # one invalid world must not poison the other

    planner = TorchSpaceTimeForest(collision, torch.tensor([-1.]), torch.tensor([1.]),
                                   torch.tensor([1.]), edges_per_query=8, roots=4,
                                   max_iterations=80, nearest_chunk=4, time_step=0.1)
    results = planner.plan_batch([[-0.8], [-0.8]], [[0.8], [0.8]], [4., 4.], [2., 2.],
                                 allowed_time=5., seed=123)
    assert results[0].paths and not results[1].paths
    for path in results[0].paths:
        assert np.all(np.diff(path[:, -1]) > 0)
        assert np.all(np.abs(np.diff(path[:, 0])) <= np.diff(path[:, -1]) + 1e-5)
        np.testing.assert_allclose(path[[0, -1], 0], [-0.8, 0.8], atol=1e-6)
        assert 2 <= path[-1, -1] <= 4
    arrivals = sorted(p[-1, -1] for p in results[0].paths)
    assert arrivals[0] <= arrivals[-1]


def test_temporarily_closed_passage_is_not_treated_as_static():
    def collision(q, t, query):
        return (q[:, 0].abs() < 0.2) & (t < 1.8)

    planner = TorchSpaceTimeForest(collision, torch.tensor([-1.]), torch.tensor([1.]),
                                   torch.tensor([1.]), edges_per_query=16, roots=4,
                                   max_iterations=100, time_step=0.04)
    result = planner.plan_batch([[-0.8]], [[0.8]], [5.], [3.], allowed_time=5., seed=7)[0]
    assert result.paths
    for path in result.paths:
        starts, goals = torch.tensor(path[:-1])[None], torch.tensor(path[1:])[None]
        blocked, _ = edge_collision_mask(starts, goals, torch.zeros(starts.shape[:2], dtype=torch.long),
                                         collision, joint_step=0.01, time_step=0.01)
        assert not blocked.any()
