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


def test_shape_and_event_randomization_covers_every_object_position_and_count():
    centers = np.tile(np.array([0.45, 0., 0.5]), (64, 10, 1))
    for count in (1, 2, 3, 4, 5):
        shapes, fractions = [], []
        for seed in range(96):
            forecast = make_forecast(centers, 8., 14., np.random.default_rng(seed), object_count=count)
            shapes.append(forecast.shape)
            fractions.append(forecast.event_times / 8.)
            assert interaction_audit(forecast, 8.)["all_objects_interact_during_execution"]
            for obj, event in enumerate(forecast.event_times):
                index = np.flatnonzero(forecast.times == event)[0]
                np.testing.assert_allclose(forecast.positions[index, obj], forecast.anchors[obj], atol=1e-12)
                size = forecast.size[obj]
                assert 0.04 <= size[0] <= 0.075
                if forecast.shape[obj] == 0:
                    assert size[1] == size[2] == 0
                elif forecast.shape[obj] == 1:
                    assert (size > 0).all()
                else:
                    assert size[1] > 0 and size[2] == 0
        shapes, fractions = np.asarray(shapes), np.asarray(fractions)
        assert ((fractions >= 0.25) & (fractions <= 0.65)).all()
        for obj in range(count):
            assert set(shapes[:, obj]) == {0, 1, 2}
            assert fractions[:, obj].min() < 0.35 and fractions[:, obj].max() > 0.55
        if count > 1:
            assert (np.diff(fractions, axis=1) < 0).any()  # no object-order time binding


def test_randomized_forecast_is_reproducible_and_seed_changes_the_world():
    centers = np.tile(np.array([0.45, 0., 0.5]), (64, 10, 1))
    first, repeated, other = [make_forecast(centers, 8., 14., np.random.default_rng(seed), object_count=5)
                              for seed in (71, 71, 72)]
    for key in ("times", "positions", "shape", "size", "event_times", "anchors"):
        np.testing.assert_array_equal(getattr(first, key), getattr(repeated, key))
    assert first.modes == repeated.modes
    assert not np.array_equal(first.event_times, other.event_times)
    assert not np.array_equal(first.positions, other.positions)


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
