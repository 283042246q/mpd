import numpy as np
import torch

from scripts.generate_data.gpu_batch_rrt import (
    GpuBatchRRTConnect,
    GpuMultiQueryRRTConnect,
)


def test_gpu_batch_rrt_connects_free_space_and_freezes_inactive_joints():
    def collision(states):
        return torch.zeros(len(states), dtype=torch.bool, device=states.device)

    planner = GpuBatchRRTConnect(
        collision,
        torch.tensor([-1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
        batch_size=8,
        max_iterations=20,
        extension_range=0.3,
        collision_step=0.05,
        seed=7,
    )
    result = planner.plan(
        torch.tensor([-0.8, 3.0, -0.8, 4.0]),
        torch.tensor([0.8, 99.0, 0.8, -99.0]),
        active_indices=[0, 2],
        allowed_time=1.0,
    )

    assert result.path is not None
    np.testing.assert_allclose(result.path[0], [-0.8, 3.0, -0.8, 4.0])
    np.testing.assert_allclose(result.path[-1], [0.8, 3.0, 0.8, 4.0])
    np.testing.assert_array_equal(result.path[:, [1, 3]], [[3.0, 4.0]] * len(result.path))
    assert result.checked_states > result.sampled_edges


def test_gpu_batch_rrt_does_not_cross_an_impenetrable_collision_band():
    def collision(states):
        return states[:, 0].abs() < 0.15

    planner = GpuBatchRRTConnect(
        collision,
        torch.tensor([-1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
        batch_size=8,
        max_iterations=12,
        extension_range=0.3,
        collision_step=0.025,
        seed=11,
    )
    result = planner.plan(
        torch.tensor([-0.8, 0.0]),
        torch.tensor([0.8, 0.0]),
        active_indices=[0, 1],
        allowed_time=1.0,
    )

    assert result.path is None


def test_gpu_multi_query_rrt_plans_independent_requests_and_freezes_joints():
    def collision(states):
        return torch.zeros(len(states), dtype=torch.bool, device=states.device)

    planner = GpuMultiQueryRRTConnect(
        collision,
        torch.tensor([-1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
        batch_size=4,
        max_iterations=30,
        extension_range=0.3,
        collision_step=0.05,
        seed=17,
    )
    starts = torch.tensor(
        [[-0.8, 3.0, -0.8, 4.0], [-0.7, 5.0, 0.6, 6.0]]
    )
    goals = torch.tensor(
        [[0.8, 99.0, 0.8, -99.0], [0.5, -99.0, -0.6, 99.0]]
    )
    results = planner.plan_batch(
        starts,
        goals,
        active_indices=[0, 2],
        allowed_time=2.0,
        query_seeds=[101, 202],
    )

    assert len(results) == 2
    for query, result in enumerate(results):
        assert result.path is not None
        np.testing.assert_allclose(result.path[0], starts[query].numpy())
        np.testing.assert_allclose(
            result.path[-1, [0, 2]], goals[query, [0, 2]].numpy()
        )
        np.testing.assert_array_equal(
            result.path[:, [1, 3]],
            np.repeat(starts[query, [1, 3]][None].numpy(), len(result.path), axis=0),
        )


def test_gpu_multi_query_rrt_isolates_invalid_and_unsolved_queries():
    def collision(states):
        invalid_endpoint = states[:, 1] > 0.9
        separating_band = states[:, 0].abs() < 0.15
        return invalid_endpoint | separating_band

    planner = GpuMultiQueryRRTConnect(
        collision,
        torch.tensor([-1.0]),
        torch.tensor([1.0]),
        batch_size=4,
        max_iterations=12,
        extension_range=0.3,
        collision_step=0.025,
        seed=23,
    )
    results = planner.plan_batch(
        torch.tensor([[-0.8, 0.0], [-0.8, 1.0]]),
        torch.tensor([[0.8, 0.0], [0.8, 1.0]]),
        active_indices=[0],
        allowed_time=1.0,
        query_seeds=[303, 404],
    )

    assert [result.path for result in results] == [None, None]
    assert results[0].iterations > 0
    assert results[1].iterations == 0
    assert results[1].checked_states == 2
