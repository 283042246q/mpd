import numpy as np
import torch

from scripts.generate_data.gpu_batch_rrt import GpuBatchRRTConnect


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
