import unittest
from types import SimpleNamespace

import torch

from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionSelfField


class SelfCollisionExplicitGradientTest(unittest.TestCase):
    def test_matches_autograd_for_unique_deepest_pair(self):
        field = CollisionSelfField(
            robot=object(),
            link_self_collision_tuples=[(0, 1, 0.5, 0.5), (1, 2, 0.1, 0.1)],
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        positions = torch.tensor(
            [[[0.0, 0.0], [0.4, 0.0], [1.5, 0.2]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        distances = field.compute_embodiment_signed_distances(None, positions)
        reference_cost = torch.max(torch.relu(-distances), dim=-1)[0]
        reference_gradient = torch.autograd.grad(reference_cost.sum(), positions)[0]
        cost, gradient = field.compute_distance_field_cost_and_gradient(positions.detach())
        torch.testing.assert_close(cost, reference_cost.detach())
        torch.testing.assert_close(gradient, reference_gradient)

    def test_coincident_centers_return_finite_zero_subgradient(self):
        field = CollisionSelfField(
            robot=object(),
            link_self_collision_tuples=[(0, 1, 0.5, 0.5)],
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        positions = torch.zeros(1, 2, 3, dtype=torch.float64)
        cost, gradient = field.compute_distance_field_cost_and_gradient(positions)
        self.assertTrue(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
        self.assertEqual(cost.item(), 1.0)

    def test_subset_spheres_and_pairs_match_full_active_pair(self):
        robot = SimpleNamespace(link_collision_spheres_names=["s0", "s1", "s2", "s3"])
        field = CollisionSelfField(
            robot=robot,
            link_self_collision_tuples=[
                (0, 1, 0.5, 0.5),
                (1, 2, 0.2, 0.2),
                (2, 3, 0.1, 0.1),
            ],
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        positions = torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [2.3, 0.0], [2.35, 0.0]]],
            dtype=torch.float64,
        )
        full_cost, full_gradient = field.compute_distance_field_cost_and_gradient(positions)
        sphere_indices = torch.tensor([1, 2, 3])
        subset_cost, subset_gradient = field.compute_distance_field_cost_and_gradient(
            positions[:, sphere_indices],
            link_indices=sphere_indices,
            self_pair_indices=torch.tensor([1, 2]),
        )
        torch.testing.assert_close(subset_cost, full_cost)
        torch.testing.assert_close(subset_gradient, full_gradient[:, sphere_indices])


if __name__ == "__main__":
    unittest.main()
