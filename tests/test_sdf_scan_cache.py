import unittest

import torch

from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    CollisionObjectDistanceField,
)


class CountingSDF:
    def __init__(self):
        self.distance_calls = 0
        self.gradient_calls = 0

    @staticmethod
    def _values(x):
        return x[..., 0] - 0.25

    @staticmethod
    def _gradients(x):
        gradient = torch.zeros_like(x)
        gradient[..., 0] = 1.0
        return gradient

    def compute_signed_distance(self, x, get_gradient=False):
        self.distance_calls += 1
        values = self._values(x)
        if get_gradient:
            self.gradient_calls += 1
            return values, self._gradients(x)
        return values

    def compute_signed_distance_gradient(self, x):
        self.gradient_calls += 1
        return self._gradients(x)


class SDFScanCacheTest(unittest.TestCase):
    def test_precomputed_distance_uses_gradient_only_query(self):
        primitive = CountingSDF()
        field = CollisionObjectDistanceField(
            robot=object(),
            df_obj_list_fn=lambda: [primitive],
            link_margins_for_object_collision_checking_tensor=torch.tensor(
                [0.1, 0.2], dtype=torch.float64
            ),
            cutoff_margin=0.01,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        positions = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]],
                    [[0.3, 0.0, 0.0], [0.8, 0.0, 0.0]],
                ]
            ],
            dtype=torch.float64,
        )
        raw_distances = field.object_signed_distances(positions)
        self.assertEqual(primitive.distance_calls, 1)

        cached_cost, cached_gradient = (
            field.compute_embodiment_taskspace_sdf_and_gradient(
                positions, precomputed_sdf_values=raw_distances
            )
        )
        self.assertEqual(primitive.distance_calls, 1)
        self.assertEqual(primitive.gradient_calls, 1)

        reference_cost, reference_gradient = (
            field.compute_embodiment_taskspace_sdf_and_gradient(positions)
        )
        self.assertEqual(primitive.distance_calls, 2)
        torch.testing.assert_close(cached_cost, reference_cost)
        torch.testing.assert_close(cached_gradient, reference_gradient)


if __name__ == "__main__":
    unittest.main()
