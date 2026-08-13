import unittest

import numpy as np
import torch

np.int = int
np.float = float
np.bool = bool

from mpd.inference.collision_risk_selector import CollisionRiskSelector
from torch_robotics.robots import RobotPanda


class PointDistanceField:
    def __init__(self, target, fine_radii):
        self.target = target
        self.collision_margins = fine_radii

    def object_signed_distances(self, positions):
        return torch.linalg.norm(positions - self.target, dim=-1)


class ParentCollisionBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RobotPanda(
            tensor_args={"device": "cpu", "dtype": torch.float64}
        )

    def test_saved_parent_bounds_contain_every_fine_sphere(self):
        robot = self.robot
        sphere_parent = robot.collision_sphere_parent_indices
        bound_parent = robot.collision_parent_bound_parent_indices
        required = (
            torch.linalg.norm(
                robot.collision_sphere_local_positions[:, None, :]
                - robot.collision_parent_bound_local_centers[None, :, :],
                dim=-1,
            )
            + robot.link_collision_spheres_radii[:, None]
        )
        same_parent = sphere_parent[:, None] == bound_parent[None, :]
        covered = (
            (required <= robot.collision_parent_bound_radii[None, :]) & same_parent
        ).any(dim=1)
        self.assertTrue(covered.all().item())
        self.assertEqual(len(robot.collision_parent_bound_radii), 18)
        self.assertTrue(
            torch.all(
                torch.bincount(bound_parent, minlength=8)
                == torch.tensor([2, 2, 2, 2, 3, 2, 2, 3])
            ).item()
        )
        self.assertEqual(len(robot.link_collision_spheres_radii), 56)

    def test_parent_environment_clearance_is_a_lower_bound_of_fine_spheres(self):
        robot = self.robot
        selector = CollisionRiskSelector(
            robot,
            config={"coarse_points": 4},
            use_parent_link_kinematics=True,
            link_broad_phase_config={"enabled": True},
        )
        q = torch.tensor(
            [
                [[0.0] * 7, [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]],
                [[-0.2, 0.1, 0.0, -0.3, 0.1, -0.2, 0.2], [0.2] * 7],
            ],
            dtype=torch.float64,
        )
        field = PointDistanceField(
            torch.tensor([0.4, -0.1, 0.7], dtype=torch.float64),
            robot.link_collision_spheres_radii,
        )
        fine = selector.compute_clearances(q, field, robot.df_collision_self, True)
        coarse = selector.compute_parent_bound_clearances(
            q, field, robot.df_collision_self
        )
        fine_environment = fine[2]
        parent_environment = coarse[2]
        for parent_idx in range(len(robot.collision_sphere_unique_parent_links)):
            sphere_mask = robot.collision_sphere_parent_indices == parent_idx
            fine_min = fine_environment[..., sphere_mask].amin(dim=-1)
            self.assertTrue(
                torch.all(parent_environment[..., parent_idx] <= fine_min + 1e-12).item()
            )

    def test_parent_self_clearance_is_a_lower_bound_of_fine_pairs(self):
        robot = self.robot
        selector = CollisionRiskSelector(
            robot,
            config={"coarse_points": 4},
            use_parent_link_kinematics=True,
            link_broad_phase_config={"enabled": True},
        )
        q = torch.tensor(
            [[[0.0] * 7, [0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]]],
            dtype=torch.float64,
        )
        fine = selector.compute_clearances(q, None, robot.df_collision_self, True)[3]
        coarse = selector.compute_parent_bound_clearances(
            q, None, robot.df_collision_self
        )[3]
        fine_pair_1 = torch.as_tensor(robot.df_collision_self.link_idx_1)
        fine_pair_2 = torch.as_tensor(robot.df_collision_self.link_idx_2)
        fine_parent_1 = robot.collision_sphere_parent_indices[fine_pair_1]
        fine_parent_2 = robot.collision_sphere_parent_indices[fine_pair_2]
        fine_parent_pairs = torch.sort(
            torch.stack((fine_parent_1, fine_parent_2), dim=-1), dim=-1
        ).values
        for pair_idx, parent_pair in enumerate(robot.collision_parent_self_pairs):
            matching = torch.all(fine_parent_pairs == parent_pair, dim=-1)
            fine_min = fine[..., matching].amin(dim=-1)
            self.assertTrue(torch.all(coarse[..., pair_idx] <= fine_min + 1e-12).item())


if __name__ == "__main__":
    unittest.main()
