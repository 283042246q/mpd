import unittest

import numpy as np
import torch

np.int = int
np.float = float
np.bool = bool

from torch_robotics.robots import RobotPanda
from mpd.inference.active_jacobian import ActiveJacobianComputer
from mpd.inference.collision_risk_selector import TemporalSelection
from mpd.inference.collision_risk_selector import FineSphereScanCache


class ParentLinkSphereKinematicsTest(unittest.TestCase):
    def test_parent_link_pose_and_spatial_jacobian_match_virtual_spheres(self):
        robot = RobotPanda(tensor_args={"device": "cpu", "dtype": torch.float64})
        q = torch.stack(
            [
                torch.zeros(7, dtype=torch.float64),
                torch.tensor([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1], dtype=torch.float64),
            ]
        )
        jacobian_legacy, pose_legacy = robot.jfk_s_collision_spheres(q)
        jacobian_parent, pose_parent = robot.jfk_s_collision_spheres_parent_links(q)
        self.assertLess(len(robot.collision_sphere_unique_parent_links), len(robot.link_collision_spheres_names))
        torch.testing.assert_close(torch.stack(pose_parent), torch.stack(pose_legacy), rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(
            torch.stack(jacobian_parent),
            torch.stack(jacobian_legacy),
            rtol=1e-9,
            atol=1e-10,
        )

    def test_parent_jfk_pose_autograd_matches_legacy_sphere_fk(self):
        robot = RobotPanda(tensor_args={"device": "cpu", "dtype": torch.float64})
        q_value = torch.tensor(
            [[0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]], dtype=torch.float64
        )
        weights = torch.linspace(
            0.1,
            1.0,
            len(robot.link_collision_spheres_names) * 3,
            dtype=torch.float64,
        ).reshape(1, -1, 3)

        def evaluate(use_parent_jfk):
            q = q_value.clone().requires_grad_(True)
            if use_parent_jfk:
                poses = robot.jfk_s_collision_spheres_parent_links(q)[1]
            else:
                poses = robot.fk_collision_spheres(q)
            positions = torch.stack(poses).transpose(0, 1)[..., :3, 3]
            gradient = torch.autograd.grad((positions * weights).sum(), q)[0]
            return positions, gradient

        expected = evaluate(False)
        actual = evaluate(True)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-9, atol=1e-10)

    def test_subset_parent_jfk_matches_full_parent_jfk(self):
        robot = RobotPanda(tensor_args={"device": "cpu", "dtype": torch.float64})
        q = torch.randn(2, 3, 7, dtype=torch.float64) * 0.1
        n_parents = len(robot.collision_sphere_unique_parent_links)
        parent_mask = torch.zeros(2, n_parents, dtype=torch.bool)
        parent_mask[0, [0, 2]] = True
        parent_mask[1, [1, min(3, n_parents - 1)]] = True
        phase = torch.arange(3).expand(2, -1)
        selection = TemporalSelection(
            active_indices=None,
            bucket_sizes=torch.full((2,), 3, dtype=torch.long),
            risk_mask=torch.ones(2, 3, dtype=torch.bool),
            environment_clearance=torch.zeros(2, 3, dtype=torch.float64),
            self_clearance=torch.zeros(2, 3, dtype=torch.float64),
            active_index_matrix=phase,
            bucket_options=(3,),
            parent_link_mask=parent_mask,
        )
        buckets = ActiveJacobianComputer(
            robot, use_parent_link_kinematics=True
        ).compute_selection_link_broad_phase(q, selection)
        full_jacobians, full_poses = robot.jfk_s_collision_spheres_parent_links(
            q.reshape(-1, 7)
        )
        full_jacobians = torch.stack(full_jacobians).transpose(0, 1).reshape(
            2, 3, -1, 6, 7
        )
        full_poses = torch.stack(full_poses).transpose(0, 1).reshape(2, 3, -1, 3, 4)
        self.assertEqual(sum(bucket.candidate_indices.numel() for bucket in buckets), 2)
        for bucket in buckets:
            candidates = bucket.candidate_indices
            spheres = bucket.sphere_indices
            torch.testing.assert_close(
                bucket.poses,
                full_poses[candidates[:, None], phase[candidates]][:, :, spheres],
                rtol=1e-9,
                atol=1e-10,
            )
            torch.testing.assert_close(
                bucket.jacobians,
                full_jacobians[candidates[:, None], phase[candidates]][:, :, spheres],
                rtol=1e-9,
                atol=1e-10,
            )

    def test_cached_pose_subset_jacobian_matches_jfk_without_fk_recompute(self):
        robot = RobotPanda(tensor_args={"device": "cpu", "dtype": torch.float64})
        q = torch.randn(2, 3, 7, dtype=torch.float64) * 0.1
        flat_q = q.reshape(-1, 7)
        related = robot.fk_collision_parent_pose_cache(flat_q)
        sphere_poses = torch.stack(
            robot.collision_sphere_poses_from_related_pose_cache(related)
        ).transpose(0, 1).reshape(2, 3, -1, 3, 4)
        scan_cache = FineSphereScanCache(
            related_link_ids=robot.collision_parent_related_link_ids,
            related_poses=tuple(pose.reshape(2, 3, 3, 4) for pose in related),
            sphere_poses=sphere_poses,
        )
        n_parents = len(robot.collision_sphere_unique_parent_links)
        parent_mask = torch.zeros(2, n_parents, dtype=torch.bool)
        parent_mask[0, [0, 2]] = True
        parent_mask[1, [1, min(3, n_parents - 1)]] = True
        phase = torch.arange(3).expand(2, -1)
        selection = TemporalSelection(
            active_indices=None,
            bucket_sizes=torch.full((2,), 3, dtype=torch.long),
            risk_mask=torch.ones(2, 3, dtype=torch.bool),
            environment_clearance=torch.zeros(2, 3, dtype=torch.float64),
            self_clearance=torch.zeros(2, 3, dtype=torch.float64),
            active_index_matrix=phase,
            bucket_options=(3,),
            parent_link_mask=parent_mask,
            fine_sphere_scan_cache=scan_cache,
        )
        computer = ActiveJacobianComputer(robot, use_parent_link_kinematics=True)
        cached_buckets = computer.compute_selection_link_broad_phase(
            q, selection, reuse_scan_cache=True
        )
        reference_buckets = computer.compute_selection_link_broad_phase(
            q, selection, reuse_scan_cache=False
        )
        self.assertEqual(len(cached_buckets), len(reference_buckets))
        for cached, reference in zip(cached_buckets, reference_buckets):
            torch.testing.assert_close(cached.poses, reference.poses, rtol=1e-9, atol=1e-10)
            torch.testing.assert_close(
                cached.jacobians, reference.jacobians, rtol=1e-9, atol=1e-10
            )


if __name__ == "__main__":
    unittest.main()
