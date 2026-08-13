import unittest
from types import SimpleNamespace

import torch

from mpd.inference.collision_risk_selector import (
    CollisionRiskSelector,
    fixed_sample_indices,
    nonuniform_trapezoid_weights,
)
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline


class CollisionRiskSelectorTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "coarse_points": 8,
            "buckets": [8, 16, 32],
            "environment_refine_margin": 0.1,
            "self_refine_margin": 0.05,
            "q_delta_threshold": None,
            "neighbor_dilation": 1,
            "always_keep_endpoints": True,
        }
        self.selector = CollisionRiskSelector(
            robot=None,
            config=self.config,
            candidate_pruning_enabled=True,
        )

    def test_fixed_indices_keep_endpoints(self):
        indices = fixed_sample_indices(32, 8)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), 31)
        self.assertEqual(indices.unique().numel(), indices.numel())

    def test_nonuniform_weights_match_torch_trapezoid(self):
        phase = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64)
        values = torch.tensor([1.0, 3.0, -2.0, 4.0], dtype=torch.float64)
        weights = nonuniform_trapezoid_weights(phase)
        torch.testing.assert_close((weights * values).sum(), torch.trapezoid(values, phase))

    def test_clear_trajectory_skips_collision_jacobian(self):
        q = torch.zeros(2, 32, 2)
        env = torch.full((2, 32), 1.0)
        self_clearance = torch.full((2, 32), 1.0)
        selection = self.selector.select_from_clearances(q, env, self_clearance)
        self.assertEqual(selection.active_counts.tolist(), [0, 0])

    def test_short_between_coarse_risk_is_retained(self):
        q = torch.zeros(1, 32, 2)
        env = torch.full((1, 32), 1.0)
        env[0, 11] = -0.001
        self_clearance = torch.full((1, 32), 1.0)
        selection = self.selector.select_from_clearances(q, env, self_clearance)
        self.assertIn(11, selection.active_indices[0].tolist())
        self.assertGreaterEqual(selection.active_counts[0].item(), 8)


    def test_select_scans_coarse_points_and_midpoints_only(self):
        class RecordingRobot:
            task_space_dim = 2

            def __init__(self):
                self.calls = []

            def fk_collision_spheres(self, q):
                self.calls.append(int(q.shape[0]))
                pose = torch.zeros(q.shape[0], 3, 4, dtype=q.dtype)
                pose[:, 0, 0] = 1.0
                pose[:, 1, 1] = 1.0
                pose[:, 2, 2] = 1.0
                return [pose]

        robot = RecordingRobot()
        selector = CollisionRiskSelector(
            robot=robot,
            config=self.config,
            candidate_pruning_enabled=True,
        )
        q = torch.zeros(2, 32, 2)
        selection = selector.select(q)
        self.assertEqual(robot.calls, [2 * 15])
        self.assertEqual(selection.active_counts.tolist(), [0, 0])

    def test_independent_parent_bound_scan_supports_temporal_without_link_broad_phase(self):
        class RecordingRobot:
            task_space_dim = 2
            collision_sphere_unique_parent_links = ["p0", "p1"]
            collision_parent_bound_parent_indices = torch.tensor([0, 1])
            collision_parent_bound_radii = torch.tensor([0.1, 0.1])
            collision_parent_self_pairs = torch.empty((0, 2), dtype=torch.long)
            collision_parent_bound_self_pairs = torch.empty((0, 2), dtype=torch.long)
            collision_parent_bound_self_pair_groups = torch.empty(0, dtype=torch.long)

            def __init__(self):
                self.calls = []

            def fk_collision_parent_bounds(self, q):
                self.calls.append(int(q.shape[0]))
                return torch.zeros(q.shape[0], 2, 3, dtype=q.dtype)

        robot = RecordingRobot()
        selector = CollisionRiskSelector(
            robot=robot,
            config=dict(self.config, enabled=True),
            use_parent_link_kinematics=True,
            link_broad_phase_config={"enabled": False},
            use_parent_bounds_scan=True,
        )
        q = torch.zeros(2, 32, 2)
        selector.select(q)
        self.assertEqual(robot.calls, [2 * 15])

    def test_full_scan_control_evaluates_every_dense_point(self):
        class RecordingRobot:
            task_space_dim = 2

            def __init__(self):
                self.calls = []

            def fk_collision_spheres(self, q):
                self.calls.append(int(q.shape[0]))
                pose = torch.zeros(q.shape[0], 3, 4, dtype=q.dtype)
                pose[:, 0, 0] = 1.0
                pose[:, 1, 1] = 1.0
                pose[:, 2, 2] = 1.0
                return [pose]

        robot = RecordingRobot()
        selector = CollisionRiskSelector(
            robot=robot,
            config=dict(self.config, coarse_scan=False),
            candidate_pruning_enabled=True,
        )
        q = torch.zeros(2, 32, 2)
        selector.select(q)
        self.assertEqual(robot.calls, [2 * 32])

    def test_real_endpoint_risk_activates_with_zero_dilation(self):
        config = dict(self.config, neighbor_dilation=0)
        selector = CollisionRiskSelector(
            robot=None,
            config=config,
            candidate_pruning_enabled=True,
        )
        q = torch.zeros(1, 32, 2)
        env = torch.ones(1, 32)
        env[0, -1] = -0.01
        selection = selector.select_from_clearances(q, env, torch.ones_like(env))
        self.assertGreater(selection.active_counts[0].item(), 0)
        self.assertIn(31, selection.active_indices[0].tolist())

    def test_vectorized_bucket_assignment_retains_all_risk_points(self):
        q = torch.zeros(4, 32, 2)
        env = torch.ones(4, 32)
        # K0, K8, K16, and K32 respectively after dilation/endpoints.
        env[1, 10] = -0.01
        env[2, torch.tensor([5, 10, 15, 20])] = -0.01
        env[3] = -0.01
        selection = self.selector.select_from_clearances(q, env, torch.ones_like(env))

        self.assertEqual(selection.bucket_sizes.tolist(), [0, 8, 16, 32])
        self.assertEqual(selection.active_index_matrix.shape, (4, 32))
        for candidate_idx in (1, 2, 3):
            required = torch.nonzero(selection.risk_mask[candidate_idx]).flatten()
            selected = selection.active_indices[candidate_idx]
            self.assertTrue(torch.isin(required, selected).all().item())
            torch.testing.assert_close(
                selected,
                selection.active_index_matrix[candidate_idx, : selected.numel()],
            )

    def test_candidate_and_temporal_sparsity_are_independent(self):
        q = torch.zeros(2, 32, 2)
        env = torch.ones(2, 32)
        env[1, 10] = -0.01
        self_clearance = torch.ones_like(env)

        time_only = CollisionRiskSelector(
            robot=None,
            config=dict(self.config, enabled=True),
            candidate_pruning_enabled=False,
        ).select_from_clearances(q, env, self_clearance)
        self.assertEqual(time_only.bucket_sizes.tolist(), [32, 8])

        candidate_only = CollisionRiskSelector(
            robot=None,
            config=dict(self.config, enabled=False),
            candidate_pruning_enabled=True,
        ).select_from_clearances(q, env, self_clearance)
        self.assertEqual(candidate_only.bucket_sizes.tolist(), [0, 32])

    def test_conditional_temporal_applies_only_below_threshold(self):
        q = torch.zeros(2, 32, 2)
        low_env = torch.ones(2, 32)
        low_env[:, 10] = -0.01
        low = CollisionRiskSelector(
            robot=None,
            config=dict(
                self.config,
                enabled=False,
                conditional_enabled=True,
                conditional_active_ratio_threshold=0.35,
            ),
            candidate_pruning_enabled=False,
        ).select_from_clearances(q, low_env, torch.ones_like(low_env))
        self.assertTrue(low.temporal_sparse_applied)
        self.assertEqual(low.bucket_sizes.tolist(), [8, 8])
        self.assertAlmostEqual(low.predicted_active_ratio, 0.25)

        high_env = torch.full((2, 32), -0.01)
        high = CollisionRiskSelector(
            robot=None,
            config=dict(
                self.config,
                enabled=False,
                conditional_enabled=True,
                conditional_active_ratio_threshold=0.35,
            ),
            candidate_pruning_enabled=False,
        ).select_from_clearances(q, high_env, torch.ones_like(high_env))
        self.assertFalse(high.temporal_sparse_applied)
        self.assertEqual(high.bucket_sizes.tolist(), [32, 32])
        self.assertAlmostEqual(high.predicted_active_ratio, 1.0)

    def test_link_broad_phase_expands_active_spheres_to_parent_links(self):
        robot = SimpleNamespace(
            collision_sphere_parent_indices=torch.tensor([0, 0, 1, 2]),
            collision_sphere_unique_parent_links=["p0", "p1", "p2"],
            df_collision_self=SimpleNamespace(
                link_idx_1=torch.tensor([0, 2]).numpy(),
                link_idx_2=torch.tensor([2, 3]).numpy(),
            ),
        )
        selector = CollisionRiskSelector(
            robot=robot,
            config=self.config,
            link_broad_phase_config={
                "enabled": True,
                "environment_margin": 0.2,
                "self_margin": 0.1,
            },
        )
        environment = torch.full((2, 4, 4), 1.0)
        environment[0, :, 1] = 0.1
        self_pairs = torch.full((2, 4, 2), 1.0)
        self_pairs[1, :, 1] = 0.05
        parents, spheres, pairs = selector._link_broad_phase_masks(
            environment, self_pairs
        )
        self.assertEqual(parents.tolist(), [[True, False, False], [False, True, True]])
        self.assertEqual(spheres.tolist(), [[True, True, False, False], [False, False, True, True]])
        self.assertEqual(pairs.tolist(), [[False, False], [False, True]])

    def test_span_certificate_skips_safe_spans_and_refines_uncertain_spans(self):
        class Origin:
            xyz = [0.0, 0.0, 0.0]

        class Joint:
            name = "joint0"
            parent = "base"
            child = "link0"
            joint_type = "revolute"
            origin = Origin()

        class FineSphereRobot:
            task_space_dim = 3
            collision_sphere_parent_links = ["link0"]
            collision_sphere_local_positions = torch.tensor([[1.0, 0.0, 0.0]])
            link_self_collision_tuples = []
            robot_urdf = SimpleNamespace(joints=[Joint()])

            @staticmethod
            def fk_collision_spheres_parent_links(q):
                pose = torch.zeros(q.shape[0], 3, 4, dtype=q.dtype, device=q.device)
                pose[:, :3, :3] = torch.eye(3, dtype=q.dtype, device=q.device)
                pose[:, 0, 3] = q[:, 0]
                return [pose]

        class ConstantSdf:
            def __init__(self, distance):
                self.distance = distance

            def compute_signed_distance_raw(self, positions):
                return torch.full(
                    positions.shape[:-1],
                    self.distance,
                    dtype=positions.dtype,
                    device=positions.device,
                )

        class Field:
            collision_margins = torch.tensor([0.1])

            def __init__(self, distance):
                self.sdf = ConstantSdf(distance)

            def df_obj_list_fn(self):
                return [self.sdf]

        trajectory = ParametricTrajectoryBspline(
            n_control_points=6,
            degree=3,
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=False,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        q_start = torch.zeros(1, dtype=torch.float64)
        trajectory.set_boundary_conditions(q_start, q_start)
        control_points = torch.zeros(2, 6, 1, dtype=torch.float64)
        q_dense = trajectory.get_q_trajectory(
            control_points, None, None, get_type=("pos",), get_time_representation=False
        )["pos"]
        selector = CollisionRiskSelector(
            robot=FineSphereRobot(),
            config=dict(self.config, enabled=False, buckets=[8, 16, 32]),
            use_parent_link_kinematics=True,
            span_certificate_config={
                "enabled": True,
                "max_subdivision_depth": 2,
                "environment_safe_margin": 0.08,
                "self_safe_margin": 0.06,
                "jacobian_bound_mode": "componentwise",
                "exact_sdf_for_certificate": True,
                "grid_error_scale": 1.0,
                "profile_stages": False,
            },
        )
        safe = selector.select_span_certificate(
            q_dense, control_points, trajectory, environment_field=Field(2.0)
        )
        self.assertEqual(safe.bucket_sizes.tolist(), [0, 0])
        self.assertEqual(
            safe.span_certificate_statistics["span_first_level_certification_ratio"],
            1.0,
        )

        uncertain = selector.select_span_certificate(
            q_dense, control_points, trajectory, environment_field=Field(0.0)
        )
        self.assertEqual(uncertain.bucket_sizes.tolist(), [32, 32])
        self.assertEqual(
            uncertain.span_certificate_statistics["span_depth_2_remaining_measure_ratio"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
