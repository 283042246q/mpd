import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from dotmap import DotMap

from mpd.inference.active_jacobian import ActiveJacobianComputer
from mpd.inference.collision_risk_selector import TemporalSelection, fixed_sample_indices
from mpd.inference.cost_guides import (
    CostGuideManagerParametricTrajectory,
    CostJointSpaceAcceleration,
    CostJointSpacePathLength,
    CostJointSpaceVelocity,
    project_collision_gradient_to_joints,
)
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline


class FakeRobot:
    def jfk_s_collision_spheres(self, q):
        pose = torch.zeros(q.shape[0], 3, 4, dtype=q.dtype)
        pose[:, 0, 0] = 1.0
        pose[:, 1, 1] = 1.0
        pose[:, 2, 2] = 1.0
        pose[:, 0, 3] = q[:, 0]
        jacobian = torch.zeros(q.shape[0], 6, q.shape[-1], dtype=q.dtype)
        jacobian[:, 0, 0] = 1.0
        return [jacobian], [pose]

    def jfk_s_collision_spheres_parent_links(self, q):
        return self.jfk_s_collision_spheres(q)


class ActiveJacobianEquivalenceTest(unittest.TestCase):
    def test_bucket_gather_preserves_candidate_and_phase_order(self):
        q = torch.arange(3 * 8 * 2, dtype=torch.float64).reshape(3, 8, 2)
        active = [torch.tensor([0, 2, 7]), torch.tensor([], dtype=torch.long), torch.tensor([1, 4, 6])]
        buckets = ActiveJacobianComputer(FakeRobot()).compute(q, active)
        self.assertEqual(len(buckets), 1)
        bucket = buckets[0]
        self.assertEqual(bucket.candidate_indices.tolist(), [0, 2])
        torch.testing.assert_close(bucket.q_active[0], q[0, active[0]])
        torch.testing.assert_close(bucket.q_active[1], q[2, active[2]])
        torch.testing.assert_close(bucket.poses[..., 0, 0, 3], bucket.q_active[..., 0])

    def test_dense_parent_batch_preserves_regular_layout(self):
        q = torch.arange(3 * 8 * 2, dtype=torch.float64).reshape(3, 8, 2)
        computer = ActiveJacobianComputer(FakeRobot(), use_parent_link_kinematics=True)
        dense = computer.compute_dense(q)
        self.assertTrue(dense.covers_full_batch)
        self.assertEqual(dense.candidate_indices.tolist(), [0, 1, 2])
        torch.testing.assert_close(dense.q_dense, q)
        torch.testing.assert_close(dense.poses[..., 0, 0, 3], q[..., 0])

        subset = computer.compute_dense(q, torch.tensor([2, 0]))
        self.assertFalse(subset.covers_full_batch)
        self.assertEqual(subset.candidate_indices.tolist(), [2, 0])
        torch.testing.assert_close(subset.q_dense, q[[2, 0]])

    def test_active_link_projection_matches_dense_projection(self):
        poses = torch.zeros(2, 5, 4, 3, 4, dtype=torch.float64)
        poses[..., 0, 0] = 1.0
        poses[..., 1, 1] = 1.0
        poses[..., 2, 2] = 1.0
        jacobians = torch.randn(2, 5, 4, 6, 3, dtype=torch.float64)
        gradients = torch.randn(2, 5, 4, 3, dtype=torch.float64)
        gradients[..., 0, :] = 0.0
        gradients[..., 2, :] = 0.0
        gradients[0, 1, 1] = 0.0
        dense = project_collision_gradient_to_joints(
            poses, jacobians, gradients, 3, active_link_pruning=False
        )
        sparse = project_collision_gradient_to_joints(
            poses, jacobians, gradients, 3, active_link_pruning=True
        )
        torch.testing.assert_close(sparse, dense, rtol=1e-12, atol=1e-12)


class FakeCollisionField:
    collision_margins = torch.tensor([0.0], dtype=torch.float64)
    cutoff_margin = 0.0

    def compute_distance_field_cost_and_gradient(self, positions):
        penetration = torch.relu(0.25 - positions[..., 0])
        gradient = torch.zeros_like(positions)
        gradient[..., 0] = torch.where(penetration > 0, -1.0, 0.0)
        return penetration, gradient


class FakeGuideRobot(FakeRobot):
    task_space_dim = 2
    link_collision_spheres_names = ["sphere"]
    link_self_collision_tuples = []
    q_pos_min = torch.tensor([-2.0, -2.0], dtype=torch.float64)
    q_pos_max = torch.tensor([2.0, 2.0], dtype=torch.float64)
    dq_max = None
    ddq_max = None

    def jfk_s_ee(self, q):
        return self.jfk_s_collision_spheres(q)


class FakeGuideTask:
    def __init__(self, trajectory):
        self.robot = FakeGuideRobot()
        self.env = object()
        self.parametric_trajectory = trajectory
        self.ee_pose_goal = None
        self.field = FakeCollisionField()

    def get_collision_objects_field(self):
        return self.field

    def get_collision_extra_objects_field(self):
        return self.field

    def get_collision_self_field(self):
        return None


class IdentityDataset:
    context_ee_goal_pose = False

    @staticmethod
    def unnormalize_control_points(value):
        return value

    @staticmethod
    def grad_unnormalized_wrt_control_points_normalized(value):
        return torch.ones_like(value)


class EEGoalIdentityDataset(IdentityDataset):
    context_ee_goal_pose = True


class AnalyticJointCostGradientTest(unittest.TestCase):
    def setUp(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=16,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        self.task = FakeGuideTask(trajectory)

    def test_velocity_and_acceleration_match_autograd(self):
        q_pos = torch.randn(2, 16, 2, dtype=torch.float64)
        q_vel = torch.randn(2, 16, 2, dtype=torch.float64, requires_grad=True)
        q_acc = torch.randn(2, 16, 2, dtype=torch.float64, requires_grad=True)

        velocity = CostJointSpaceVelocity(self.task)
        cost_vel, grads_vel = velocity.compute_cost_grad_wrt_q(None, q_pos, q_vel, q_acc)
        reference_vel = torch.autograd.grad(cost_vel.sum(), q_vel, retain_graph=True)[0]
        torch.testing.assert_close(grads_vel["vel"], reference_vel)

        acceleration = CostJointSpaceAcceleration(self.task)
        cost_acc, grads_acc = acceleration.compute_cost_grad_wrt_q(None, q_pos, q_vel, q_acc)
        reference_acc = torch.autograd.grad(cost_acc.sum(), q_acc)[0]
        torch.testing.assert_close(grads_acc["acc"], reference_acc)

    def test_path_length_matches_autograd_including_zero_segments(self):
        q = torch.randn(2, 16, 2, dtype=torch.float64)
        q[:, 5] = q[:, 4]
        q_reference = q.clone().requires_grad_(True)
        diff = torch.zeros_like(q_reference)
        diff[..., 1:, :] = torch.diff(q_reference, dim=-2)
        reference_cost = 0.5 * torch.linalg.norm(diff, dim=-1)
        reference_gradient = torch.autograd.grad(reference_cost.sum(), q_reference)[0]

        path = CostJointSpacePathLength(self.task)
        cost, gradients = path.compute_cost_grad_wrt_q(None, q, q, q)
        torch.testing.assert_close(cost, reference_cost.detach())
        torch.testing.assert_close(gradients["pos"], reference_gradient)


class ForceAllActiveEquivalenceTest(unittest.TestCase):
    def test_fused_bspline_mapping_matches_materialized_fallback(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        common = {
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                },
                "CostJointSpaceVelocity": {"weight": 0.02},
                "CostJointSpaceAcceleration": {"weight": 0.005},
                "CostJointSpacePathLength": {"weight": 0.1},
            },
            "project_gradient_hierarchy": False,
        }

        def make_manager(fused):
            args = DotMap(common)
            args.gradient_pruning = {
                "enabled": True,
                "force_all_active": True,
                "temporal": {"enabled": False},
                "mapping": {"fused_bspline_integration": fused},
            }
            return CostGuideManagerParametricTrajectory(task, IdentityDataset(), args)

        materialized = make_manager(False)
        fused = make_manager(True)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        control_points.requires_grad_(True)
        cost_materialized, grad_materialized = materialized(
            control_points.clone(), return_cost=True
        )
        cost_fused, grad_fused = fused(control_points.clone(), return_cost=True)
        torch.testing.assert_close(cost_fused, cost_materialized, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_fused, grad_materialized, rtol=1e-7, atol=1e-9)
        self.assertFalse(cost_fused.requires_grad)
        self.assertFalse(grad_fused.requires_grad)

    def test_sparse_bspline_support_matches_materialized_fallback(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=10,
            degree=5,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        common = {
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                },
                "CostJointSpaceVelocity": {"weight": 0.02},
                "CostJointSpaceAcceleration": {"weight": 0.005},
                "CostJointSpacePathLength": {"weight": 0.1},
            },
            "project_gradient_hierarchy": False,
        }

        def make_manager(sparse_support):
            args = DotMap(common)
            args.gradient_pruning = {
                "enabled": True,
                "force_all_active": True,
                "temporal": {"enabled": False},
                "mapping": {
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": sparse_support,
                },
            }
            return CostGuideManagerParametricTrajectory(task, IdentityDataset(), args)

        materialized = make_manager(False)
        sparse = make_manager(True)
        control_points = torch.randn(3, 10, 2, dtype=torch.float64)
        cost_materialized, grad_materialized = materialized(
            control_points.clone(), return_cost=True
        )
        cost_sparse, grad_sparse = sparse(control_points.clone(), return_cost=True)
        torch.testing.assert_close(cost_sparse, cost_materialized, rtol=1e-10, atol=1e-11)
        torch.testing.assert_close(grad_sparse, grad_materialized, rtol=1e-10, atol=1e-11)

    def test_zero_weight_cost_is_not_evaluated(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        args = DotMap(
            {
                "costs": {"CostJointSpacePathLength": {"weight": 0.0}},
                "project_gradient_hierarchy": False,
                "gradient_pruning": {
                    "enabled": True,
                    "force_all_active": True,
                    "temporal": {"enabled": False},
                },
            }
        )
        manager = CostGuideManagerParametricTrajectory(task, IdentityDataset(), args)
        cost = manager.costs["CostJointSpacePathLength"].cost
        cost.compute_cost_grad_wrt_q = mock.Mock(
            side_effect=AssertionError("zero-weight cost was evaluated")
        )
        control_points = torch.zeros(2, 8, 2, dtype=torch.float64)
        value, gradient = manager(control_points, return_cost=True)
        torch.testing.assert_close(value, torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(gradient, torch.zeros_like(control_points))

    def test_force_all_active_matches_legacy_collision_cost_and_gradient(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        common = {
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                }
            },
            "project_gradient_hierarchy": False,
        }
        legacy = CostGuideManagerParametricTrajectory(task, IdentityDataset(), DotMap(common))
        pruned_args = DotMap(common)
        pruned_args.gradient_pruning = {
            "enabled": True,
            "force_all_active": True,
            "endpoint": {"ee_only_last_point": True},
        }
        pruned = CostGuideManagerParametricTrajectory(task, IdentityDataset(), pruned_args)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        cost_legacy, grad_legacy = legacy(control_points.clone(), return_cost=True)
        cost_pruned, grad_pruned = pruned(control_points.clone(), return_cost=True)
        torch.testing.assert_close(cost_pruned, cost_legacy, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_pruned, grad_legacy, rtol=1e-7, atol=1e-9)

    def test_dense_parent_fast_path_matches_generic_full_bucket(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        common = {
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                }
            },
            "project_gradient_hierarchy": False,
        }
        generic_args = DotMap(common)
        generic_args.gradient_pruning = {
            "enabled": True,
            "record_active_statistics": True,
            "temporal": {"enabled": False},
            "spatial": {
                "parent_link_kinematics": True,
                "dense_parent_fast_path": False,
            },
        }
        fast_args = DotMap(common)
        fast_args.gradient_pruning = {
            "enabled": True,
            "record_active_statistics": True,
            "temporal": {"enabled": False},
            "spatial": {
                "parent_link_kinematics": True,
                "dense_parent_fast_path": True,
            },
        }
        generic = CostGuideManagerParametricTrajectory(task, IdentityDataset(), generic_args)
        fast = CostGuideManagerParametricTrajectory(task, IdentityDataset(), fast_args)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        cost_generic, grad_generic = generic(control_points.clone(), return_cost=True)
        cost_fast, grad_fast = fast(control_points.clone(), return_cost=True)
        trajectory = fast.parametric_trajectory.get_q_trajectory(
            control_points,
            None,
            None,
            get_type=("pos", "vel", "acc"),
            get_time_representation=False,
        )
        dense = fast.active_jacobian_computer.compute_dense(trajectory["pos"])
        phase5_state = SimpleNamespace(
            q=trajectory["pos"],
            q_s=trajectory["vel"],
            q_ss=trajectory["acc"],
            collision_sphere_poses=dense.poses,
            collision_sphere_jacobians=dense.jacobians,
        )
        with mock.patch.object(
            fast.active_jacobian_computer,
            "compute_dense",
            wraps=fast.active_jacobian_computer.compute_dense,
        ) as compute_dense:
            cost_cached, grad_cached = fast(
                control_points.clone(),
                return_cost=True,
                _phase5_trajectory_state=phase5_state,
            )
        torch.testing.assert_close(cost_fast, cost_generic, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_fast, grad_generic, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(cost_cached, cost_fast, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_cached, grad_fast, rtol=1e-7, atol=1e-9)
        compute_dense.assert_not_called()
        self.assertFalse(generic.guidance_profiler.records[-1]["dense_parent_fast_path"])
        self.assertTrue(fast.guidance_profiler.records[-1]["dense_parent_fast_path"])
        self.assertTrue(
            fast.guidance_profiler.records[-1]["phase5_kinematics_cache_reused"]
        )

    def test_temporal_mixed_full_and_sparse_buckets_match_generic_parent_path(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=64,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        common = {
            "costs": {
                "CostTaskSpaceCollisionObjects": {
                    "weight": 1.0,
                    "use_only_on_extra_objects": False,
                }
            },
            "project_gradient_hierarchy": False,
        }

        def make_manager(dense_fast, fused=True, sparse_support=False):
            args = DotMap(common)
            args.gradient_pruning = {
                "enabled": True,
                "record_active_statistics": True,
                "temporal": {"enabled": True},
                "spatial": {
                    "parent_link_kinematics": True,
                    "dense_parent_fast_path": dense_fast,
                },
                "mapping": {
                    "fused_bspline_integration": fused,
                    "sparse_bspline_support": sparse_support,
                },
            }
            manager = CostGuideManagerParametricTrajectory(task, IdentityDataset(), args)
            full = torch.arange(64)
            sparse = fixed_sample_indices(64, 32)
            selection = TemporalSelection(
                active_indices=[full, sparse],
                bucket_sizes=torch.tensor([64, 32]),
                risk_mask=torch.ones((2, 64), dtype=torch.bool),
                environment_clearance=torch.full((2, 64), torch.nan, dtype=torch.float64),
                self_clearance=torch.full((2, 64), torch.nan, dtype=torch.float64),
            )
            manager.collision_risk_selector.select = lambda *args, **kwargs: selection
            return manager

        generic = make_manager(False)
        fast = make_manager(True)
        materialized = make_manager(False, fused=False)
        sparse_support = make_manager(False, fused=False, sparse_support=True)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        cost_generic, grad_generic = generic(control_points.clone(), return_cost=True)
        cost_fast, grad_fast = fast(control_points.clone(), return_cost=True)
        cost_materialized, grad_materialized = materialized(
            control_points.clone(), return_cost=True
        )
        cost_sparse_support, grad_sparse_support = sparse_support(
            control_points.clone(), return_cost=True
        )
        torch.testing.assert_close(cost_fast, cost_generic, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_fast, grad_generic, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(cost_generic, cost_materialized, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_generic, grad_materialized, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(cost_sparse_support, cost_materialized, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_sparse_support, grad_materialized, rtol=1e-7, atol=1e-9)
        record = fast.guidance_profiler.records[-1]
        self.assertTrue(record["dense_parent_fast_path"])
        self.assertEqual(record["bucket_128"], 1)
        self.assertEqual(record["bucket_32"], 1)

    def test_temporal_selection_is_reused_only_within_same_ddim_step(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=64,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        args = DotMap(
            {
                "costs": {
                    "CostTaskSpaceCollisionObjects": {
                        "weight": 1.0,
                        "use_only_on_extra_objects": False,
                    }
                },
                "project_gradient_hierarchy": False,
                "gradient_pruning": {
                    "enabled": True,
                    "record_active_statistics": True,
                    "temporal": {
                        "enabled": True,
                        "reuse_selection_within_ddim_step": True,
                    },
                    "spatial": {
                        "parent_link_kinematics": True,
                        "dense_parent_fast_path": True,
                    },
                },
            }
        )
        manager = CostGuideManagerParametricTrajectory(task, IdentityDataset(), args)
        sparse = fixed_sample_indices(64, 32)
        selection = TemporalSelection(
            active_indices=[sparse, sparse],
            bucket_sizes=torch.tensor([32, 32]),
            risk_mask=torch.ones((2, 64), dtype=torch.bool),
            environment_clearance=torch.full((2, 64), torch.nan, dtype=torch.float64),
            self_clearance=torch.full((2, 64), torch.nan, dtype=torch.float64),
        )
        select_calls = []

        def select(*args, **kwargs):
            select_calls.append(1)
            return selection

        manager.collision_risk_selector.select = select
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        manager(
            control_points.clone(),
            diffusion_timestep=torch.tensor([7]),
            guide_iteration=0,
        )
        manager(
            control_points.clone() + 0.01,
            diffusion_timestep=torch.tensor([7]),
            guide_iteration=1,
        )
        manager(
            control_points.clone(),
            diffusion_timestep=torch.tensor([6]),
            guide_iteration=0,
        )

        self.assertEqual(len(select_calls), 2)
        records = manager.guidance_profiler.records[-3:]
        self.assertFalse(records[0]["temporal_selection_cache_hit"])
        self.assertTrue(records[1]["temporal_selection_cache_hit"])
        self.assertFalse(records[2]["temporal_selection_cache_hit"])

    def test_endpoint_only_ee_cost_matches_full_legacy_kinematics(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            remove_outer_control_points=False,
            num_T_pts=32,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            q_pos_goal=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        task = FakeGuideTask(trajectory)
        task.ee_pose_goal = torch.tensor(
            [[1.0, 0.0, 0.0, 0.8], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=torch.float64,
        )
        common = {
            "costs": {"CostTaskSpaceEEGoalPosition": {"weight": 1.0}},
            "project_gradient_hierarchy": False,
        }
        legacy = CostGuideManagerParametricTrajectory(task, EEGoalIdentityDataset(), DotMap(common))
        pruned_args = DotMap(common)
        pruned_args.gradient_pruning = {"enabled": True, "force_all_active": True}
        pruned = CostGuideManagerParametricTrajectory(task, EEGoalIdentityDataset(), pruned_args)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).view(1, 8, 1).repeat(2, 1, 2)
        cost_legacy, grad_legacy = legacy(control_points.clone(), return_cost=True)
        cost_pruned, grad_pruned = pruned(control_points.clone(), return_cost=True)
        torch.testing.assert_close(cost_pruned, cost_legacy, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(grad_pruned, grad_legacy, rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
