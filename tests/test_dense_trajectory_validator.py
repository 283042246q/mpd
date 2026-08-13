import unittest

import torch

from mpd.inference.dense_trajectory_validator import (
    DenseTrajectoryValidator,
    DenseValidationResult,
)
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline


class FakeEnvironmentField:
    collision_margins = torch.tensor([0.1], dtype=torch.float64)

    def object_signed_distances(self, positions):
        # A vertical plane at x=0.5; signed free-space distance is |x-0.5|.
        return torch.abs(positions[..., 0] - 0.5).unsqueeze(-2)


class FakeRobot:
    task_space_dim = 2
    q_pos_min = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    q_pos_max = torch.tensor([1.0, 1.0], dtype=torch.float64)
    dq_max = torch.tensor([2.0, 2.0], dtype=torch.float64)
    ddq_max = torch.tensor([4.0, 4.0], dtype=torch.float64)

    def fk_collision_spheres(self, q):
        pose = torch.zeros(q.shape[0], 3, 4, dtype=q.dtype)
        pose[:, 0, 0] = 1
        pose[:, 1, 1] = 1
        pose[:, 2, 2] = 1
        pose[:, 0, 3] = q[:, 0]
        pose[:, 1, 3] = q[:, 1]
        return [pose]


class FakeTask:
    robot = FakeRobot()
    parametric_trajectory = object()

    def get_collision_objects_field(self):
        return FakeEnvironmentField()

    def get_collision_ws_boundaries_field(self):
        return None

    def get_collision_self_field(self):
        return None


class DenseTrajectoryValidatorTest(unittest.TestCase):
    def test_dense_bspline_uses_structural_degree_when_public_degree_is_overwritten(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=8,
            degree=3,
            num_T_pts=16,
            zero_vel_at_start_and_goal=False,
            zero_acc_at_start_and_goal=False,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        trajectory.set_boundary_conditions(
            q_pos_start=torch.zeros(2, dtype=torch.float64),
            q_pos_goal=torch.ones(2, dtype=torch.float64),
        )
        task = FakeTask()
        task.parametric_trajectory = trajectory
        validator = DenseTrajectoryValidator(task)
        control_points = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)[None, :, None].repeat(1, 1, 2)

        trajectory.bspline.d = torch.arange(4, dtype=torch.float64)
        q_position, q_velocity, q_acceleration = validator._dense_bspline(control_points, 33)

        self.assertEqual(q_position.shape, (1, 33, 2))
        self.assertEqual(q_velocity.shape, q_position.shape)
        self.assertEqual(q_acceleration.shape, q_position.shape)
        self.assertTrue(torch.isfinite(q_position).all())

    def test_detects_between_input_waypoint_collision(self):
        validator = DenseTrajectoryValidator(FakeTask())
        q_position = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float64)
        q_velocity = torch.zeros_like(q_position)
        q_acceleration = torch.zeros_like(q_position)
        result = validator.validate(
            q_position=q_position,
            q_velocity=q_velocity,
            q_acceleration=q_acceleration,
            num_points=33,
        )
        self.assertFalse(result.trajectory_valid_mask.item())
        self.assertTrue(result.environment_collision_mask.any().item())
        self.assertGreater(result.first_invalid_index.item(), 0)

    def test_joint_limit_violation_is_reported_separately(self):
        validator = DenseTrajectoryValidator(FakeTask())
        q_position = torch.tensor([[[1.2, 0.0], [1.2, 0.0]]], dtype=torch.float64)
        zeros = torch.zeros_like(q_position)
        result = validator.validate(
            q_position=q_position,
            q_velocity=zeros,
            q_acceleration=zeros,
            num_points=4,
            check_environment=False,
        )
        self.assertTrue(result.joint_position_violation_mask.item())
        self.assertTrue(result.joint_limit_mask.all().item())

    def test_ranked_batches_stop_after_first_batch_with_valid_candidate(self):
        class RankedValidator(DenseTrajectoryValidator):
            def __init__(self):
                self.seen = []

            def validate(self, control_points=None, num_points=4, **kwargs):
                candidate_ids = control_points[:, 0, 0].long()
                self.seen.append(candidate_ids.tolist())
                batch = candidate_ids.numel()
                valid = candidate_ids == 3
                zeros_waypoint = torch.zeros(batch, num_points, dtype=torch.bool)
                zeros_trajectory = torch.zeros(batch, dtype=torch.bool)
                q = torch.zeros(batch, num_points, 2, dtype=torch.float64)
                return DenseValidationResult(
                    trajectory_valid_mask=valid,
                    environment_collision_mask=zeros_waypoint.clone(),
                    self_collision_mask=zeros_waypoint.clone(),
                    joint_limit_mask=zeros_waypoint.clone(),
                    joint_position_violation_mask=zeros_trajectory.clone(),
                    joint_velocity_violation_mask=zeros_trajectory.clone(),
                    joint_acceleration_violation_mask=zeros_trajectory.clone(),
                    minimum_environment_clearance=torch.ones(batch, dtype=torch.float64),
                    minimum_self_clearance=torch.ones(batch, dtype=torch.float64),
                    first_invalid_index=torch.full((batch,), -1, dtype=torch.long),
                    q_position=q.clone(),
                    q_velocity=q.clone(),
                    q_acceleration=q.clone(),
                )

        validator = RankedValidator()
        control_points = torch.zeros(6, 2, 1, dtype=torch.float64)
        control_points[:, 0, 0] = torch.arange(6)
        result = validator.validate_ranked_batches(
            control_points,
            ranked_indices=torch.tensor([0, 1, 3, 2, 4, 5]),
            batch_buckets=[2],
            num_points=4,
        )
        self.assertEqual(validator.seen, [[0, 1], [3, 2]])
        self.assertEqual(result.validated_indices.tolist(), [0, 1, 3, 2])
        self.assertEqual(result.trajectory_checked_mask.tolist(), [True, True, True, True, False, False])
        self.assertEqual(result.trajectory_valid_mask.tolist(), [False, False, False, True, False, False])
        self.assertEqual(result.batches_evaluated, 2)
        self.assertEqual(result.bucket_capacities_evaluated, (2, 2))
        self.assertEqual(result.padding_slots_evaluated, 0)
        self.assertFalse(result.complete)
        self.assertTrue(torch.isnan(result.q_position[4:]).all())

    def test_fixed_buckets_pad_final_44_candidates_into_64_slots(self):
        class NoValidValidator(DenseTrajectoryValidator):
            def __init__(self):
                self._bucket_workspaces = {}
                self.seen_batch_sizes = []

            def validate(self, control_points=None, num_points=4, **kwargs):
                batch = control_points.shape[0]
                self.seen_batch_sizes.append(batch)
                zeros_waypoint = torch.zeros(batch, num_points, dtype=torch.bool)
                zeros_trajectory = torch.zeros(batch, dtype=torch.bool)
                q = torch.zeros(batch, num_points, 2, dtype=torch.float64)
                return DenseValidationResult(
                    trajectory_valid_mask=zeros_trajectory.clone(),
                    environment_collision_mask=zeros_waypoint.clone(),
                    self_collision_mask=zeros_waypoint.clone(),
                    joint_limit_mask=zeros_waypoint.clone(),
                    joint_position_violation_mask=zeros_trajectory.clone(),
                    joint_velocity_violation_mask=zeros_trajectory.clone(),
                    joint_acceleration_violation_mask=zeros_trajectory.clone(),
                    minimum_environment_clearance=torch.ones(batch, dtype=torch.float64),
                    minimum_self_clearance=torch.ones(batch, dtype=torch.float64),
                    first_invalid_index=torch.zeros(batch, dtype=torch.long),
                    q_position=q.clone(),
                    q_velocity=q.clone(),
                    q_acceleration=q.clone(),
                )

        validator = NoValidValidator()
        control_points = torch.zeros(100, 2, 1, dtype=torch.float64)
        control_points[:, 0, 0] = torch.arange(100)
        result = validator.validate_ranked_batches(
            control_points,
            ranked_indices=torch.arange(100),
            batch_buckets=[8, 16, 32, 64],
            num_points=4,
        )

        self.assertEqual(validator.seen_batch_sizes, [8, 16, 32, 64])
        self.assertEqual(
            result.bucket_capacities_evaluated, (8, 16, 32, 64)
        )
        self.assertEqual(result.padding_slots_evaluated, 20)
        self.assertEqual(int(result.trajectory_checked_mask.sum()), 100)
        self.assertTrue(result.complete)


if __name__ == "__main__":
    unittest.main()
