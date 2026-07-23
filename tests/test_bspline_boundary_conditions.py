import unittest

import torch

from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline


class BsplineBoundaryConditionsTest(unittest.TestCase):
    def test_accepts_nonzero_boundary_derivatives(self):
        for phase_time_class in ["PhaseTimeLinear", "PhaseTimeSigmoid"]:
            with self.subTest(phase_time_class=phase_time_class):
                trajectory = ParametricTrajectoryBspline(
                    n_control_points=12,
                    degree=5,
                    zero_vel_at_start_and_goal=True,
                    zero_acc_at_start_and_goal=True,
                    remove_outer_control_points=True,
                    num_T_pts=64,
                    trajectory_duration=4.0,
                    phase_time_class=phase_time_class,
                    tensor_args={"device": "cpu", "dtype": torch.float64},
                )
                q_pos_start = torch.tensor([0.2, -0.4], dtype=torch.float64)
                q_pos_goal = torch.tensor([1.1, 0.7], dtype=torch.float64)
                q_vel_start = torch.tensor([0.15, -0.08], dtype=torch.float64)
                q_vel_goal = torch.tensor([-0.03, 0.12], dtype=torch.float64)
                q_acc_start = torch.tensor([0.04, -0.02], dtype=torch.float64)
                q_acc_goal = torch.tensor([-0.01, 0.03], dtype=torch.float64)
                inner_control_points = torch.randn(3, 6, 2, dtype=torch.float64)

                result = trajectory.get_q_trajectory(
                    inner_control_points,
                    q_pos_start,
                    q_pos_goal,
                    q_vel_start=q_vel_start,
                    q_vel_goal=q_vel_goal,
                    q_acc_start=q_acc_start,
                    q_acc_goal=q_acc_goal,
                )

                for batch_idx in range(inner_control_points.shape[0]):
                    torch.testing.assert_close(result["pos"][batch_idx, 0], q_pos_start)
                    torch.testing.assert_close(result["pos"][batch_idx, -1], q_pos_goal)
                    torch.testing.assert_close(result["vel"][batch_idx, 0], q_vel_start, atol=1e-9, rtol=1e-9)
                    torch.testing.assert_close(result["vel"][batch_idx, -1], q_vel_goal, atol=1e-9, rtol=1e-9)
                    torch.testing.assert_close(result["acc"][batch_idx, 0], q_acc_start, atol=1e-8, rtol=1e-8)
                    torch.testing.assert_close(result["acc"][batch_idx, -1], q_acc_goal, atol=1e-8, rtol=1e-8)

    def test_defaults_boundary_derivatives_to_zero(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=12,
            degree=5,
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=True,
            remove_outer_control_points=True,
            num_T_pts=32,
            trajectory_duration=2.0,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        q_pos_start = torch.tensor([0.0, -0.2], dtype=torch.float64)
        q_pos_goal = torch.tensor([0.8, 0.5], dtype=torch.float64)
        inner_control_points = torch.randn(2, 6, 2, dtype=torch.float64)

        result = trajectory.get_q_trajectory(inner_control_points, q_pos_start, q_pos_goal)

        torch.testing.assert_close(result["vel"][:, 0], torch.zeros(2, 2, dtype=torch.float64), atol=1e-9, rtol=0)
        torch.testing.assert_close(result["vel"][:, -1], torch.zeros(2, 2, dtype=torch.float64), atol=1e-9, rtol=0)
        torch.testing.assert_close(result["acc"][:, 0], torch.zeros(2, 2, dtype=torch.float64), atol=1e-8, rtol=0)
        torch.testing.assert_close(result["acc"][:, -1], torch.zeros(2, 2, dtype=torch.float64), atol=1e-8, rtol=0)

    def test_ee_goal_context_preserves_each_generated_joint_endpoint(self):
        trajectory = ParametricTrajectoryBspline(
            n_control_points=12,
            degree=5,
            zero_vel_at_start_and_goal=True,
            zero_acc_at_start_and_goal=True,
            remove_outer_control_points=True,
            keep_last_control_point=True,
            num_T_pts=32,
            trajectory_duration=2.0,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )
        q_pos_start = torch.tensor([0.0, -0.2], dtype=torch.float64)
        q_pos_goal = torch.tensor([0.8, 0.5], dtype=torch.float64)
        inner_control_points = torch.randn(2, 7, 2, dtype=torch.float64)
        generated_joint_endpoints = inner_control_points[:, -1].clone()

        result = trajectory.get_q_trajectory(inner_control_points, q_pos_start, q_pos_goal)

        torch.testing.assert_close(result["pos"][:, -1], generated_joint_endpoints)
        torch.testing.assert_close(result["vel"][:, -1], torch.zeros(2, 2, dtype=torch.float64), atol=1e-9, rtol=0)
        torch.testing.assert_close(result["acc"][:, -1], torch.zeros(2, 2, dtype=torch.float64), atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
