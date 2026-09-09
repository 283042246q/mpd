import inspect
import json
from pathlib import Path
import subprocess
import sys

from dotmap import DotMap
import numpy as np
import torch

from mpd.bimanual.cost_guide import BimanualCostGuideManagerParametricTrajectory
from mpd.bimanual.costs import dual_ee_goal_cost_gradient
from mpd.bimanual.runtime_contract import BimanualRequest, JOINT_NAMES, SCHEMA
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from scripts.inference import inference_marvin_bimanual as entrypoint


ROOT = Path(__file__).resolve().parents[1]


def _identity_pose(batch_shape=()):
    pose = torch.zeros(*batch_shape, 3, 4, dtype=torch.float64)
    pose[..., 0, 0] = 1.0
    pose[..., 1, 1] = 1.0
    pose[..., 2, 2] = 1.0
    return pose


class _Robot:
    task_space_dim = 3
    link_collision_spheres_names = ["Link7_L_collision", "Link7_R_collision"]
    link_self_collision_tuples = [(0, 1, 0.1, 0.1)]
    q_pos_min = torch.full((14,), -2.0, dtype=torch.float64)
    q_pos_max = torch.full((14,), 2.0, dtype=torch.float64)
    dq_max = None
    ddq_max = None

    def jfk_s_collision_spheres(self, q):
        poses = []
        jacobians = []
        for arm, index in enumerate((0, 7)):
            pose = _identity_pose((q.shape[0],))
            pose[:, 0, 3] = q[:, index] + 0.15 * arm
            jacobian = torch.zeros(q.shape[0], 6, 14, dtype=q.dtype)
            jacobian[:, 0, index] = 1.0
            poses.append(pose)
            jacobians.append(jacobian)
        return jacobians, poses

    def jfk_s_ee(self, q):
        poses = []
        jacobians = []
        for index in (0, 7):
            pose = _identity_pose((q.shape[0],))
            pose[:, 0, 3] = q[:, index]
            jacobian = torch.zeros(q.shape[0], 6, 14, dtype=q.dtype)
            jacobian[:, 0, index] = 1.0
            poses.append(pose)
            jacobians.append(jacobian)
        return jacobians, poses


class _Task:
    def __init__(self, trajectory):
        from torch_robotics.torch_planning_objectives.fields.distance_fields import (
            CollisionSelfField,
        )

        self.robot = _Robot()
        self.env = object()
        self.parametric_trajectory = trajectory
        self.active_ee_mask = torch.ones(2, dtype=torch.float64)
        self.ee_pose_goal = _identity_pose((2,))
        self.ee_pose_goal[:, 0, 3] = torch.tensor([0.7, -0.4])
        self.self_field = CollisionSelfField(
            robot=self.robot,
            link_self_collision_tuples=self.robot.link_self_collision_tuples,
            tensor_args={"device": "cpu", "dtype": torch.float64},
        )

    def jfk_s_ee(self, q):
        return self.robot.jfk_s_ee(q)

    def get_collision_objects_field(self):
        return None

    def get_collision_ws_boundaries_field(self):
        return None

    def get_collision_extra_objects_field(self):
        return None

    def get_collision_self_field(self):
        return self.self_field


class _Dataset:
    context_ee_goal_pose = True
    context_ee_goal_pose_bimanual = True

    @staticmethod
    def unnormalize_control_points(value):
        return value

    @staticmethod
    def grad_unnormalized_wrt_control_points_normalized(value):
        return torch.ones_like(value)


def _guide(task, enabled):
    args = DotMap(
        {
            "costs": {
                "CostTaskSpaceCollisionSelf": {"weight": 0.7},
                "CostTaskSpaceEEGoalPosition": {
                    "weight": 1.0,
                    "error_scale": 0.5,
                },
                "CostTaskSpaceEEGoalOrientation": {
                    "weight": 0.4,
                    "error_scale": 1.0,
                },
            },
            "project_gradient_hierarchy": False,
            "gradient_pruning": {
                "enabled": enabled,
                "force_all_active": True,
                "endpoint": {"ee_only_last_point": True},
                "temporal": {"enabled": False},
                "spatial": {
                    "parent_link_kinematics": False,
                    "dense_parent_fast_path": False,
                    "active_link_pruning": False,
                },
                "mapping": {
                    "fused_bspline_integration": False,
                    "sparse_bspline_support": False,
                },
            },
        }
    )
    return BimanualCostGuideManagerParametricTrajectory(task, _Dataset(), args)


def test_dual_ee_gradient_has_fixed_arm_columns_and_mask_average():
    poses = _identity_pose((1, 1, 2))
    jacobians = torch.zeros(1, 1, 2, 6, 14, dtype=torch.float64)
    jacobians[..., 0, 0, 0] = 1.0
    jacobians[..., 1, 0, 7] = 1.0
    goals = _identity_pose((2,))
    goals[:, 0, 3] = torch.tensor([0.2, -0.4], dtype=torch.float64)
    cost, gradient, _ = dual_ee_goal_cost_gradient(
        poses, jacobians, goals, torch.ones(2, dtype=torch.float64), component="position"
    )
    torch.testing.assert_close(cost, torch.tensor([[0.05]], dtype=torch.float64))
    assert torch.count_nonzero(gradient).item() == 2
    torch.testing.assert_close(gradient[0, 0, 0], torch.tensor(-0.1, dtype=torch.float64))
    torch.testing.assert_close(gradient[0, 0, 7], torch.tensor(0.2, dtype=torch.float64))
    assert torch.count_nonzero(gradient[..., 1:7]).item() == 0
    assert torch.count_nonzero(gradient[..., 8:]).item() == 0


def test_marvin_dual_ee_api_returns_full_14d_jacobians():
    from types import SimpleNamespace

    from mpd.bimanual.trajectory_validator import BimanualDenseTrajectoryValidator
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual

    robot = RobotMarvinBimanual(
        tensor_args={"device": "cpu", "dtype": torch.float64}
    )
    try:
        q = torch.linspace(-0.2, 0.25, 14, dtype=torch.float64).unsqueeze(0)
        jacobians, poses = robot.jfk_s_ee_bimanual(q)
        assert len(jacobians) == len(poses) == 2
        assert jacobians[0].shape == jacobians[1].shape == (1, 6, 14)
        torch.testing.assert_close(
            jacobians[0][..., 7:], torch.zeros_like(jacobians[0][..., 7:])
        )
        torch.testing.assert_close(
            jacobians[1][..., :7], torch.zeros_like(jacobians[1][..., :7])
        )
        zero = torch.zeros(1, 14, dtype=torch.float64)
        task = SimpleNamespace(
            robot=robot,
            parametric_trajectory=None,
            ee_pose_goal=torch.stack((robot.fk_left(zero)[0], robot.fk_right(zero)[0])),
            active_ee_mask=torch.ones(2, dtype=torch.float64),
            get_collision_objects_field=lambda: None,
            get_collision_ws_boundaries_field=lambda: None,
            get_collision_self_field=lambda: robot.df_collision_self,
        )
        report = BimanualDenseTrajectoryValidator(task).validate(
            q_position=zero[:, None].expand(1, 8, 14),
            q_velocity=torch.zeros(1, 8, 14, dtype=torch.float64),
            q_acceleration=torch.zeros(1, 8, 14, dtype=torch.float64),
            num_points=8,
        )
        assert report.trajectory_valid_mask.tolist() == [True]
        assert torch.isfinite(report.minimum_interarm_clearance).all()
        assert report.failure_codes == [None]
    finally:
        robot.cleanup()


def test_full_and_b3_dual_endpoint_cost_and_gradient_are_equivalent():
    trajectory = ParametricTrajectoryBspline(
        n_control_points=8,
        degree=3,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        num_T_pts=32,
        tensor_args={"device": "cpu", "dtype": torch.float64},
    )
    trajectory.set_boundary_conditions(
        q_pos_start=torch.zeros(14, dtype=torch.float64),
        q_pos_goal=torch.zeros(14, dtype=torch.float64),
    )
    task = _Task(trajectory)
    control_points = torch.linspace(-0.2, 0.3, 8, dtype=torch.float64)[None, :, None].repeat(3, 1, 14)
    full_cost, full_gradient = _guide(task, False)(control_points, return_cost=True)
    b3_cost, b3_gradient = _guide(task, True)(control_points, return_cost=True)
    torch.testing.assert_close(b3_cost, full_cost, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(b3_gradient, full_gradient, rtol=1e-10, atol=1e-11)


def _request_payload():
    pose = {"frame_id": "world", "pose_xyzw": [0.5, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0]}
    return {
        "schema": SCHEMA,
        "request_id": "phase1-stub",
        "task_mode": "dual_independent",
        "runtime_mode": "snapshot_no_time",
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
        "joint_names": list(JOINT_NAMES),
        "q_start": [0.0] * 14,
        "q_goal": [0.1] * 14,
        "left_goal_pose": pose,
        "right_goal_pose": pose,
    }


def test_contract_stub_is_explicit_and_writes_phase1_artifact(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload()))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inference/inference_marvin_bimanual.py"),
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "artifact"),
            "--backend",
            "contract_stub",
            "--stub-points",
            "8",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads((tmp_path / "artifact/result.json").read_text())
    assert result["validation"]["backend"] == "contract_stub"
    with np.load(tmp_path / "artifact/trajectory.npz") as trajectory:
        assert trajectory["positions"].shape == (8, 14)
        assert trajectory["top_k_positions"].shape == (1, 8, 14)
        assert trajectory["joint_names"].tolist() == list(JOINT_NAMES)
    assert (tmp_path / "artifact/scene.json").is_file()
    assert completed.returncode == 0


def test_python_plan_keeps_entrypoint_and_defaults_to_real_mpd():
    signature = inspect.signature(entrypoint.plan)
    assert signature.parameters["backend"].default == "mpd"
    request = BimanualRequest.from_dict(_request_payload())
    result = entrypoint.plan(request, backend="contract_stub", points=4)
    assert result["status"] == "success"


def test_fixed_warehouse_golden_suite_has_required_coverage():
    requests = json.loads(
        (ROOT / "tests/data/marvin_bimanual_warehouse_golden_requests.json").read_text()
    )
    assert len(requests) == 12
    parsed = [BimanualRequest.from_dict(item) for item in requests]
    assert len({item.request_id for item in parsed}) == len(parsed)
    cases = {item.scene["case"] for item in parsed}
    assert {"left_right_swap", "interarm_proximity", "redundant_posture"} <= cases
