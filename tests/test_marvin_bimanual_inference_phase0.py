import copy
import inspect

import pytest
import torch

from mpd.bimanual.checkpoint_contract import validate_checkpoint_args
from mpd.bimanual.planning_task import BimanualPlanningTask
from mpd.bimanual.runtime_contract import (
    BimanualRequest,
    ContractError,
    JOINT_NAMES,
    RESULT_SCHEMA,
    SCHEMA,
    validate_result,
)
from mpd.inference.inference import GenerativeOptimizationPlanner


def _pose(x):
    return {
        "frame_id": "world",
        "pose_xyzw": [x, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
    }


def _request():
    return {
        "schema": SCHEMA,
        "request_id": "phase0",
        "task_mode": "dual_independent",
        "runtime_mode": "snapshot_no_time",
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
        "joint_names": list(JOINT_NAMES),
        "q_start": [0.0] * 14,
        "left_goal_pose": _pose(0.4),
        "right_goal_pose": _pose(0.6),
    }


def test_v2_request_round_trip_and_mask():
    request = BimanualRequest.from_dict(_request())
    assert request.active_ee_mask == (1.0, 1.0)
    assert BimanualRequest.from_dict(request.to_dict()) == request


def test_dual_independent_rejects_missing_goal_and_wrong_slot_frame():
    raw = _request()
    raw["right_goal_pose"] = None
    with pytest.raises(ContractError, match="both left_goal_pose"):
        BimanualRequest.from_dict(raw)
    raw = _request()
    raw["left_goal_pose"]["frame_id"] = "base_link"
    with pytest.raises(ContractError, match="planning_frame"):
        BimanualRequest.from_dict(raw)


def test_failed_result_does_not_require_a_trajectory():
    request = BimanualRequest.from_dict(_request())
    result = {
        "schema": RESULT_SCHEMA,
        "request_id": request.request_id,
        "status": "no_valid_trajectory",
        "error": {"message": "no safe candidate"},
    }
    assert validate_result(result, request=request)["status"] == "no_valid_trajectory"


def test_success_result_checks_start_and_derivative_shapes():
    request = BimanualRequest.from_dict(_request())
    result = {
        "schema": RESULT_SCHEMA,
        "request_id": request.request_id,
        "status": "success",
        "joint_names": list(JOINT_NAMES),
        "positions": [[0.0] * 14, [0.1] * 14],
        "velocities": [[0.0] * 14, [0.0] * 14],
        "accelerations": [[0.0] * 14, [0.0] * 14],
        "time_from_start": [0.0, 1.0],
        "world_version": 0,
        "validation": {"valid": True},
    }
    assert validate_result(result, request=request)["status"] == "success"
    bad = copy.deepcopy(result)
    bad["positions"][0][0] = 0.1
    with pytest.raises(ContractError, match="start differs"):
        validate_result(bad, request=request)


def test_checkpoint_contract_accepts_all_variants_and_rejects_drift():
    base = {
        "robot_model": "marvin_bimanual",
        "task_family": "independent",
        "context_qs": True,
        "context_ee_goal_pose": True,
        "context_ee_goal_pose_bimanual": True,
        "state_dim": 14,
        "context_q_dim": 14,
        "raw_context_dim": 40,
        "parametric_trajectory_class": "ParametricTrajectoryBspline",
        "bspline_num_control_points_desired": 22,
        "bspline_num_control_points_exact": True,
        "dataset_subdir": "warehouse",
    }
    for variant in "ABCD":
        args = dict(base, bimanual_network_variant=variant)
        assert validate_checkpoint_args(
            args, expected_dataset_subdir="warehouse", expected_variant=variant
        )["bimanual_network_variant"] == variant
    with pytest.raises(ValueError, match="raw_context_dim"):
        validate_checkpoint_args(dict(base, raw_context_dim=28))


def test_bimanual_goal_setter_keeps_fixed_slot_order():
    task = object.__new__(BimanualPlanningTask)
    task.q_pos_start = torch.zeros(14)
    poses = torch.zeros(2, 3, 4)
    poses[0, 0, 3] = 1.0
    poses[1, 0, 3] = 2.0
    BimanualPlanningTask.set_ee_pose_goal(
        task, poses, torch.tensor([1.0, 1.0])
    )
    assert task.left_goal_pose[0, 3] == 1.0
    assert task.right_goal_pose[0, 3] == 2.0


def test_python_planner_entrypoint_exposes_optional_mask():
    signature = inspect.signature(GenerativeOptimizationPlanner.plan_trajectory)
    assert signature.parameters["active_ee_mask"].default is None
