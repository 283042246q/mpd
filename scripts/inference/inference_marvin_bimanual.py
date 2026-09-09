#!/usr/bin/env python3
"""Run one Marvin bimanual MPD request and write a portable artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpd.bimanual.checkpoint_contract import validate_checkpoint_args
from mpd.bimanual.runtime_contract import (
    BimanualRequest,
    ContractError,
    JOINT_NAMES,
    RESULT_SCHEMA,
    SUCCESS_STATUS,
    validate_result,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "scripts/inference/cfgs/config_EnvWarehouse-RobotMarvinBimanual-independent-runtime.yaml"
)


class InferenceConfigurationError(RuntimeError):
    pass


class NoValidTrajectoryError(RuntimeError):
    pass


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _pose_xyzw_to_matrix(pose: tuple[float, ...]) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = pose
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    rotation = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return np.concatenate((rotation, np.asarray([[x], [y], [z]])), axis=1)


def _matrix_to_pose_xyzw(matrix: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(matrix, dtype=np.float64)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [*matrix[:3, 3].tolist(), *quaternion.tolist()]


def _runtime_value(config, key, default=None):
    runtime = config.get("runtime", {})
    return runtime.get(key, config.get(key, default))


def _validate_runtime_config(config: dict) -> None:
    expected = {
        "schema": "marvin_bimanual_request/v2",
        "result_schema": RESULT_SCHEMA,
        "robot_model": "marvin_bimanual",
        "planning_frame": "world",
        "scene_id": "EnvWarehouseMarvinBimanual",
        "scene_version": "marvin_warehouse_v2",
    }
    for key, wanted in expected.items():
        actual = _runtime_value(config, key)
        if actual != wanted:
            raise InferenceConfigurationError(
                f"runtime.{key} must be {wanted!r}, got {actual!r}"
            )
    if tuple(_runtime_value(config, "joint_names", ())) != JOINT_NAMES:
        raise InferenceConfigurationError("runtime.joint_names is not canonical")
    if config.get("task_mode") != "dual_independent":
        raise InferenceConfigurationError(
            "Phase 1 runtime supports task_mode=dual_independent only"
        )


def _resolve_model_dir(config: dict) -> Path:
    if config.get("model_selection") != "bspline":
        raise InferenceConfigurationError("Marvin runtime requires model_selection=bspline")
    value = config.get("model_dir_ddpm_bspline")
    if not isinstance(value, str) or not value:
        raise InferenceConfigurationError("model_dir_ddpm_bspline is required")
    configured = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if (configured / "args.yaml").is_file():
        return configured
    # Training creates one timestamped run below the experiment directory.
    # Resolve that layout only when it is unambiguous; never guess between
    # multiple checkpoints because doing so would break reproducibility.
    runs = sorted(
        child
        for child in configured.iterdir()
        if child.is_dir() and (child / "args.yaml").is_file()
    ) if configured.is_dir() else []
    if len(runs) == 1:
        return runs[0]
    if not runs:
        return configured
    raise InferenceConfigurationError(
        f"Model directory {configured} contains multiple runs; configure an exact run path"
    )


def _deadline_guard(request: BimanualRequest) -> None:
    if request.deadline_unix_ns and time.time_ns() >= request.deadline_unix_ns:
        raise TimeoutError("request deadline has expired")


def _stub_plan(request: BimanualRequest, points: int, duration_s: float):
    """Explicit contract-only backend; never selected by default."""
    start = np.asarray(request.q_start, dtype=np.float64)
    goal = np.asarray(request.q_goal or request.q_start, dtype=np.float64)
    count = max(2, int(points))
    alpha = np.linspace(0.0, 1.0, count)[:, None]
    positions = start + alpha * (goal - start)
    times = np.linspace(0.0, float(duration_s), count)
    edge_order = 2 if count > 2 else 1
    velocities = np.gradient(positions, times, axis=0, edge_order=edge_order)
    accelerations = np.gradient(velocities, times, axis=0, edge_order=edge_order)
    result = {
        "schema": RESULT_SCHEMA,
        "request_id": request.request_id,
        "status": SUCCESS_STATUS,
        "joint_names": list(JOINT_NAMES),
        "positions": positions.tolist(),
        "velocities": velocities.tolist(),
        "accelerations": accelerations.tolist(),
        "time_from_start": times.tolist(),
        "world_version": request.world_version,
        "trajectory_file": "trajectory.npz",
        "scene_file": "scene.json",
        "validation": {
            "valid": True,
            "backend": "contract_stub",
            "warning": "No model sampling or collision validation was performed",
        },
    }
    arrays = {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "time_from_start": times,
        "top_k_positions": positions[None],
        "top_k_velocities": velocities[None],
        "top_k_accelerations": accelerations[None],
        "top_k_candidate_indices": np.asarray([0], dtype=np.int64),
        "joint_names": np.asarray(JOINT_NAMES, dtype=np.str_),
    }
    if request.left_goal_pose is not None and request.right_goal_pose is not None:
        arrays["ee_goal_pose"] = np.stack(
            (
                _pose_xyzw_to_matrix(request.left_goal_pose),
                _pose_xyzw_to_matrix(request.right_goal_pose),
            )
        )
        arrays["active_ee_mask"] = np.asarray(request.active_ee_mask, dtype=np.float64)
    scene = {
        "schema": "mpd_isaaclab_scene",
        "schema_version": 1,
        "env_name": request.scene_id,
        "frame_id": request.planning_frame,
        "obstacles": [],
        "unsupported_obstacles": [],
        "backend": "contract_stub",
    }
    return validate_result(result, request=request), arrays, scene


def _real_plan(request: BimanualRequest, config_path: Path, device_text: str):
    import torch
    from dotmap import DotMap

    from mpd.bimanual.planning_task import BimanualPlanningTask
    from mpd.inference.inference import (
        GenerativeOptimizationPlanner,
        resolve_model_checkpoint_path,
    )
    from mpd.paths import DATASET_BASE_DIR
    from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml
    from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
    from torch_robotics.robots.robot_marvin_bimanual import RobotMarvinBimanual
    from torch_robotics.torch_utils.seed import fix_random_seed

    _deadline_guard(request)
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise InferenceConfigurationError(
            f"CUDA device {device_text!r} requested but CUDA is unavailable"
        )
    tensor_args = {"device": torch.device(device_text), "dtype": torch.float32}
    fix_random_seed(request.seed)

    raw_config = load_params_from_yaml(config_path)
    _validate_runtime_config(raw_config)
    config = DotMap(raw_config)
    model_dir = _resolve_model_dir(raw_config)
    args_path = model_dir / "args.yaml"
    if not args_path.is_file():
        raise InferenceConfigurationError(f"Model args file not found: {args_path}")
    raw_train = load_params_from_yaml(args_path)
    validated_train = validate_checkpoint_args(
        raw_train,
        expected_dataset_subdir=config.dataset_subdir,
        expected_variant=config.runtime.network_variant,
    )
    try:
        checkpoint_path = resolve_model_checkpoint_path(
            model_dir,
            bool(validated_train.get("use_ema", False)),
            checkpoint=config.get("checkpoint"),
        )
    except (FileNotFoundError, ValueError) as error:
        raise InferenceConfigurationError(str(error)) from error
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if request.checkpoint_hash and request.checkpoint_hash != checkpoint_sha256:
        raise InferenceConfigurationError("request checkpoint_hash does not match checkpoint")

    dataset_dir = Path(DATASET_BASE_DIR) / config.dataset_subdir
    for required in (dataset_dir / "args.yaml", dataset_dir / config.dataset_file_merged):
        if not required.is_file():
            raise InferenceConfigurationError(
                f"Inference dataset artifact not found: {required}. Merge the generated shards first."
            )

    args_train = DotMap(raw_train)
    args_inference = DotMap(raw_config)
    args_inference.model_dir = model_dir.as_posix()
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=model_dir.as_posix(),
        load_indices=(model_dir / "train_subset_indices.pt").is_file(),
        tensor_args=tensor_args,
    )
    planning_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)
    if not isinstance(planning_task, BimanualPlanningTask):
        raise InferenceConfigurationError("Loader did not construct BimanualPlanningTask")
    if not isinstance(planning_task.robot, RobotMarvinBimanual):
        raise InferenceConfigurationError("Loader did not construct RobotMarvinBimanual")
    robot = planning_task.robot
    robot_hash = getattr(robot, "asset_hash", None)
    if request.robot_model_hash and request.robot_model_hash != robot_hash:
        raise InferenceConfigurationError("request robot_model_hash does not match Marvin asset")

    scene = export_isaaclab_scene_payload(planning_task.env, include_boxes=True)
    scene["frame_id"] = request.planning_frame
    scene_hash = _sha256_json(scene)
    if request.scene_hash and request.scene_hash != scene_hash:
        raise InferenceConfigurationError("request scene_hash does not match Warehouse scene")

    q_start = torch.as_tensor(request.q_start, **tensor_args)
    q_goal = torch.as_tensor(request.q_goal or request.q_start, **tensor_args)
    q_velocity_start = torch.as_tensor(request.q_velocity_start, **tensor_args)
    q_acceleration_start = torch.as_tensor(request.q_acceleration_start, **tensor_args)
    if torch.any(q_start < robot.q_pos_min) or torch.any(q_start > robot.q_pos_max):
        raise ContractError("q_start violates Marvin joint limits")
    if torch.any(q_goal < robot.q_pos_min) or torch.any(q_goal > robot.q_pos_max):
        raise ContractError("q_goal violates Marvin joint limits")

    fk_goals = (robot.fk_left(q_goal), robot.fk_right(q_goal))
    goal_values = (request.left_goal_pose, request.right_goal_pose)
    goal_matrices = []
    for supplied, fk_goal in zip(goal_values, fk_goals):
        goal_matrices.append(
            fk_goal.detach().cpu().numpy()
            if supplied is None
            else _pose_xyzw_to_matrix(supplied)
        )
    ee_goal = torch.as_tensor(np.stack(goal_matrices), **tensor_args)
    active_mask = torch.as_tensor(request.active_ee_mask, **tensor_args)

    planner = GenerativeOptimizationPlanner(
        planning_task,
        train_subset.dataset,
        args_train,
        args_inference,
        tensor_args,
        sampling_based_planner_fn=None,
        debug=False,
    )
    results = planner.plan_trajectory(
        q_start,
        q_goal,
        ee_goal,
        active_ee_mask=active_mask,
        q_vel_start=q_velocity_start,
        q_acc_start=q_acceleration_start,
        results_ns=DotMap(t_generator=0.0, t_guide=0.0),
        debug=False,
    )
    _deadline_guard(request)
    valid_indices = torch.nonzero(results.valid_trajectory_mask).flatten()
    if valid_indices.numel() == 0 or results.q_trajs_pos_best is None:
        raise NoValidTrajectoryError("MPD produced no dense-valid trajectory")

    best_positions = results.q_trajs_pos_best
    best_velocities = results.q_trajs_vel_best
    best_accelerations = results.q_trajs_acc_best
    best_report = planner.dense_validator.validate(
        q_position=best_positions.unsqueeze(0),
        q_velocity=best_velocities.unsqueeze(0),
        q_acceleration=best_accelerations.unsqueeze(0),
        num_points=int(config.dense_validation.runtime_points),
        check_environment=True,
        check_self_collision=True,
        check_joint_position=True,
        check_joint_velocity=True,
        check_joint_acceleration=True,
    )
    if not bool(best_report.trajectory_valid_mask[0].item()):
        raise NoValidTrajectoryError(
            f"Selected trajectory failed final oracle: {best_report.failure_codes[0]}"
        )

    selected_index = int(
        results.best_trajectory_selection_details["selected_candidate_index"]
    )
    scores = results.valid_trajectory_selection_scores
    ordered_valid = valid_indices if scores is None else valid_indices[torch.argsort(scores)]
    top_k_count = min(int(config.runtime_top_k_valid_trajectories), ordered_valid.numel())
    if top_k_count < 1:
        raise InferenceConfigurationError("runtime_top_k_valid_trajectories must be positive")
    top_k_indices = ordered_valid[:top_k_count]
    if selected_index not in top_k_indices.tolist():
        selected_tensor = torch.as_tensor(
            [selected_index], dtype=top_k_indices.dtype, device=top_k_indices.device
        )
        top_k_indices = torch.cat((selected_tensor, top_k_indices[: top_k_count - 1]))
    else:
        selected_offset = top_k_indices.tolist().index(selected_index)
        if selected_offset:
            top_k_indices = torch.cat(
                (
                    top_k_indices[selected_offset : selected_offset + 1],
                    top_k_indices[:selected_offset],
                    top_k_indices[selected_offset + 1 :],
                )
            )

    top_positions = results.q_trajs_pos_iter_0.index_select(0, top_k_indices)
    top_velocities = results.q_trajs_vel_iter_0.index_select(0, top_k_indices)
    top_accelerations = results.q_trajs_acc_iter_0.index_select(0, top_k_indices)
    times = results.timesteps

    validation = {
        "valid": True,
        "failure_code": None,
        "left_ee_position_error_m": best_report.ee_position_error_m[0, 0],
        "right_ee_position_error_m": best_report.ee_position_error_m[0, 1],
        "left_ee_orientation_error_rad": best_report.ee_orientation_error_rad[0, 0],
        "right_ee_orientation_error_rad": best_report.ee_orientation_error_rad[0, 1],
        "minimum_environment_clearance_m": best_report.minimum_environment_clearance[0],
        "minimum_self_clearance_m": best_report.minimum_self_clearance[0],
        "minimum_left_self_clearance_m": best_report.minimum_left_self_clearance[0],
        "minimum_right_self_clearance_m": best_report.minimum_right_self_clearance[0],
        "minimum_interarm_clearance_m": best_report.minimum_interarm_clearance[0],
        "self_collision_pair_counts": planner.cost_guide.self_collision_pair_counts,
        "joint_position_violation": best_report.joint_position_violation_mask[0],
        "joint_velocity_violation": best_report.joint_velocity_violation_mask[0],
        "joint_acceleration_violation": best_report.joint_acceleration_violation_mask[0],
    }
    result = {
        "schema": RESULT_SCHEMA,
        "request_id": request.request_id,
        "status": SUCCESS_STATUS,
        "joint_names": list(JOINT_NAMES),
        "positions": best_positions,
        "velocities": best_velocities,
        "accelerations": best_accelerations,
        "time_from_start": times,
        "world_version": request.world_version,
        "trajectory_file": "trajectory.npz",
        "scene_file": "scene.json",
        "validation": validation,
        "goals": {
            "active_ee_mask": active_mask,
            "left_pose_xyzw": _matrix_to_pose_xyzw(goal_matrices[0]),
            "right_pose_xyzw": _matrix_to_pose_xyzw(goal_matrices[1]),
        },
        "model": {
            "directory": model_dir,
            "checkpoint": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "args_sha256": _sha256_file(args_path),
            "network_variant": config.runtime.network_variant,
        },
        "scene": {
            "scene_id": request.scene_id,
            "scene_version": request.scene_version,
            "scene_sha256": scene_hash,
            "robot_asset_sha256": robot_hash,
        },
        "candidates": {
            "generated": int(results.q_trajs_pos_iter_0.shape[0]),
            "dense_checked": int(results.dense_validation_candidates_checked),
            "valid": int(valid_indices.numel()),
            "top_k_saved": int(top_k_count),
            "top_k_candidate_indices": top_k_indices,
            "selected_candidate_index": selected_index,
        },
        "timing": {
            "inference_total_s": results.t_inference_total,
            "generator_s": results.t_generator,
            "guide_s": results.t_guide,
            "dense_validation_s": results.dense_validation_time,
            "ranking_s": results.trajectory_ranking_time,
        },
        "gradient_pruning": results.gradient_pruning_statistics,
        "created_unix_ns": time.time_ns(),
    }
    result = validate_result(_jsonable(result), request=request)
    arrays = {
        "positions": best_positions.detach().cpu().numpy().astype(np.float64),
        "velocities": best_velocities.detach().cpu().numpy().astype(np.float64),
        "accelerations": best_accelerations.detach().cpu().numpy().astype(np.float64),
        "time_from_start": times.detach().cpu().numpy().astype(np.float64),
        "top_k_positions": top_positions.detach().cpu().numpy().astype(np.float64),
        "top_k_velocities": top_velocities.detach().cpu().numpy().astype(np.float64),
        "top_k_accelerations": top_accelerations.detach().cpu().numpy().astype(np.float64),
        "top_k_candidate_indices": top_k_indices.detach().cpu().numpy().astype(np.int64),
        "joint_names": np.asarray(JOINT_NAMES, dtype=np.str_),
        "ee_goal_pose": ee_goal.detach().cpu().numpy().astype(np.float64),
        "active_ee_mask": active_mask.detach().cpu().numpy().astype(np.float64),
        "mpd_tcp_pose_start": np.stack(
            (
                robot.fk_left(q_start).detach().cpu().numpy(),
                robot.fk_right(q_start).detach().cpu().numpy(),
            )
        ).astype(np.float64),
        "mpd_top_k_tcp_pose_final": torch.stack(
            (
                robot.fk_left(top_positions[:, -1]),
                robot.fk_right(top_positions[:, -1]),
            ),
            dim=1,
        ).detach().cpu().numpy().astype(np.float64),
    }
    return result, arrays, scene


def plan(
    request: BimanualRequest,
    *,
    backend: str = "mpd",
    config_path: Path = DEFAULT_CONFIG,
    device: str = "cuda:0",
    points: int = 64,
    duration_s: float = 2.0,
):
    """Python API retained for callers; real MPD is the default backend."""
    if backend == "contract_stub":
        return _stub_plan(request, points, duration_s)[0]
    if backend != "mpd":
        raise ValueError("backend must be mpd or contract_stub")
    return _real_plan(request, Path(config_path), device)[0]


def _failure(request_id, status, error):
    return {
        "schema": RESULT_SCHEMA,
        "request_id": request_id,
        "status": status,
        "error": {"type": type(error).__name__, "message": str(error)},
        "created_unix_ns": time.time_ns(),
    }


def _release_inference_resources() -> None:
    """Release MPD allocations before Isaac Lab starts in a separate process."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _run_isaaclab_backend(args, output_dir: Path) -> dict:
    from scripts.isaaclab.marvin_bimanual_subprocess import (
        run_marvin_isaaclab_evaluator,
        run_marvin_isaaclab_replay,
    )

    evaluation_path = output_dir / "isaaclab-evaluation.json"
    evaluation = run_marvin_isaaclab_evaluator(
        output_dir,
        evaluation_path,
        output_dir / "isaaclab-evaluation.log",
        conda_env=args.isaaclab_conda_env,
        device=args.isaaclab_device,
        headless=args.isaaclab_headless,
        action_repeat=args.isaaclab_action_repeat,
        timeout_s=args.isaaclab_timeout_s,
        asset_cache=args.isaaclab_asset_cache,
    )
    summary = {
        "schema": "marvin_bimanual_isaaclab_run/v1",
        "status": (
            "safety_validation_failed"
            if evaluation.get("safety_false_negative")
            else "completed"
        ),
        "artifact": output_dir.as_posix(),
        "evaluation": evaluation,
        "replay": None,
    }
    if args.isaaclab_replay:
        video_path = None
        screenshot_path = None
        if args.isaaclab_capture:
            video_path = args.isaaclab_video or (output_dir / "isaaclab-replay.mp4")
            screenshot_path = args.isaaclab_screenshot or (output_dir / "isaaclab-replay.png")
        summary["replay"] = run_marvin_isaaclab_replay(
            output_dir,
            output_dir / "isaaclab-replay.json",
            output_dir / "isaaclab-replay.log",
            evaluation=evaluation_path,
            video_path=video_path,
            screenshot_path=screenshot_path,
            trajectory_index=args.isaaclab_trajectory_index,
            conda_env=args.isaaclab_conda_env,
            device=args.isaaclab_device,
            headless=args.isaaclab_headless,
            action_repeat=args.isaaclab_action_repeat,
            timeout_s=args.isaaclab_timeout_s,
            video_fps=args.isaaclab_video_fps,
            width=args.isaaclab_width,
            height=args.isaaclab_height,
            asset_cache=args.isaaclab_asset_cache,
        )
    return summary


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--output", type=Path, help="Compatibility path for result.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--backend",
        choices=("mpd", "contract_stub"),
        default="mpd",
        help="contract_stub is for schema/wiring tests only",
    )
    parser.add_argument("--stub-points", type=int, default=64)
    parser.add_argument("--stub-duration", type=float, default=2.0)
    parser.add_argument(
        "--sim-backend",
        choices=("none", "isaaclab"),
        default="none",
        help="Optionally validate and replay the completed artifact in a separate Isaac Lab process",
    )
    parser.add_argument("--isaaclab-conda-env", default="env_isaaclab")
    parser.add_argument("--isaaclab-device", default="cuda:0")
    parser.add_argument("--isaaclab-action-repeat", type=int, default=4)
    parser.add_argument("--isaaclab-timeout-s", type=int, default=900)
    parser.add_argument("--isaaclab-trajectory-index", type=int, default=0)
    parser.add_argument("--isaaclab-video-fps", type=float, default=24.0)
    parser.add_argument("--isaaclab-width", type=int, default=960)
    parser.add_argument("--isaaclab-height", type=int, default=540)
    parser.add_argument(
        "--isaaclab-asset-cache",
        type=Path,
        default=REPO_ROOT / ".cache/isaaclab/marvin_bimanual",
    )
    parser.add_argument("--isaaclab-video", type=Path, default=None)
    parser.add_argument("--isaaclab-screenshot", type=Path, default=None)
    headless = parser.add_mutually_exclusive_group()
    headless.add_argument("--isaaclab-headless", dest="isaaclab_headless", action="store_true")
    headless.add_argument("--no-isaaclab-headless", dest="isaaclab_headless", action="store_false")
    parser.set_defaults(isaaclab_headless=True)
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--isaaclab-replay", dest="isaaclab_replay", action="store_true")
    replay.add_argument("--no-isaaclab-replay", dest="isaaclab_replay", action="store_false")
    parser.set_defaults(isaaclab_replay=True)
    capture = parser.add_mutually_exclusive_group()
    capture.add_argument("--isaaclab-capture", dest="isaaclab_capture", action="store_true")
    capture.add_argument("--no-isaaclab-capture", dest="isaaclab_capture", action="store_false")
    parser.set_defaults(isaaclab_capture=True)
    return parser


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if (
        args.isaaclab_action_repeat < 1
        or args.isaaclab_timeout_s < 1
        or args.isaaclab_video_fps <= 0.0
        or args.isaaclab_width < 1
        or args.isaaclab_height < 1
    ):
        raise SystemExit("Isaac Lab repeat, timeout, fps, width, and height must be positive")
    if args.isaaclab_trajectory_index < 0:
        raise SystemExit("--isaaclab-trajectory-index must be non-negative")
    if args.output is not None:
        result_path = args.output.expanduser().resolve()
        output_dir = result_path.parent
    else:
        output_dir = (args.output_dir or Path.cwd()).expanduser().resolve()
        result_path = output_dir / "result.json"
    trajectory_path = output_dir / "trajectory.npz"
    scene_path = output_dir / "scene.json"
    trajectory_path.unlink(missing_ok=True)
    scene_path.unlink(missing_ok=True)
    request_id = None
    try:
        raw_request = json.loads(args.request.expanduser().read_text())
        request_id = raw_request.get("request_id")
        request = BimanualRequest.from_dict(raw_request)
        request_id = request.request_id
        if args.backend == "contract_stub":
            result, arrays, scene = _stub_plan(
                request, args.stub_points, args.stub_duration
            )
        else:
            result, arrays, scene = _real_plan(
                request, args.config.expanduser().resolve(), args.device
            )
        _write_npz(trajectory_path, arrays)
        _write_json(scene_path, scene)
        _write_json(result_path, result)
        if args.sim_backend == "isaaclab":
            _release_inference_resources()
            isaaclab_run_path = output_dir / "isaaclab-run.json"
            try:
                isaaclab_summary = _run_isaaclab_backend(args, output_dir)
                _write_json(isaaclab_run_path, isaaclab_summary)
                if isaaclab_summary["status"] != "completed":
                    return 7
            except Exception as error:
                _write_json(
                    isaaclab_run_path,
                    {
                        "schema": "marvin_bimanual_isaaclab_run/v1",
                        "status": "fault",
                        "artifact": output_dir.as_posix(),
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                )
                print(error, file=sys.stderr)
                return 6
        print(result_path)
        return 0
    except ContractError as error:
        result = _failure(request_id, "invalid_request", error)
        _write_json(result_path, result)
        print(error, file=sys.stderr)
        return 2
    except TimeoutError as error:
        result = _failure(request_id, "deadline_exceeded", error)
        _write_json(result_path, result)
        print(error, file=sys.stderr)
        return 3
    except NoValidTrajectoryError as error:
        result = _failure(request_id, "no_valid_trajectory", error)
        _write_json(result_path, result)
        print(error, file=sys.stderr)
        return 4
    except Exception as error:
        result = _failure(request_id, "fault", error)
        _write_json(result_path, result)
        print(error, file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
