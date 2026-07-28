#!/usr/bin/env python3
"""Run one MPD Warehouse request and export a neutral trajectory result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from mpd.utils.patches import numpy_monkey_patch

numpy_monkey_patch()

import numpy as np

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts/inference/cfgs/config_EnvWarehouse-RobotPanda-runtime.yaml"
EXPECTED_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
EXPECTED_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
DEFAULT_REQUEST_ID = "diffusion_planner_example"
DEFAULT_SCENE_ID = "EnvWarehouseExtraObjectsV00"
DEFAULT_SEED = 0


class RuntimeContractError(RuntimeError):
    status = "runtime_error"
    exit_code = 1


class RequestValidationError(RuntimeContractError):
    status = "invalid_request"
    exit_code = 2


class ConfigurationError(RuntimeContractError):
    status = "invalid_configuration"
    exit_code = 2


class NoValidTrajectoryError(RuntimeContractError):
    status = "no_valid_trajectory"
    exit_code = 3


class ResultValidationError(RuntimeContractError):
    status = "invalid_result"
    exit_code = 4


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        return _jsonable(tensor.item() if tensor.numel() == 1 else tensor.tolist())
    if hasattr(value, "toDict"):
        return _jsonable(value.toDict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _jsonable(payload),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def _load_request(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw_request = path.read_bytes()
    except OSError as error:
        raise RequestValidationError(f"Cannot read request file {path}: {error}") from error
    try:
        request = json.loads(raw_request)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RequestValidationError(f"Request must be valid UTF-8 JSON: {error}") from error
    if not isinstance(request, dict):
        raise RequestValidationError("Request root must be a JSON object.")
    return request, _sha256_bytes(raw_request)


def _require_string(request: dict[str, Any], key: str) -> str:
    value = request.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{key} must be a non-empty string.")
    return value


def _require_vector(
    request: dict[str, Any],
    key: str,
    *,
    default: np.ndarray | None = None,
) -> np.ndarray:
    if key not in request:
        if default is None:
            raise RequestValidationError(f"Missing required field: {key}.")
        return default.copy()
    try:
        value = np.asarray(request[key], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"{key} must contain numeric values.") from error
    if value.shape != (len(EXPECTED_JOINT_NAMES),):
        raise RequestValidationError(f"{key} must have shape [{len(EXPECTED_JOINT_NAMES)}], got {list(value.shape)}.")
    if not np.isfinite(value).all():
        raise RequestValidationError(f"{key} contains NaN or Inf.")
    return value


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    schema_version = request.get("schema_version", EXPECTED_SCHEMA_VERSION)
    if isinstance(schema_version, bool) or schema_version != EXPECTED_SCHEMA_VERSION:
        raise RequestValidationError(f"schema_version must be {EXPECTED_SCHEMA_VERSION}, got {schema_version!r}.")

    request_id = request.get("request_id", DEFAULT_REQUEST_ID)
    if not isinstance(request_id, str) or not request_id.strip():
        raise RequestValidationError("request_id must be a non-empty string.")
    scene_id = request.get("scene_id", DEFAULT_SCENE_ID)
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise RequestValidationError("scene_id must be a non-empty string.")

    joint_names = request.get("joint_names")
    if not isinstance(joint_names, list) or tuple(joint_names) != EXPECTED_JOINT_NAMES:
        raise RequestValidationError(
            "joint_names must exactly match the ordered Panda joints: " f"{list(EXPECTED_JOINT_NAMES)}."
        )

    seed = request.get("seed", DEFAULT_SEED)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise RequestValidationError("seed must be an integer in [0, 2**32).")

    robot_model = request.get("robot_model")
    if robot_model is not None and robot_model != "franka_panda":
        raise RequestValidationError("robot_model must be 'franka_panda' when provided.")
    planning_frame = request.get("planning_frame")
    if planning_frame is not None and planning_frame != "panda_link0":
        raise RequestValidationError("planning_frame must be 'panda_link0' when provided.")
    scene_hash = request.get("scene_hash")
    if scene_hash is not None and (not isinstance(scene_hash, str) or not scene_hash.strip()):
        raise RequestValidationError("scene_hash must be a non-empty string when provided.")

    zero_boundary = np.zeros(len(EXPECTED_JOINT_NAMES), dtype=np.float64)
    return {
        "schema_version": schema_version,
        "request_id": request_id,
        "scene_id": scene_id,
        "joint_names": list(joint_names),
        "seed": seed,
        "q_pos_start": _require_vector(request, "q_pos_start"),
        "q_pos_goal": _require_vector(request, "q_pos_goal"),
        "q_vel_start": _require_vector(request, "q_vel_start", default=zero_boundary),
        "q_vel_goal": _require_vector(request, "q_vel_goal", default=zero_boundary),
        "q_acc_start": _require_vector(request, "q_acc_start", default=zero_boundary),
        "q_acc_goal": _require_vector(request, "q_acc_goal", default=zero_boundary),
        "robot_model": robot_model or "franka_panda",
        "planning_frame": planning_frame or "panda_link0",
        "joint_state_stamp": request.get("joint_state_stamp"),
        "scene_hash": scene_hash,
    }


def _git_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"commit": None, "dirty": None}
    try:
        metadata["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        metadata["dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return metadata


def _resolve_model_dir(args_inference: Any) -> Path:
    if "cvae" in args_inference.planner_alg:
        model_key = f"model_dir_cvae_{args_inference.model_selection}"
    else:
        model_key = f"model_dir_ddpm_{args_inference.model_selection}"
    model_dir = args_inference.get(model_key)
    if not model_dir:
        raise ConfigurationError(f"Missing model directory setting: {model_key}.")
    return Path(os.path.expandvars(os.path.expanduser(str(model_dir)))).resolve()


def _runtime_config_value(args_inference: Any, key: str, default: Any) -> Any:
    runtime_config = args_inference.get("runtime", {})
    if hasattr(runtime_config, "get"):
        return runtime_config.get(key, default)
    return default


def _validate_runtime_config(args_inference: Any) -> None:
    required_values = {
        "start_goal_source": "states_file",
        "model_selection": "bspline",
        "planner_alg": "mpd",
        "diffusion_sampling_method": "ddim",
        "env_id_replace": "EnvWarehouseExtraObjectsV00",
        "num_T_pts": 128,
        "n_trajectory_samples": 100,
    }
    for key, expected in required_values.items():
        actual = args_inference.get(key)
        if actual != expected:
            raise ConfigurationError(f"{key} must be {expected!r}, got {actual!r}.")
    if not math.isclose(float(args_inference.trajectory_duration), 10.0, abs_tol=1e-9):
        raise ConfigurationError(
            f"trajectory_duration must be 10.0 seconds, got {args_inference.trajectory_duration!r}."
        )
    configured_joint_names = tuple(_runtime_config_value(args_inference, "joint_names", EXPECTED_JOINT_NAMES))
    if configured_joint_names != EXPECTED_JOINT_NAMES:
        raise ConfigurationError("runtime.joint_names does not match the Panda runtime contract.")

    runtime_schema_version = _runtime_config_value(args_inference, "schema_version", None)
    if runtime_schema_version != EXPECTED_SCHEMA_VERSION:
        raise ConfigurationError(
            f"runtime.schema_version must be {EXPECTED_SCHEMA_VERSION}, got {runtime_schema_version!r}."
        )


def _validate_device(device_text: str):
    import torch

    try:
        device = torch.device(device_text)
    except (TypeError, RuntimeError) as error:
        raise ConfigurationError(f"Invalid torch device {device_text!r}.") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError(f"CUDA device requested but CUDA is unavailable: {device_text}.")
        device_index = 0 if device.index is None else device.index
        if device_index >= torch.cuda.device_count():
            raise ConfigurationError(
                f"CUDA device index {device_index} is unavailable; found {torch.cuda.device_count()} device(s)."
            )
    return device


def _validate_robot_state(planning_task: Any, q_pos: Any, state_name: str) -> None:
    import torch

    robot = planning_task.robot
    if bool(torch.any(q_pos < robot.q_pos_min).item()) or bool(torch.any(q_pos > robot.q_pos_max).item()):
        raise RequestValidationError(f"{state_name} is outside the MPD Panda joint position limits.")
    collisions = planning_task.compute_collision(
        q_pos,
        margin=planning_task.margin_for_dense_collision_checking,
    )
    if bool(torch.as_tensor(collisions).any().item()):
        raise RequestValidationError(f"{state_name} is in collision in the configured MPD scene.")


def _validate_boundary_derivative(values: Any, limits: Any, field_name: str) -> None:
    import torch

    if limits is None:
        return
    utilization = torch.abs(values) / limits
    if bool(torch.any(utilization > 1.0).item()):
        max_utilization = float(torch.amax(utilization).item())
        raise RequestValidationError(f"{field_name} exceeds the MPD Panda limit (utilization={max_utilization:.6f}).")


def _validate_best_trajectory(
    results: Any,
    planning_task: Any,
    q_pos_start: Any,
    expected_horizon: int,
    expected_duration: float,
) -> dict[str, float]:
    import torch

    positions = results.q_trajs_pos_best
    velocities = results.q_trajs_vel_best
    accelerations = results.q_trajs_acc_best
    timesteps = results.timesteps
    expected_shape = (expected_horizon, len(EXPECTED_JOINT_NAMES))

    for name, values in (
        ("positions", positions),
        ("velocities", velocities),
        ("accelerations", accelerations),
    ):
        if values is None or tuple(values.shape) != expected_shape:
            actual_shape = None if values is None else list(values.shape)
            raise ResultValidationError(f"{name} must have shape {list(expected_shape)}, got {actual_shape}.")
        if not bool(torch.isfinite(values).all().item()):
            raise ResultValidationError(f"{name} contains NaN or Inf.")

    if timesteps is None or tuple(timesteps.shape) != (expected_horizon,):
        actual_shape = None if timesteps is None else list(timesteps.shape)
        raise ResultValidationError(f"time_from_start must have shape [{expected_horizon}], got {actual_shape}.")
    if not bool(torch.isfinite(timesteps).all().item()):
        raise ResultValidationError("time_from_start contains NaN or Inf.")
    if abs(float(timesteps[0].item())) > 1e-8:
        raise ResultValidationError("time_from_start must begin at 0 seconds.")
    if not bool(torch.all(torch.diff(timesteps) > 0).item()):
        raise ResultValidationError("time_from_start must be strictly increasing.")
    if not math.isclose(float(timesteps[-1].item()), expected_duration, abs_tol=1e-5):
        raise ResultValidationError(
            f"Final time must be {expected_duration:.6f} seconds, got {timesteps[-1].item():.6f}."
        )

    start_tolerance = 1e-5
    start_error = float(torch.amax(torch.abs(positions[0] - q_pos_start)).item())
    if start_error > start_tolerance:
        raise ResultValidationError(
            f"Trajectory start differs from q_pos_start by {start_error:.6g} rad " f"(limit {start_tolerance:.6g})."
        )

    robot = planning_task.robot
    limit_tolerance = 1e-5
    if bool(torch.any(positions < robot.q_pos_min - limit_tolerance).item()) or bool(
        torch.any(positions > robot.q_pos_max + limit_tolerance).item()
    ):
        raise ResultValidationError("Best trajectory exceeds MPD joint position limits.")

    velocity_utilization = 0.0
    if robot.dq_max is not None:
        velocity_utilization = float(torch.amax(torch.abs(velocities) / robot.dq_max).item())
        if velocity_utilization > 1.0 + limit_tolerance:
            raise ResultValidationError(
                f"Best trajectory exceeds MPD velocity limits (utilization={velocity_utilization:.6f})."
            )

    acceleration_utilization = 0.0
    if robot.ddq_max is not None:
        acceleration_utilization = float(torch.amax(torch.abs(accelerations) / robot.ddq_max).item())
        if acceleration_utilization > 1.0 + limit_tolerance:
            raise ResultValidationError(
                "Best trajectory exceeds MPD acceleration limits " f"(utilization={acceleration_utilization:.6f})."
            )

    collisions = planning_task.compute_collision(
        positions,
        margin=planning_task.margin_for_dense_collision_checking,
    )
    if bool(torch.as_tensor(collisions).any().item()):
        raise ResultValidationError("Best trajectory collides in the configured MPD scene.")

    return {
        "start_max_abs_error_rad": start_error,
        "velocity_limit_utilization": velocity_utilization,
        "acceleration_limit_utilization": acceleration_utilization,
    }


def _run_inference(
    request: dict[str, Any],
    request_sha256: str,
    config_path: Path,
    output_dir: Path,
    device_text: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    import torch
    from dotmap import DotMap

    from mpd.inference.inference import GenerativeOptimizationPlanner
    from mpd.metrics.metrics import PlanningMetricsCalculator
    from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml
    from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
    from torch_robotics.robots import RobotPanda
    from torch_robotics.torch_utils.seed import fix_random_seed
    from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

    device = _validate_device(device_text)
    tensor_args = {"device": device, "dtype": torch.float32}
    fix_random_seed(request["seed"])

    try:
        args_inference = DotMap(load_params_from_yaml(config_path))
    except (OSError, TypeError, ValueError) as error:
        raise ConfigurationError(f"Cannot load runtime config {config_path}: {error}") from error
    _validate_runtime_config(args_inference)

    expected_scene_id = _runtime_config_value(
        args_inference,
        "scene_id",
        args_inference.env_id_replace,
    )
    if request["scene_id"] != expected_scene_id:
        raise RequestValidationError(f"scene_id must be {expected_scene_id!r}, got {request['scene_id']!r}.")

    model_dir = _resolve_model_dir(args_inference)
    args_path = model_dir / "args.yaml"
    if not args_path.is_file():
        raise ConfigurationError(f"Model args file not found: {args_path}.")
    args_train = DotMap(load_params_from_yaml(args_path))
    checkpoint_name = f"{'ema_' if args_train.get('use_ema') else ''}model_current.pth"
    checkpoint_path = model_dir / "checkpoints" / checkpoint_name
    if not checkpoint_path.is_file():
        raise ConfigurationError(f"Model checkpoint not found: {checkpoint_path}.")

    args_inference.model_dir = model_dir.as_posix()
    args_train.update(
        **args_inference,
        gripper=True,
        reload_data=False,
        results_dir=output_dir.as_posix(),
        load_indices=True,
        tensor_args=tensor_args,
    )

    planning_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)
    if not isinstance(planning_task.robot, RobotPanda) or planning_task.robot.q_dim != 7:
        raise ConfigurationError("Runtime config must load the 7-DoF RobotPanda model.")
    actual_scene_id = getattr(planning_task.env, "name", type(planning_task.env).__name__)
    if actual_scene_id != expected_scene_id:
        raise ConfigurationError(
            f"Loaded scene {actual_scene_id!r} does not match expected scene {expected_scene_id!r}."
        )

    q_pos_start = to_torch(request["q_pos_start"], **tensor_args)
    q_pos_goal = to_torch(request["q_pos_goal"], **tensor_args)
    q_vel_start = to_torch(request["q_vel_start"], **tensor_args)
    q_vel_goal = to_torch(request["q_vel_goal"], **tensor_args)
    q_acc_start = to_torch(request["q_acc_start"], **tensor_args)
    q_acc_goal = to_torch(request["q_acc_goal"], **tensor_args)
    _validate_robot_state(planning_task, q_pos_start, "q_pos_start")
    _validate_robot_state(planning_task, q_pos_goal, "q_pos_goal")
    _validate_boundary_derivative(q_vel_start, planning_task.robot.dq_max, "q_vel_start")
    _validate_boundary_derivative(q_vel_goal, planning_task.robot.dq_max, "q_vel_goal")
    _validate_boundary_derivative(q_acc_start, planning_task.robot.ddq_max, "q_acc_start")
    _validate_boundary_derivative(q_acc_goal, planning_task.robot.ddq_max, "q_acc_goal")

    ee_pose_goal_full = planning_task.robot.get_EE_pose(q_pos_goal)
    ee_pose_goal = ee_pose_goal_full[0, :3, :4]

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
        q_pos_start,
        q_pos_goal,
        ee_pose_goal,
        q_vel_start=q_vel_start,
        q_vel_goal=q_vel_goal,
        q_acc_start=q_acc_start,
        q_acc_goal=q_acc_goal,
        results_ns=DotMap(t_generator=0.0, t_guide=0.0),
        debug=False,
    )

    valid_trajectory_count = int(results.q_trajs_pos_valid.shape[0])
    if valid_trajectory_count == 0 or results.q_trajs_pos_best is None:
        raise NoValidTrajectoryError("MPD produced no valid trajectory for this request.")

    trajectory_validation = _validate_best_trajectory(
        results,
        planning_task,
        q_pos_start,
        expected_horizon=int(args_inference.num_T_pts),
        expected_duration=float(args_inference.trajectory_duration),
    )
    results.metrics = PlanningMetricsCalculator(planning_task).compute_metrics(results)

    scene_payload = export_isaaclab_scene_payload(planning_task.env, include_boxes=True)
    scene_sha256 = _canonical_json_sha256(scene_payload)
    config_sha256 = _sha256_file(config_path)
    model_args_sha256 = _sha256_file(args_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    git_metadata = _git_metadata()

    metrics_best = results.metrics.trajs_best
    generated_trajectory_count = int(results.q_trajs_pos_iter_0.shape[0])
    collision_trajectory_count = int(results.collision_trajectory_mask.sum().item())
    joint_position_violation_count = int(results.joint_position_violation_mask.sum().item())
    joint_velocity_violation_count = int(results.joint_velocity_violation_mask.sum().item())
    joint_acceleration_violation_count = int(results.joint_acceleration_violation_mask.sum().item())

    result_payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "success",
        "request_id": request["request_id"],
        "joint_names": request["joint_names"],
        "trajectory_file": "trajectory.npz",
        "request": {
            "schema_version": request["schema_version"],
            "sha256": request_sha256,
            "seed": request["seed"],
            "robot_model": request["robot_model"],
            "planning_frame": request["planning_frame"],
            "joint_state_stamp": request["joint_state_stamp"],
        },
        "scene": {
            "scene_id": actual_scene_id,
            "request_scene_hash": request["scene_hash"],
            "mpd_scene_sha256": scene_sha256,
        },
        "model": {
            "model_dir": model_dir.as_posix(),
            "args_path": args_path.as_posix(),
            "args_sha256": model_args_sha256,
            "checkpoint_path": checkpoint_path.as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "config": {
            "path": config_path.as_posix(),
            "sha256": config_sha256,
        },
        "mpd_source": git_metadata,
        "trajectory": {
            "horizon": int(results.q_trajs_pos_best.shape[0]),
            "duration_s": float(results.timesteps[-1].item()),
            "position_unit": "rad",
            "velocity_unit": "rad/s",
            "acceleration_unit": "rad/s^2",
            **trajectory_validation,
        },
        "candidates": {
            "generated": generated_trajectory_count,
            "valid": valid_trajectory_count,
            "colliding": collision_trajectory_count,
            "joint_position_violations": joint_position_violation_count,
            "joint_velocity_violations": joint_velocity_violation_count,
            "joint_acceleration_violations": joint_acceleration_violation_count,
        },
        "best_trajectory_diagnostics": {
            "selection": results.best_trajectory_selection_details,
            "ee_position_error_m": metrics_best.ee_pose_goal_error_position_norm,
            "ee_orientation_error_deg": metrics_best.ee_pose_goal_error_orientation_norm,
            "path_length": metrics_best.path_length,
            "smoothness": metrics_best.smoothness,
        },
        "timing": {
            "inference_total_sec": results.t_inference_total,
            "generator_sec": results.t_generator,
            "guide_sec": results.t_guide,
        },
        "created_unix_time": time.time(),
    }
    trajectory_arrays = {
        "positions": to_numpy(results.q_trajs_pos_best, dtype=np.float64),
        "velocities": to_numpy(results.q_trajs_vel_best, dtype=np.float64),
        "accelerations": to_numpy(results.q_trajs_acc_best, dtype=np.float64),
        "time_from_start": to_numpy(results.timesteps, dtype=np.float64),
        "joint_names": np.asarray(request["joint_names"], dtype=np.str_),
    }
    return _jsonable(result_payload), trajectory_arrays


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one MPD Warehouse request and export trajectory.npz plus result.json. "
            "This command never publishes robot commands and does not run IsaacLab or PyBullet."
        )
    )
    parser.add_argument("--request", required=True, type=Path, help="Input request JSON path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Runtime inference YAML (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device, normally cuda:0.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    request_path = args.request.expanduser().resolve()
    result_path = output_dir / "result.json"
    trajectory_path = output_dir / "trajectory.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path.unlink(missing_ok=True)

    request_id = None
    request_sha256 = None
    try:
        raw_request, request_sha256 = _load_request(request_path)
        request_id_value = raw_request.get("request_id")
        request_id = request_id_value if isinstance(request_id_value, str) else None
        request = validate_request(raw_request)
        request_id = request["request_id"]
        result_payload, trajectory_arrays = _run_inference(
            request,
            request_sha256,
            config_path,
            output_dir,
            args.device,
        )
        _atomic_write_npz(trajectory_path, **trajectory_arrays)
        _atomic_write_json(result_path, result_payload)
        print(result_path.as_posix())
        return 0
    except RuntimeContractError as error:
        trajectory_path.unlink(missing_ok=True)
        failure_payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": error.status,
            "request_id": request_id,
            "trajectory_file": None,
            "request_sha256": request_sha256,
            "error": {"type": type(error).__name__, "message": str(error)},
            "created_unix_time": time.time(),
        }
        _atomic_write_json(result_path, failure_payload)
        print(str(error), file=sys.stderr)
        return error.exit_code
    except Exception as error:
        trajectory_path.unlink(missing_ok=True)
        failure_payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "inference_error",
            "request_id": request_id,
            "trajectory_file": None,
            "request_sha256": request_sha256,
            "error": {"type": type(error).__name__, "message": str(error)},
            "created_unix_time": time.time(),
        }
        _atomic_write_json(result_path, failure_payload)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
