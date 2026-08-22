"""Long-lived MPD runtime engine shared by resident request handlers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable
import uuid

from scripts.runtime.infer_once import (
    ConfigurationError,
    NoValidTrajectoryError,
    RESULT_SCHEMA_VERSION,
    ResultValidationError,
    START_BOUNDARY_TOLERANCE,
    _canonical_json_sha256,
    _cartesian_poses_xyzw,
    _git_metadata,
    _jsonable,
    _pose_xyzw_to_transform,
    _resolve_model_dir,
    _runtime_config_value,
    _sha256_file,
    _solve_cartesian_goal,
    _validate_best_trajectory,
    _validate_boundary_derivative,
    _validate_device,
    _validate_robot_state,
    _validate_runtime_config,
    validate_request,
)


@dataclass(frozen=True)
class PlanArtifacts:
    result_payload: dict[str, Any]
    trajectory_arrays: dict[str, Any]


class MpdRuntimeEngine:
    """Own one planning task and one GPU planner for the process lifetime."""

    def __init__(
        self,
        config_path: Path,
        runtime_output_root: Path,
        device_text: str = "cuda:0",
        state_callback: Callable[[str], None] | None = None,
    ) -> None:
        import torch
        from dotmap import DotMap

        from mpd.inference.inference import GenerativeOptimizationPlanner, resolve_model_checkpoint_path
        from mpd.utils.loaders import get_planning_task_and_dataset, load_params_from_yaml
        from scripts.isaaclab.scene_payload import export_isaaclab_scene_payload
        from torch_robotics.robots import RobotPanda

        self.instance_id = str(uuid.uuid4())
        self.started_unix_time = time.time()
        self.config_path = Path(config_path).expanduser().resolve()
        self.runtime_output_root = Path(runtime_output_root).expanduser().resolve()
        self.runtime_output_root.mkdir(parents=True, exist_ok=True)
        self.device = _validate_device(device_text)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}
        notify = state_callback or (lambda _state: None)

        notify("LOADING")
        try:
            self.args_inference = DotMap(load_params_from_yaml(self.config_path))
        except (OSError, TypeError, ValueError) as error:
            raise ConfigurationError(f"Cannot load runtime config {self.config_path}: {error}") from error
        _validate_runtime_config(self.args_inference)

        self.expected_scene_id = _runtime_config_value(
            self.args_inference,
            "scene_id",
            self.args_inference.env_id_replace,
        )
        self.model_dir = _resolve_model_dir(self.args_inference)
        self.args_path = self.model_dir / "args.yaml"
        if not self.args_path.is_file():
            raise ConfigurationError(f"Model args file not found: {self.args_path}.")
        self.args_train = DotMap(load_params_from_yaml(self.args_path))
        try:
            self.checkpoint_path = resolve_model_checkpoint_path(
                self.model_dir,
                self.args_train.get("use_ema"),
                checkpoint=_runtime_config_value(self.args_inference, "checkpoint", None),
            )
        except (FileNotFoundError, ValueError) as error:
            raise ConfigurationError(str(error)) from error

        self.args_inference.model_dir = self.model_dir.as_posix()
        self.args_train.update(
            **self.args_inference,
            gripper=True,
            reload_data=False,
            results_dir=self.runtime_output_root.as_posix(),
            load_indices=True,
            tensor_args=self.tensor_args,
        )

        self.planning_task, train_subset, _, _, _ = get_planning_task_and_dataset(**self.args_train)
        if not isinstance(self.planning_task.robot, RobotPanda) or self.planning_task.robot.q_dim != 7:
            raise ConfigurationError("Runtime config must load the configured 7-DoF planning backend.")
        self.actual_scene_id = getattr(
            self.planning_task.env,
            "name",
            type(self.planning_task.env).__name__,
        )
        if self.actual_scene_id != self.expected_scene_id:
            raise ConfigurationError(
                f"Loaded scene {self.actual_scene_id!r} does not match " f"expected scene {self.expected_scene_id!r}."
            )

        notify("WARMING")
        warmup_started = time.perf_counter()
        self.planner = GenerativeOptimizationPlanner(
            self.planning_task,
            train_subset.dataset,
            self.args_train,
            self.args_inference,
            self.tensor_args,
            sampling_based_planner_fn=None,
            debug=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.warmup_elapsed_sec = time.perf_counter() - warmup_started

        scene_payload = export_isaaclab_scene_payload(self.planning_task.env, include_boxes=True)
        self.scene_sha256 = _canonical_json_sha256(scene_payload)
        self.config_sha256 = _sha256_file(self.config_path)
        self.model_args_sha256 = _sha256_file(self.args_path)
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        self.git_metadata = _git_metadata()

    def health(self) -> dict[str, Any]:
        dense_cfg = self.planner.dense_validation_config
        ranked_cfg = dense_cfg.get("ranked_early_exit", {})
        return {
            "instance_id": self.instance_id,
            "started_unix_time": self.started_unix_time,
            "warmup_elapsed_sec": self.warmup_elapsed_sec,
            "device": str(self.device),
            "scene_id": self.actual_scene_id,
            "scene_sha256": self.scene_sha256,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dense_validation": {
                "enabled": bool(dense_cfg["enabled"]),
                "runtime_points": int(dense_cfg["runtime_points"]),
                "ranked_batch_buckets": list(ranked_cfg.get("batch_buckets", ())),
                "fully_warmed": bool(dense_cfg["enabled"] and ranked_cfg.get("enabled", False)),
            },
        }

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        import numpy as np
        import torch
        from dotmap import DotMap

        from mpd.metrics.metrics import PlanningMetricsCalculator
        from torch_robotics.torch_utils.seed import fix_random_seed
        from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch

        request = validate_request(raw_request)
        if request["scene_id"] != self.expected_scene_id:
            raise ConfigurationError(f"scene_id must be {self.expected_scene_id!r}, got {request['scene_id']!r}.")
        request_sha256 = _canonical_json_sha256(raw_request)
        fix_random_seed(request["seed"])

        q_pos_start = to_torch(request["q_pos_start"], **self.tensor_args)
        q_vel_start = to_torch(request["q_vel_start"], **self.tensor_args)
        q_vel_goal = to_torch(request["q_vel_goal"], **self.tensor_args)
        q_acc_start = to_torch(request["q_acc_start"], **self.tensor_args)
        q_acc_goal = to_torch(request["q_acc_goal"], **self.tensor_args)
        _validate_robot_state(self.planning_task, q_pos_start, "q_pos_start")

        ik_metadata = None
        if request["goal_type"] == "joint":
            q_pos_goal = to_torch(request["q_pos_goal"], **self.tensor_args)
            _validate_robot_state(self.planning_task, q_pos_goal, "q_pos_goal")
            ee_pose_goal = self.planning_task.robot.get_EE_pose(q_pos_goal)[0, :3, :4]
            target_pose_xyzw = _cartesian_poses_xyzw(self.planning_task.robot, q_pos_goal)[0]
        else:
            target_transform = _pose_xyzw_to_transform(request["ee_pose_goal"])
            ee_pose_goal = to_torch(target_transform[:3, :4], **self.tensor_args)
            target_pose_xyzw = to_torch(request["ee_pose_goal"], **self.tensor_args)
            q_pos_goal, ik_metadata = _solve_cartesian_goal(
                self.planning_task,
                q_pos_start,
                ee_pose_goal,
                ik_candidates=request["ik_candidates"],
                ik_max_iters=request["ik_max_iters"],
            )
            _validate_robot_state(self.planning_task, q_pos_goal, "cartesian_goal_ik_condition")

        _validate_boundary_derivative(q_vel_start, self.planning_task.robot.dq_max, "q_vel_start")
        _validate_boundary_derivative(q_vel_goal, self.planning_task.robot.dq_max, "q_vel_goal")
        _validate_boundary_derivative(q_acc_start, self.planning_task.robot.ddq_max, "q_acc_start")
        _validate_boundary_derivative(q_acc_goal, self.planning_task.robot.ddq_max, "q_acc_goal")

        request_started = time.perf_counter()
        results = self.planner.plan_trajectory(
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
        # Phase-5 keeps the planner result device-resident long enough for its
        # separate runtime subclass to attach candidate-specific timing.  The
        # Phase-3/4 artifact path below is unchanged.
        self._last_plan_results = results
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        request_elapsed_sec = time.perf_counter() - request_started

        valid_trajectory_count = int(results.valid_trajectory_mask.sum().item())
        if valid_trajectory_count == 0 or results.q_trajs_pos_best is None:
            raise NoValidTrajectoryError("MPD produced no valid trajectory for this request.")

        trajectory_validation = _validate_best_trajectory(
            results,
            self.planning_task,
            q_pos_start,
            q_vel_start,
            q_acc_start,
            expected_horizon=int(self.args_inference.num_T_pts),
            expected_duration=float(self.args_inference.trajectory_duration),
        )
        results.metrics = PlanningMetricsCalculator(self.planning_task).compute_metrics(results)

        selection_scores = results.valid_trajectory_selection_scores
        if selection_scores is None or selection_scores.ndim != 1:
            raise ResultValidationError("MPD did not expose one selection score per valid trajectory.")
        if len(selection_scores) != len(results.q_trajs_pos_valid):
            raise ResultValidationError("MPD valid trajectory scores and arrays have inconsistent lengths.")
        if not bool(torch.isfinite(selection_scores).all().item()):
            raise ResultValidationError("MPD valid trajectory scores contain NaN or Inf.")
        requested_top_k = int(
            _runtime_config_value(self.args_inference, "runtime_top_k_valid_trajectories", 8)
        )
        if requested_top_k <= 0:
            raise ConfigurationError("runtime_top_k_valid_trajectories must be positive.")
        top_k_count = min(requested_top_k, len(selection_scores))
        top_k_order = torch.argsort(selection_scores, stable=True)[:top_k_count]
        top_k_positions = results.q_trajs_pos_valid.index_select(0, top_k_order)
        top_k_velocities = results.q_trajs_vel_valid.index_select(0, top_k_order)
        top_k_accelerations = results.q_trajs_acc_valid.index_select(0, top_k_order)
        top_k_scores = selection_scores.index_select(0, top_k_order)
        top_k_source_indices = torch.nonzero(
            results.valid_trajectory_mask, as_tuple=False
        ).flatten().index_select(0, top_k_order)
        top_k_boundary_errors = {
            "position": torch.amax(torch.abs(top_k_positions[:, 0] - q_pos_start)),
            "velocity": torch.amax(torch.abs(top_k_velocities[:, 0] - q_vel_start)),
            "acceleration": torch.amax(torch.abs(top_k_accelerations[:, 0] - q_acc_start)),
        }
        boundary_tolerance = START_BOUNDARY_TOLERANCE
        for derivative, error in top_k_boundary_errors.items():
            if float(error.item()) > boundary_tolerance:
                raise ResultValidationError(
                    f"Top-K {derivative} start boundary error {error.item():.6g} exceeds "
                    f"{boundary_tolerance:.6g}."
                )

        metrics_best = results.metrics.trajs_best
        generated_count = int(results.q_trajs_pos_iter_0.shape[0])
        dense_checked_count = (
            int(results.dense_validation_candidates_checked)
            if results.dense_validation_candidates_checked is not None
            else generated_count
        )
        terminal_pose = _cartesian_poses_xyzw(
            self.planning_task.robot,
            results.q_trajs_pos_best[-1],
        )[0]

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
            "goal": {
                "input_type": request["goal_type"],
                "target_pose_xyzw": target_pose_xyzw,
                "q_pos_goal_condition": q_pos_goal,
                "ik": ik_metadata,
                "end_effector_frame": "fr3_hand",
            },
            "scene": {
                "scene_id": self.actual_scene_id,
                "request_scene_hash": request["scene_hash"],
                "mpd_scene_sha256": self.scene_sha256,
            },
            "model": {
                "model_dir": self.model_dir.as_posix(),
                "args_path": self.args_path.as_posix(),
                "args_sha256": self.model_args_sha256,
                "checkpoint_path": self.checkpoint_path.as_posix(),
                "checkpoint_sha256": self.checkpoint_sha256,
            },
            "config": {
                "path": self.config_path.as_posix(),
                "sha256": self.config_sha256,
            },
            "mpd_source": self.git_metadata,
            "resident_runtime": {
                "instance_id": self.instance_id,
                "started_unix_time": self.started_unix_time,
                "warmup_elapsed_sec": self.warmup_elapsed_sec,
            },
            "trajectory": {
                "horizon": int(results.q_trajs_pos_best.shape[0]),
                "duration_s": float(results.timesteps[-1].item()),
                "position_unit": "rad",
                "velocity_unit": "rad/s",
                "acceleration_unit": "rad/s^2",
                "cartesian_pose_format": "[x, y, z, qx, qy, qz, qw]",
                "cartesian_position_unit": "m",
                "cartesian_reference_frame": request["planning_frame"],
                **trajectory_validation,
            },
            "candidates": {
                "generated": generated_count,
                "dense_checked": dense_checked_count,
                "dense_unchecked": generated_count - dense_checked_count,
                "dense_complete": bool(results.dense_validation_complete),
                "dense_batches_evaluated": int(results.dense_validation_batches_evaluated),
                "dense_bucket_capacities": results.dense_validation_bucket_capacities,
                "dense_padding_slots": int(results.dense_validation_padding_slots),
                "valid": valid_trajectory_count,
                "returned_top_k": top_k_count,
                "requested_top_k": requested_top_k,
                "colliding": int(results.collision_trajectory_mask.sum().item()),
                "joint_position_violations": int(results.joint_position_violation_mask.sum().item()),
                "joint_velocity_violations": int(results.joint_velocity_violation_mask.sum().item()),
                "joint_acceleration_violations": int(results.joint_acceleration_violation_mask.sum().item()),
            },
            "best_trajectory_diagnostics": {
                "selection": results.best_trajectory_selection_details,
                "ee_position_error_m": metrics_best.ee_pose_goal_error_position_norm,
                "ee_orientation_error_deg": metrics_best.ee_pose_goal_error_orientation_norm,
                "path_length": metrics_best.path_length,
                "smoothness": metrics_best.smoothness,
                "terminal_cartesian_pose_xyzw": terminal_pose,
            },
            "top_k_trajectory_diagnostics": {
                "selection_method": results.best_trajectory_selection_details["method"],
                "scores": top_k_scores,
                "source_candidate_indices": top_k_source_indices,
                "start_position_max_abs_error_rad": top_k_boundary_errors["position"],
                "start_velocity_max_abs_error_rad_s": top_k_boundary_errors["velocity"],
                "start_acceleration_max_abs_error_rad_s2": top_k_boundary_errors["acceleration"],
            },
            "timing": {
                "request_total_sec": request_elapsed_sec,
                "inference_total_sec": results.t_inference_total,
                "generator_sec": results.t_generator,
                "guide_sec": results.t_guide,
                "trajectory_ranking_sec": results.trajectory_ranking_time,
                "dense_validation_sec": results.dense_validation_time,
            },
            "created_unix_time": time.time(),
        }
        trajectory_arrays = {
            "positions": to_numpy(results.q_trajs_pos_best, dtype=np.float64),
            "velocities": to_numpy(results.q_trajs_vel_best, dtype=np.float64),
            "accelerations": to_numpy(results.q_trajs_acc_best, dtype=np.float64),
            "time_from_start": to_numpy(results.timesteps, dtype=np.float64),
            "joint_names": np.asarray(request["joint_names"], dtype=np.str_),
            "terminal_cartesian_pose_xyzw": to_numpy(terminal_pose, dtype=np.float64),
            "topk_positions": to_numpy(top_k_positions, dtype=np.float64),
            "topk_velocities": to_numpy(top_k_velocities, dtype=np.float64),
            "topk_accelerations": to_numpy(top_k_accelerations, dtype=np.float64),
            "topk_scores": to_numpy(top_k_scores, dtype=np.float64),
            "topk_source_candidate_indices": to_numpy(top_k_source_indices, dtype=np.int64),
        }
        return PlanArtifacts(
            result_payload=_jsonable(result_payload),
            trajectory_arrays=trajectory_arrays,
        )
