"""Resident static Marvin bimanual MPD engine."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
import sys
from typing import Any, Callable
import uuid

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpd.bimanual.checkpoint_contract import validate_checkpoint_args
from mpd.bimanual.runtime_contract import (
    BimanualRequest,
    ContractError,
    JOINT_NAMES,
    RESULT_SCHEMA,
    validate_result,
)
from scripts.inference.inference_marvin_bimanual import (
    DEFAULT_CONFIG,
    InferenceConfigurationError,
    NoValidTrajectoryError,
    _deadline_guard,
    _jsonable,
    _matrix_to_pose_xyzw,
    _pose_xyzw_to_matrix,
    _resolve_model_dir,
    _runtime_value,
    _sha256_file,
    _sha256_json,
    _validate_runtime_config,
)
from scripts.runtime.infer_once import RuntimeContractError


@dataclass(frozen=True)
class PlanArtifacts:
    result_payload: dict[str, Any]
    trajectory_arrays: dict[str, Any]


class MarvinRuntimeContractError(RuntimeContractError):
    """Expose Marvin validation failures through the shared socket service."""

    status = "invalid_request"


class MarvinBimanualPlanningSession:
    """Own the dataset, task, checkpoint and CUDA planner exactly once."""

    def __init__(
        self,
        config_path: Path,
        runtime_output_root: Path,
        device_text: str,
        state_callback: Callable[[str], None] | None = None,
    ) -> None:
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

        notify = state_callback or (lambda _state: None)
        notify("LOADING")
        self.instance_id = str(uuid.uuid4())
        self.started_unix_ns = time.time_ns()
        self.config_path = Path(config_path).expanduser().resolve()
        self.output_root = Path(runtime_output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if device_text.startswith("cuda") and not torch.cuda.is_available():
            raise InferenceConfigurationError(f"CUDA device {device_text!r} requested but CUDA is unavailable")
        self.device = torch.device(device_text)
        self.tensor_args = {"device": self.device, "dtype": torch.float32}
        raw_config = load_params_from_yaml(self.config_path)
        _validate_runtime_config(raw_config)
        self.raw_config = raw_config
        self.config = DotMap(raw_config)
        self.model_dir = _resolve_model_dir(raw_config)
        self.args_path = self.model_dir / "args.yaml"
        if not self.args_path.is_file():
            raise InferenceConfigurationError(f"Model args file not found: {self.args_path}")
        raw_train = load_params_from_yaml(self.args_path)
        validated_train = validate_checkpoint_args(
            raw_train,
            expected_dataset_subdir=self.config.dataset_subdir,
            expected_variant=self.config.runtime.network_variant,
        )
        try:
            self.checkpoint_path = resolve_model_checkpoint_path(
                self.model_dir,
                bool(validated_train.get("use_ema", False)),
                checkpoint=self.config.get("checkpoint"),
            )
        except (FileNotFoundError, ValueError) as error:
            raise InferenceConfigurationError(str(error)) from error
        dataset_dir = Path(DATASET_BASE_DIR) / self.config.dataset_subdir
        for required in (
            dataset_dir / "args.yaml",
            dataset_dir / self.config.dataset_file_merged,
        ):
            if not required.is_file():
                raise InferenceConfigurationError(f"Inference dataset artifact not found: {required}")
        args_train = DotMap(raw_train)
        args_inference = DotMap(raw_config)
        args_inference.model_dir = self.model_dir.as_posix()
        args_train.update(
            **args_inference,
            gripper=True,
            reload_data=False,
            results_dir=self.output_root.as_posix(),
            load_indices=(self.model_dir / "train_subset_indices.pt").is_file(),
            tensor_args=self.tensor_args,
        )
        self.planning_task, train_subset, _, _, _ = get_planning_task_and_dataset(**args_train)
        self.dataset = train_subset.dataset
        if not isinstance(self.planning_task, BimanualPlanningTask):
            raise InferenceConfigurationError("Loader did not construct BimanualPlanningTask")
        if not isinstance(self.planning_task.robot, RobotMarvinBimanual):
            raise InferenceConfigurationError("Loader did not construct RobotMarvinBimanual")
        self.robot = self.planning_task.robot
        if raw_config.get("task_mode") == "cooperative_rigid":
            # The resident planner builds its cost objects before the first
            # request. Cooperative closure/payload costs therefore need the
            # fixed grasp transforms during session construction, unlike the
            # one-shot path which can install them from its request generator.
            import yaml

            from scripts.generate_data.generate_marvin_warehouse_cooperative import (
                MarvinWarehouseCooperativeGenerator,
                validate_config as validate_cooperative_config,
            )

            generation_path = Path(raw_config["cooperative_generation_config"])
            if not generation_path.is_absolute():
                generation_path = (self.config_path.parent / generation_path).resolve()
            cooperative_config = validate_cooperative_config(
                yaml.safe_load(generation_path.read_text(encoding="utf-8"))
            )
            for section, values in raw_config.get(
                "cooperative_generator_overrides", {}
            ).items():
                if not isinstance(values, dict) or section not in cooperative_config:
                    raise InferenceConfigurationError(
                        f"invalid cooperative_generator_overrides section {section!r}"
                    )
                cooperative_config[section].update(values)
            validate_cooperative_config(cooperative_config)
            bootstrap_generator = MarvinWarehouseCooperativeGenerator(
                cooperative_config,
                0,
                progress_label="cooperative-resident-bootstrap",
            )
            self.planning_task.object_to_left_grasp = torch.as_tensor(
                bootstrap_generator.object_to_left, **self.tensor_args
            )
            self.planning_task.object_to_right_grasp = torch.as_tensor(
                bootstrap_generator.object_to_right, **self.tensor_args
            )
            self.planning_task.set_object_goal(
                torch.eye(4, **self.tensor_args)[:3, :]
            )
            bootstrap_generator.close()
        scene = export_isaaclab_scene_payload(self.planning_task.env, include_boxes=True)
        scene["frame_id"] = "world"
        self.scene = scene
        self.scene_sha256 = _sha256_json(scene)
        self.robot_sha256 = getattr(self.robot, "asset_hash", None)
        self.config_sha256 = _sha256_file(self.config_path)
        self.checkpoint_sha256 = _sha256_file(self.checkpoint_path)
        self.args_sha256 = _sha256_file(self.args_path)
        self.accepted_runtime_modes = {"snapshot_no_time"}
        self.result_postprocessor = None
        self.last_plan_results = None

        notify("WARMING")
        warmup_started = time.perf_counter()
        self.planner = GenerativeOptimizationPlanner(
            self.planning_task,
            train_subset.dataset,
            args_train,
            args_inference,
            self.tensor_args,
            sampling_based_planner_fn=None,
            debug=False,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.warmup_elapsed_sec = time.perf_counter() - warmup_started

    def health(self) -> dict[str, Any]:
        import torch

        dense = self.planner.dense_validation_config
        pruning = self.config.get("gradient_pruning", {})
        memory = int(torch.cuda.memory_allocated(self.device)) if self.device.type == "cuda" else 0
        return {
            "instance_id": self.instance_id,
            "started_unix_ns": self.started_unix_ns,
            "loads": 1,
            "device": str(self.device),
            "warmup_elapsed_sec": self.warmup_elapsed_sec,
            "gpu_memory_allocated_bytes": memory,
            "model_sha256": self.args_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "scene_sha256": self.scene_sha256,
            "robot_sha256": self.robot_sha256,
            "dense_validation_enabled": bool(dense.get("enabled", True)),
            "b3_enabled": bool(pruning.get("enabled", False)),
        }

    def plan(self, request: BimanualRequest) -> PlanArtifacts:
        import numpy as np
        import torch
        from dotmap import DotMap
        from torch_robotics.torch_utils.seed import fix_random_seed

        request_started = time.perf_counter()
        _deadline_guard(request)
        if request.runtime_mode not in self.accepted_runtime_modes:
            expected = ", ".join(sorted(self.accepted_runtime_modes))
            raise ContractError(f"resident worker requires runtime_mode in {{{expected}}}")
        if request.checkpoint_hash and request.checkpoint_hash != self.checkpoint_sha256:
            raise InferenceConfigurationError("request checkpoint_hash does not match checkpoint")
        if request.robot_model_hash and request.robot_model_hash != self.robot_sha256:
            raise InferenceConfigurationError("request robot_model_hash does not match Marvin asset")
        if request.scene_hash and request.scene_hash != self.scene_sha256:
            raise InferenceConfigurationError("request scene_hash does not match Warehouse scene")
        fix_random_seed(request.seed)
        self.planning_task.task_mode = request.task_mode
        self.planning_task.active_joint_mask = torch.ones(14, dtype=torch.bool, device=self.device)
        if request.task_mode == "left_only":
            self.planning_task.active_joint_mask[7:] = False
        elif request.task_mode == "right_only":
            self.planning_task.active_joint_mask[:7] = False
        q_start = torch.as_tensor(request.q_start, **self.tensor_args)
        q_goal = torch.as_tensor(request.q_goal or request.q_start, **self.tensor_args)
        q_velocity_start = torch.as_tensor(request.q_velocity_start, **self.tensor_args)
        q_acceleration_start = torch.as_tensor(request.q_acceleration_start, **self.tensor_args)
        if torch.any(q_start < self.robot.q_pos_min) or torch.any(q_start > self.robot.q_pos_max):
            raise ContractError("q_start violates Marvin joint limits")
        if torch.any(q_goal < self.robot.q_pos_min) or torch.any(q_goal > self.robot.q_pos_max):
            raise ContractError("q_goal violates Marvin joint limits")
        fk_goals = (self.robot.fk_left(q_goal), self.robot.fk_right(q_goal))
        goal_matrices = [
            fk.detach().cpu().numpy() if supplied is None else _pose_xyzw_to_matrix(supplied)
            for supplied, fk in zip((request.left_goal_pose, request.right_goal_pose), fk_goals)
        ]
        ee_goal = torch.as_tensor(np.stack(goal_matrices), **self.tensor_args)
        active_mask = torch.as_tensor(request.active_ee_mask, **self.tensor_args)
        self.planning_task.active_ee_mask = active_mask
        cooperative_adapter = None
        cooperative_generator = None
        plan_kwargs = {}
        if request.task_mode == "cooperative_rigid":
            if self.raw_config.get("task_mode") != "cooperative_rigid":
                raise InferenceConfigurationError("cooperative request requires a cooperative_rigid runtime config")
            import yaml

            from mpd.bimanual.cooperative_prior import CooperativePriorAdapter
            from scripts.generate_data.generate_marvin_warehouse_cooperative import (
                MarvinWarehouseCooperativeGenerator,
                object_transform,
                validate_config as validate_cooperative_config,
            )

            generation_path = Path(self.raw_config["cooperative_generation_config"])
            if not generation_path.is_absolute():
                generation_path = (self.config_path.parent / generation_path).resolve()
            cooperative_config = validate_cooperative_config(
                yaml.safe_load(generation_path.read_text(encoding="utf-8"))
            )
            for section, values in self.raw_config.get("cooperative_generator_overrides", {}).items():
                if not isinstance(values, dict) or section not in cooperative_config:
                    raise InferenceConfigurationError(f"invalid cooperative_generator_overrides section {section!r}")
                cooperative_config[section].update(values)
            validate_cooperative_config(cooperative_config)
            cooperative_generator = MarvinWarehouseCooperativeGenerator(
                cooperative_config,
                request.seed,
                progress_label="cooperative-resident-inference",
            )
            self.planning_task.object_to_left_grasp = torch.as_tensor(
                cooperative_generator.object_to_left, **self.tensor_args
            )
            self.planning_task.object_to_right_grasp = torch.as_tensor(
                cooperative_generator.object_to_right, **self.tensor_args
            )
            if request.object_goal_pose is not None:
                self.planning_task.set_object_goal(
                    torch.as_tensor(
                        _pose_xyzw_to_matrix(request.object_goal_pose),
                        **self.tensor_args,
                    )
                )
            else:
                self.planning_task.set_object_goal(
                    torch.as_tensor(
                        object_transform(
                            np.asarray(request.q_goal, dtype=np.float64),
                            cooperative_generator.base_rotation,
                        )[:3, :],
                        **self.tensor_args,
                    )
                )
            scene_request = dict(request.scene or {})
            source = scene_request.get("object_start_state_xyz_yaw")
            target = scene_request.get("object_goal_state_xyz_yaw")
            if source is None:
                source = cooperative_generator.object_state_from_q(
                    np.asarray(request.q_start, dtype=np.float64)
                ).tolist()
            if target is None and request.q_goal is not None:
                target = cooperative_generator.object_state_from_q(
                    np.asarray(request.q_goal, dtype=np.float64)
                ).tolist()
            if target is None and request.object_goal_pose is not None:
                x, y, z, qx, qy, qz, qw = request.object_goal_pose
                yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
                target = [x, y, z, yaw]
            if target is None:
                raise ContractError("cooperative request has no resolvable object goal")
            self.planning_task.set_q_pos_start_goal(q_start, q_goal)
            self.planning_task.parametric_trajectory.set_boundary_conditions(
                q_pos_start=q_start,
                q_pos_goal=q_goal,
                q_vel_start=q_velocity_start,
                q_vel_goal=torch.zeros_like(q_goal),
                q_acc_start=q_acceleration_start,
                q_acc_goal=torch.zeros_like(q_goal),
            )
            cooperative_adapter = CooperativePriorAdapter(
                cooperative_generator,
                self.planning_task,
                self.dataset,
                self.raw_config["cooperative_inference"].get("projection", {}),
            )
            cooperative_adapter.set_task(source, target, request.q_start, request.q_goal)
            options = self.raw_config["cooperative_inference"]
            if options["prior_mode"] == "reference_residual":
                plan_kwargs["initial_control_points_normalized"] = cooperative_adapter.build_reference_normalized()
                fraction = float(options.get("reference_noise_fraction", 0.4))
                if not 0.0 < fraction <= 1.0:
                    raise InferenceConfigurationError("reference_noise_fraction must lie in (0, 1]")
                plan_kwargs["initial_noise_timestep"] = int(
                    round(fraction * (self.planner.model.n_diffusion_steps - 1))
                )
            if bool(options.get("closed_chain_projection", True)):
                plan_kwargs["control_point_postprocessor"] = cooperative_adapter.project_normalized
        else:
            self.planning_task.object_goal_pose = None
        started = time.perf_counter()
        setup_elapsed = started - request_started
        results = self.planner.plan_trajectory(
            q_start,
            q_goal,
            ee_goal,
            active_ee_mask=active_mask,
            q_vel_start=q_velocity_start,
            q_acc_start=q_acceleration_start,
            results_ns=DotMap(t_generator=0.0, t_guide=0.0),
            debug=False,
            **plan_kwargs,
        )
        if self.result_postprocessor is not None:
            results = self.result_postprocessor(results)
        self.last_plan_results = results
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        selection_started = time.perf_counter()
        _deadline_guard(request)
        generic_valid_indices = torch.nonzero(results.valid_trajectory_mask).flatten()
        cooperative_audits = {}
        if cooperative_adapter is None:
            valid_indices = generic_valid_indices
        else:
            cooperative_valid = []
            for candidate_index in generic_valid_indices.tolist():
                audit = cooperative_adapter.audit_path(
                    results.q_trajs_pos_iter_0[candidate_index].detach().cpu().numpy()
                )
                cooperative_audits[candidate_index] = audit
                if audit.valid:
                    cooperative_valid.append(candidate_index)
            valid_indices = torch.as_tensor(cooperative_valid, dtype=torch.long, device=self.device)
        if valid_indices.numel() == 0:
            raise NoValidTrajectoryError("MPD produced no dense-valid trajectory")
        scores = results.valid_trajectory_selection_scores
        original_valid = generic_valid_indices.tolist()
        score_by_candidate = (
            {} if scores is None else {index: scores[offset] for offset, index in enumerate(original_valid)}
        )
        ordered_values = sorted(
            valid_indices.tolist(),
            key=lambda index: float(score_by_candidate.get(index, 0.0)),
        )
        selected_index = int(ordered_values[0])
        best_positions = results.q_trajs_pos_iter_0[selected_index]
        best_velocities = results.q_trajs_vel_iter_0[selected_index]
        best_accelerations = results.q_trajs_acc_iter_0[selected_index]
        # DotMap materializes missing attributes as nested DotMaps, so use
        # mapping lookup to distinguish fixed-time from candidate timing.
        candidate_times = results.get("candidate_timesteps", None)
        selected_times = candidate_times[selected_index] if candidate_times is not None else results.timesteps
        report = self.planner.dense_validator.validate(
            q_position=best_positions.unsqueeze(0),
            q_velocity=best_velocities.unsqueeze(0),
            q_acceleration=best_accelerations.unsqueeze(0),
            trajectory_times=(selected_times.unsqueeze(0) if candidate_times is not None else None),
            num_points=int(self.config.dense_validation.runtime_points),
            check_environment=True,
            check_self_collision=True,
            check_joint_position=True,
            check_joint_velocity=True,
            check_joint_acceleration=True,
        )
        if not bool(report.trajectory_valid_mask[0].item()):
            raise NoValidTrajectoryError(f"Selected trajectory failed final oracle: {report.failure_codes[0]}")
        ordered = torch.as_tensor(ordered_values, dtype=torch.long, device=self.device)
        top_count = min(int(self.config.runtime_top_k_valid_trajectories), ordered.numel())
        top_indices = ordered[:top_count]
        top_scores = torch.stack(
            [score_by_candidate.get(int(index), torch.tensor(0.0, **self.tensor_args)) for index in top_indices]
        )
        top_positions = results.q_trajs_pos_iter_0.index_select(0, top_indices)
        top_velocities = results.q_trajs_vel_iter_0.index_select(0, top_indices)
        top_accelerations = results.q_trajs_acc_iter_0.index_select(0, top_indices)
        times = selected_times
        validation = {
            "valid": True,
            "failure_code": None,
            "left_ee_position_error_m": report.ee_position_error_m[0, 0],
            "right_ee_position_error_m": report.ee_position_error_m[0, 1],
            "left_ee_orientation_error_rad": report.ee_orientation_error_rad[0, 0],
            "right_ee_orientation_error_rad": report.ee_orientation_error_rad[0, 1],
            "minimum_environment_clearance_m": report.minimum_environment_clearance[0],
            "minimum_left_self_clearance_m": report.minimum_left_self_clearance[0],
            "minimum_right_self_clearance_m": report.minimum_right_self_clearance[0],
            "minimum_interarm_clearance_m": report.minimum_interarm_clearance[0],
        }
        if cooperative_adapter is not None:
            selected_audit = cooperative_audits[selected_index]
            validation.update(
                max_closure_translation_error_m=(selected_audit.max_closure_translation_m),
                max_closure_rotation_error_rad=(selected_audit.max_closure_rotation_rad),
                cooperative_points_checked=selected_audit.checked_points,
                payload_collision=False,
            )
        result = validate_result(
            _jsonable(
                {
                    "schema": RESULT_SCHEMA,
                    "request_id": request.request_id,
                    "status": "success",
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
                        "object_pose_xyzw": (
                            None if request.object_goal_pose is None else list(request.object_goal_pose)
                        ),
                    },
                    "model": {
                        "checkpoint_sha256": self.checkpoint_sha256,
                        "args_sha256": self.args_sha256,
                    },
                    "scene": {
                        "scene_sha256": self.scene_sha256,
                        "robot_asset_sha256": self.robot_sha256,
                    },
                    "resident_runtime": {
                        "instance_id": self.instance_id,
                        "request_elapsed_s": elapsed,
                    },
                    "candidates": {
                        "generated": int(results.q_trajs_pos_iter_0.shape[0]),
                        "dense_checked": int(results.dense_validation_candidates_checked),
                        "dense_complete": bool(results.dense_validation_complete),
                    },
                }
            ),
            request=request,
        )
        arrays = {
            "positions": best_positions.detach().cpu().numpy().astype(np.float64),
            "velocities": best_velocities.detach().cpu().numpy().astype(np.float64),
            "accelerations": best_accelerations.detach().cpu().numpy().astype(np.float64),
            "time_from_start": times.detach().cpu().numpy().astype(np.float64),
            "top_k_positions": top_positions.detach().cpu().numpy().astype(np.float64),
            "top_k_velocities": top_velocities.detach().cpu().numpy().astype(np.float64),
            "top_k_accelerations": top_accelerations.detach().cpu().numpy().astype(np.float64),
            "top_k_candidate_indices": top_indices.detach().cpu().numpy().astype(np.int64),
            "top_k_scores": top_scores.detach().cpu().numpy().astype(np.float64),
            "joint_names": np.asarray(JOINT_NAMES, dtype=np.str_),
            "ee_goal_pose": ee_goal.detach().cpu().numpy().astype(np.float64),
            "active_ee_mask": active_mask.detach().cpu().numpy().astype(np.float64),
            "mpd_tcp_pose_start": np.stack(
                (
                    self.robot.fk_left(q_start).detach().cpu().numpy(),
                    self.robot.fk_right(q_start).detach().cpu().numpy(),
                )
            ).astype(np.float64),
            "mpd_top_k_tcp_pose_final": torch.stack(
                (
                    self.robot.fk_left(top_positions[:, -1]),
                    self.robot.fk_right(top_positions[:, -1]),
                ),
                dim=1,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64),
        }
        if cooperative_adapter is not None:
            from scripts.generate_data.generate_marvin_warehouse_cooperative import (
                object_transform,
            )

            top_numpy = top_positions.detach().cpu().numpy()
            top_object_states = np.stack(
                [
                    np.stack([cooperative_generator.object_state_from_q(state) for state in trajectory])
                    for trajectory in top_numpy
                ]
            ).astype(np.float64)
            top_object_poses = np.stack(
                [
                    np.stack(
                        [
                            _matrix_to_pose_xyzw(object_transform(state, cooperative_generator.base_rotation))
                            for state in trajectory
                        ]
                    )
                    for trajectory in top_object_states
                ]
            ).astype(np.float64)
            arrays.update(
                object_path=top_object_states[0],
                top_k_object_path=top_object_states,
                object_path_pose_xyzw=top_object_poses[0],
                top_k_object_path_pose_xyzw=top_object_poses,
                payload_size_xyz=np.asarray(cooperative_generator.payload_size, dtype=np.float64),
            )
            cooperative_generator.close()
        result["resident_runtime"]["timing_s"] = {
            "request_setup": setup_elapsed,
            "mpd_plan_and_dense": elapsed,
            "candidate_selection_and_artifact": time.perf_counter() - selection_started,
        }
        return PlanArtifacts(result_payload=result, trajectory_arrays=arrays)


class MarvinBimanualRuntimeEngine:
    """Public engine wrapper with one immutable loaded planning session."""

    def __init__(
        self,
        config_path: Path = DEFAULT_CONFIG,
        runtime_output_root: Path = Path("/tmp/mpd-marvin-bimanual-resident"),
        device_text: str = "cuda:0",
        state_callback: Callable[[str], None] | None = None,
        session_factory=MarvinBimanualPlanningSession,
    ) -> None:
        self._session = session_factory(config_path, runtime_output_root, device_text, state_callback)
        self.instance_id = self._session.instance_id

    def health(self):
        return self._session.health()

    @property
    def scene_payload(self) -> dict[str, Any]:
        """Return the immutable scene used by both resident and one-shot paths."""
        return self._session.scene

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        try:
            return self._session.plan(BimanualRequest.from_dict(raw_request))
        except (ContractError, InferenceConfigurationError) as error:
            raise MarvinRuntimeContractError(str(error)) from error
