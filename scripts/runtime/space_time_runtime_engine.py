"""Separate Phase-5 inference-only space-time MPD runtime entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mpd.inference.space_time_guidance import (
    InferenceOnlySpaceTimeGuide,
    SpaceTimeGuidanceSettings,
)
from scripts.runtime.dynamic_runtime_engine import DynamicMpdRuntimeEngine
from scripts.runtime.infer_once import _validate_best_trajectory
from scripts.runtime.runtime_engine import PlanArtifacts
from scripts.runtime.timing_contract import attach_candidate_timing


class SpaceTimeMpdRuntimeEngine(DynamicMpdRuntimeEngine):
    """Phase-5 worker; Phase-4's engine and server remain untouched."""

    def __init__(
        self,
        config_path: Path,
        runtime_output_root: Path,
        device_text: str = "cuda:0",
        state_callback: Callable[[str], None] | None = None,
        *,
        timing_mode: str = "phase5_joint",
        space_time_settings: dict[str, Any] | None = None,
        max_dynamic_objects: int = 16,
        covariance_sigma: float = 3.0,
        process_acceleration_std_m_s2: float = 0.01,
        static_spatial_pruning_enabled: bool = True,
        dynamic_space_time_pruning_enabled: bool = False,
    ) -> None:
        if dynamic_space_time_pruning_enabled:
            raise ValueError(
                "candidate-specific dynamic space-time pruning is not implemented"
            )
        settings = SpaceTimeGuidanceSettings.from_mapping(
            space_time_settings, mode=timing_mode
        )
        super().__init__(
            config_path=config_path,
            runtime_output_root=runtime_output_root,
            device_text=device_text,
            state_callback=state_callback,
            max_dynamic_objects=max_dynamic_objects,
            covariance_sigma=covariance_sigma,
            process_acceleration_std_m_s2=process_acceleration_std_m_s2,
            # Candidate-specific timing uses the uncached table path, while
            # object-capacity and shape reductions remain algebraically exact.
            capacity_buckets_enabled=True,
            shape_grouping_enabled=True,
            time_table_cache_enabled=False,
            fused_reduction_enabled=True,
            dynamic_guide_pruning_enabled=False,
            trajectory_schema_version=2,
            collision_spheres_float32=True,
            deduplicate_best_trajectory=True,
        )
        if self.planner.cost_guide is None:
            raise ValueError("Phase-5 runtime requires the configured MPD cost guide")
        spatial_guide = self.planner.cost_guide
        if static_spatial_pruning_enabled and not spatial_guide.gradient_pruning_config[
            "enabled"
        ]:
            raise ValueError(
                "static spatial pruning was requested but is disabled in the MPD config"
            )
        spatial_guide.gradient_pruning_enabled = bool(
            static_spatial_pruning_enabled
        )
        self.static_spatial_pruning_enabled = bool(
            static_spatial_pruning_enabled
        )
        self.dynamic_space_time_pruning_enabled = False
        self.space_time_settings = settings
        self.space_time_guide = InferenceOnlySpaceTimeGuide(
            spatial_guide,
            self.planning_task,
            self.planner.dataset,
            self.dynamic_field,
            settings,
            self.tensor_args,
        )
        self.planner.cost_guide = self.space_time_guide
        if self.static_spatial_pruning_enabled:
            import torch

            template = torch.zeros(
                (
                    int(self.args_inference.n_trajectory_samples),
                    *self.planner.dataset.control_points_dim,
                ),
                **self.tensor_args,
            )
            self.space_time_guide._spatial_descent(template, warmup=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def _postprocess_plan_results(self, results):
        import torch

        control_points = results.control_points_iter_0
        normalized = self.planner.dataset.normalize_control_points(control_points)
        timing_control_points = self.space_time_guide.timing_control_points
        with torch.no_grad():
            candidate_cost, _, timing = self.space_time_guide.evaluate_control_points(
                normalized, timing_control_points
            )
            dense_cfg = self.planner.dense_validation_config
            if int(dense_cfg["runtime_points"]) != int(timing.q.shape[1]):
                raise ValueError(
                    "Phase-5 full DenseCheck resolution must match the timing spline grid"
                )
            dense = self.planner.dense_validator.validate(
                control_points=None,
                num_points=int(dense_cfg["runtime_points"]),
                q_position=timing.q,
                q_velocity=timing.dq,
                q_acceleration=timing.ddq,
                trajectory_times=timing.time_from_start,
                check_environment=dense_cfg["check_environment"],
                check_self_collision=dense_cfg["check_self_collision"],
                check_joint_position=dense_cfg["check_joint_position"],
                check_joint_velocity=dense_cfg["check_joint_velocity"],
                check_joint_acceleration=dense_cfg["check_joint_acceleration"],
            )

        duration_valid = (
            (timing.duration >= self.space_time_settings.duration_min)
            & (timing.duration <= self.space_time_settings.duration_max)
        )
        finite = (
            torch.isfinite(timing.q).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.dq).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.ddq).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.time_from_start).all(dim=-1)
        )
        valid_mask = dense.trajectory_valid_mask & duration_valid & finite
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        collision_waypoint_mask = (
            dense.environment_collision_mask | dense.self_collision_mask
        )
        collision_trajectory_mask = collision_waypoint_mask.any(dim=-1)
        first_collision = collision_waypoint_mask.long().argmax(dim=-1)
        first_collision = torch.where(
            collision_trajectory_mask,
            first_collision,
            torch.full_like(first_collision, -1),
        )

        results.update(
            q_trajs_pos_iter_0=timing.q,
            q_trajs_vel_iter_0=timing.dq,
            q_trajs_acc_iter_0=timing.ddq,
            collision_waypoint_mask=collision_waypoint_mask,
            collision_trajectory_mask=collision_trajectory_mask,
            first_collision_steps=first_collision,
            joint_position_violation_mask=dense.joint_position_violation_mask,
            joint_velocity_violation_mask=dense.joint_velocity_violation_mask,
            joint_acceleration_violation_mask=dense.joint_acceleration_violation_mask,
            valid_trajectory_mask=valid_mask,
            dense_validation_enabled=True,
            dense_validation_points=int(timing.q.shape[1]),
            dense_validation_checked_mask=dense.trajectory_checked_mask,
            dense_validation_candidates_checked=int(control_points.shape[0]),
            dense_validation_batches_evaluated=1,
            dense_validation_bucket_capacities=[int(control_points.shape[0])],
            dense_validation_padding_slots=0,
            dense_validation_complete=True,
            dense_validation_ranked_early_exit=False,
            dense_environment_collision_mask=dense.environment_collision_mask,
            dense_self_collision_mask=dense.self_collision_mask,
            dense_minimum_environment_clearance=dense.minimum_environment_clearance,
            dense_minimum_self_clearance=dense.minimum_self_clearance,
            dense_first_invalid_index=dense.first_invalid_index,
            candidate_timesteps=timing.time_from_start,
            timing_control_points=timing_control_points,
            candidate_durations=timing.duration,
        )
        if not valid_indices.numel():
            empty = timing.q[:0]
            results.update(
                control_points_valid=control_points[:0],
                q_trajs_pos_valid=empty,
                q_trajs_vel_valid=timing.dq[:0],
                q_trajs_acc_valid=timing.ddq[:0],
                valid_candidate_timesteps=timing.time_from_start[:0],
                valid_timing_control_points=timing_control_points[:0],
                valid_trajectory_selection_scores=candidate_cost[:0],
                control_points_best=None,
                q_trajs_pos_best=None,
                q_trajs_vel_best=None,
                q_trajs_acc_best=None,
                best_trajectory_selection_details={
                    "method": "phase5_space_time_cost",
                    "reason": "no_full_candidate_specific_dense_valid_trajectory",
                },
                timesteps=None,
            )
            return results

        valid_cost = candidate_cost.index_select(0, valid_indices)
        selected_valid = int(torch.argmin(valid_cost).item())
        selected_candidate = int(valid_indices[selected_valid].item())
        results.update(
            control_points_valid=control_points.index_select(0, valid_indices),
            q_trajs_pos_valid=timing.q.index_select(0, valid_indices),
            q_trajs_vel_valid=timing.dq.index_select(0, valid_indices),
            q_trajs_acc_valid=timing.ddq.index_select(0, valid_indices),
            valid_candidate_timesteps=timing.time_from_start.index_select(
                0, valid_indices
            ),
            valid_timing_control_points=timing_control_points.index_select(
                0, valid_indices
            ),
            valid_trajectory_selection_scores=valid_cost,
            control_points_best=control_points[selected_candidate],
            q_trajs_pos_best=timing.q[selected_candidate],
            q_trajs_vel_best=timing.dq[selected_candidate],
            q_trajs_acc_best=timing.ddq[selected_candidate],
            best_trajectory_selection_details={
                "method": "phase5_space_time_cost",
                "selected_candidate_index": selected_candidate,
                "selected_valid_index": selected_valid,
                "score": float(valid_cost[selected_valid].item()),
                "duration_s": float(timing.duration[selected_candidate].item()),
            },
            timesteps=timing.time_from_start[selected_candidate],
        )
        return results

    def _validate_plan_results(
        self,
        results,
        q_pos_start,
        q_vel_start,
        q_acc_start,
    ):
        validation = _validate_best_trajectory(
            results,
            self.planning_task,
            q_pos_start,
            q_vel_start,
            q_acc_start,
            expected_horizon=int(self.args_inference.num_T_pts),
            expected_duration=None,
            skip_collision_check=True,
        )
        if not bool(results.dense_validation_complete):
            raise ValueError("Phase-5 full candidate-specific DenseCheck is incomplete")
        selected = int(results.best_trajectory_selection_details["selected_candidate_index"])
        if not bool(results.valid_trajectory_mask[selected].item()):
            raise ValueError("Phase-5 selected trajectory did not pass its full DenseCheck")
        validation.update(
            duration_s=float(results.timesteps[-1].item()),
            minimum_environment_clearance_m=float(
                results.dense_minimum_environment_clearance[selected].item()
            ),
            minimum_self_clearance_m=float(
                results.dense_minimum_self_clearance[selected].item()
            ),
        )
        return validation

    def health(self) -> dict[str, Any]:
        response = super().health()
        response["dynamic_world"]["trajectory_schema_version"] = 3
        response["space_time"] = {
            "enabled": True,
            "mode": self.space_time_settings.mode,
            "timing_schema_version": 1,
            "trajectory_schema_version": 3,
            "candidate_specific_time": True,
            "duration_min_s": self.space_time_settings.duration_min,
            "duration_max_s": self.space_time_settings.duration_max,
            "phase4_entry_unchanged": True,
            "pruning": {
                "static_spatial": self.static_spatial_pruning_enabled,
                "dynamic_space_time": self.dynamic_space_time_pruning_enabled,
            },
            "reuse_spatial_kinematics": (
                self.space_time_guide.reuse_spatial_kinematics_enabled
            ),
        }
        return response

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        import numpy as np
        import torch
        from torch_robotics.torch_kinematics_tree.geometrics.utils import (
            link_pos_from_link_tensor,
        )
        from torch_robotics.torch_utils.torch_utils import to_numpy

        self.space_time_guide.reset(int(self.args_inference.n_trajectory_samples))
        artifacts = super().plan(raw_request)
        results = self._last_plan_results
        source_indices = torch.as_tensor(
            artifacts.trajectory_arrays["topk_source_candidate_indices"],
            dtype=torch.long,
            device=self.device,
        )
        timing_control_points = self.space_time_guide.timing_control_points.index_select(
            0, source_indices
        )
        topk_times = results.candidate_timesteps.index_select(0, source_indices)

        # Re-evaluate spatial derivatives with the selected timing state.  This
        # does not infer velocities from sampled positions.
        normalized = self.planner.dataset.normalize_control_points(
            results.control_points_iter_0.index_select(0, source_indices)
        )
        _, _, evaluation = self.space_time_guide.evaluate_control_points(
            normalized, timing_control_points
        )
        if not all(
            torch.isfinite(value).all().item()
            for value in (evaluation.q, evaluation.dq, evaluation.ddq)
        ):
            raise ValueError("Phase-5 produced non-finite top-K trajectory derivatives")
        artifacts.trajectory_arrays.update(
            topk_positions=to_numpy(evaluation.q, dtype=np.float64),
            topk_velocities=to_numpy(evaluation.dq, dtype=np.float64),
            topk_accelerations=to_numpy(evaluation.ddq, dtype=np.float64),
        )
        topk_shape = evaluation.q.shape[:2]
        poses = torch.stack(
            self.planning_task.robot.fk_collision_spheres(
                evaluation.q.reshape(-1, evaluation.q.shape[-1])
            ),
            dim=-3,
        )
        sphere_positions = link_pos_from_link_tensor(poses)[..., :3].reshape(
            *topk_shape, poses.shape[-3], 3
        )
        artifacts.trajectory_arrays["topk_collision_sphere_positions"] = to_numpy(
            sphere_positions, dtype=np.float32
        )
        artifacts = attach_candidate_timing(
            artifacts,
            topk_time_from_start=to_numpy(topk_times, dtype=np.float64),
            timing_mode=self.space_time_settings.mode,
            duration_min=self.space_time_settings.duration_min,
            duration_max=self.space_time_settings.duration_max,
            timing_control_points=to_numpy(timing_control_points, dtype=np.float64),
        )
        artifacts.result_payload["space_time_guidance"] = {
            "settings": {
                key: value
                for key, value in self.space_time_settings.__dict__.items()
            },
            "steps": self.space_time_guide.statistics,
            "full_candidate_specific_dense_check": True,
            "status": "full_candidate_specific_dense_validated",
        }
        artifacts.result_payload["dynamic_world"].update(
            fixed_timing=False,
            candidate_specific_time=True,
            timing_schema_version=1,
        )
        return artifacts
