"""Phase-7 inference-only space-time runtime for Marvin bimanual MPD."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mpd.inference.inference import _compute_candidate_ranking
from mpd.inference.space_time_guidance import (
    InferenceOnlySpaceTimeGuide,
    SpaceTimeGuidanceSettings,
)
from scripts.runtime.dynamic_runtime_engine_marvin_bimanual import (
    MarvinBimanualDynamicRuntimeEngine,
)
from scripts.runtime.space_time_runtime_engine import _phase5_selection_score
from scripts.runtime.timing_contract import (
    TIMING_SCHEMA_VERSION,
    validate_candidate_times,
)


class MarvinBimanualSpaceTimeRuntimeEngine(MarvinBimanualDynamicRuntimeEngine):
    """Optimize candidate timing at inference while retaining the spatial checkpoint."""

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
        dynamic_selection_enabled: bool = True,
    ) -> None:
        if dynamic_space_time_pruning_enabled:
            raise ValueError("candidate-specific dynamic space-time pruning is not implemented")
        settings = SpaceTimeGuidanceSettings.from_mapping(space_time_settings, mode=timing_mode)
        super().__init__(
            config_path=config_path,
            runtime_output_root=runtime_output_root,
            device_text=device_text,
            state_callback=state_callback,
            max_dynamic_objects=max_dynamic_objects,
            covariance_sigma=covariance_sigma,
            process_acceleration_std_m_s2=process_acceleration_std_m_s2,
            runtime_mode="fixed_time_dynamic",
        )
        session = self._session
        self.runtime_mode = "inference_time_optimized"
        session.accepted_runtime_modes = {self.runtime_mode}
        if session.planner.cost_guide is None:
            raise ValueError("space-time runtime requires the configured MPD cost guide")
        spatial_guide = session.planner.cost_guide
        if static_spatial_pruning_enabled and not spatial_guide.gradient_pruning_config["enabled"]:
            raise ValueError("static spatial pruning was requested but is disabled in the MPD config")
        spatial_guide.gradient_pruning_enabled = bool(static_spatial_pruning_enabled)
        self.static_spatial_pruning_enabled = bool(static_spatial_pruning_enabled)
        self.dynamic_space_time_pruning_enabled = False
        self.dynamic_selection_enabled = bool(dynamic_selection_enabled)
        self.space_time_settings = settings
        self.space_time_guide = InferenceOnlySpaceTimeGuide(
            spatial_guide,
            session.planning_task,
            session.planner.dataset,
            self.guide_dynamic_field,
            settings,
            session.tensor_args,
            collision_robot=getattr(spatial_guide, "guide_collision_robot", None),
        )
        session.planner.cost_guide = self.space_time_guide
        session.result_postprocessor = self._postprocess_plan_results

    def _postprocess_plan_results(self, results):
        import torch

        session = self._session
        control_points = results.control_points_iter_0
        normalized = session.planner.dataset.normalize_control_points(control_points)
        timing_control_points = self.space_time_guide.timing_control_points
        with torch.no_grad():
            _, cost_breakdown, timing = self.space_time_guide.evaluate_control_points(normalized, timing_control_points)
            dense_cfg = session.planner.dense_validation_config
            if int(dense_cfg["runtime_points"]) != int(timing.q.shape[1]):
                raise ValueError("space-time DenseCheck resolution must match the timing spline grid")
            dense = session.planner.dense_validator.validate(
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

        duration_valid = (timing.duration >= self.space_time_settings.duration_min) & (
            timing.duration <= self.space_time_settings.duration_max
        )
        finite = (
            torch.isfinite(timing.q).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.dq).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.ddq).flatten(start_dim=1).all(dim=-1)
            & torch.isfinite(timing.time_from_start).all(dim=-1)
        )
        valid_mask = dense.trajectory_valid_mask & duration_valid & finite
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        spatial_score, _ = _compute_candidate_ranking(
            "weighted_metrics",
            session.config,
            session.planner.dataset,
            session.planning_task,
            None,
            control_points,
            timing.q,
            timing.dq,
            timing.ddq,
            results.ee_pose_goal,
            results.active_ee_mask,
        )
        collision_waypoint_mask = dense.environment_collision_mask | dense.self_collision_mask
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
            results.update(
                control_points_valid=control_points[:0],
                q_trajs_pos_valid=timing.q[:0],
                q_trajs_vel_valid=timing.dq[:0],
                q_trajs_acc_valid=timing.ddq[:0],
                valid_candidate_timesteps=timing.time_from_start[:0],
                valid_timing_control_points=timing_control_points[:0],
                valid_trajectory_selection_scores=spatial_score[:0],
                control_points_best=None,
                q_trajs_pos_best=None,
                q_trajs_vel_best=None,
                q_trajs_acc_best=None,
                best_trajectory_selection_details={
                    "method": "marvin_spatial_space_time_cost",
                    "reason": "no_full_candidate_specific_dense_valid_trajectory",
                },
                timesteps=None,
            )
            return results

        valid_spatial_score = spatial_score.index_select(0, valid_indices)
        valid_dynamic_risk = cost_breakdown["dynamic_collision"].index_select(0, valid_indices)
        valid_duration = timing.duration.index_select(0, valid_indices)
        valid_timing_smoothness = cost_breakdown["timing_smoothness"].index_select(0, valid_indices)
        valid_cost, score_components = _phase5_selection_score(
            valid_spatial_score,
            valid_dynamic_risk,
            valid_duration,
            valid_timing_smoothness,
            duration_min=self.space_time_settings.duration_min,
            duration_max=self.space_time_settings.duration_max,
            dynamic_selection_enabled=self.dynamic_selection_enabled,
        )
        selected_valid = int(torch.argmin(valid_cost).item())
        selected_candidate = int(valid_indices[selected_valid].item())
        results.update(
            control_points_valid=control_points.index_select(0, valid_indices),
            q_trajs_pos_valid=timing.q.index_select(0, valid_indices),
            q_trajs_vel_valid=timing.dq.index_select(0, valid_indices),
            q_trajs_acc_valid=timing.ddq.index_select(0, valid_indices),
            valid_candidate_timesteps=timing.time_from_start.index_select(0, valid_indices),
            valid_timing_control_points=timing_control_points.index_select(0, valid_indices),
            valid_trajectory_selection_scores=valid_cost,
            control_points_best=control_points[selected_candidate],
            q_trajs_pos_best=timing.q[selected_candidate],
            q_trajs_vel_best=timing.dq[selected_candidate],
            q_trajs_acc_best=timing.ddq[selected_candidate],
            best_trajectory_selection_details={
                "method": "marvin_spatial_space_time_cost",
                "selected_candidate_index": selected_candidate,
                "selected_valid_index": selected_valid,
                "score": float(valid_cost[selected_valid].item()),
                "duration_s": float(timing.duration[selected_candidate].item()),
                "components": {
                    name: float(value[selected_valid].item())
                    for name, value in score_components.items()
                    if name != "dynamic_selection_enabled"
                },
                "dynamic_selection_enabled": self.dynamic_selection_enabled,
            },
            timesteps=timing.time_from_start[selected_candidate],
        )
        return results

    def health(self) -> dict[str, Any]:
        response = super().health()
        response["dynamic_world"].update(
            mode=self.runtime_mode,
            candidate_specific_time=True,
            trajectory_schema_version=3,
        )
        response["space_time"] = {
            "enabled": True,
            "mode": self.space_time_settings.mode,
            "timing_schema_version": TIMING_SCHEMA_VERSION,
            "trajectory_schema_version": 3,
            "candidate_specific_time": True,
            "duration_min_s": self.space_time_settings.duration_min,
            "duration_max_s": self.space_time_settings.duration_max,
            "pruning": {
                "static_spatial": self.static_spatial_pruning_enabled,
                "dynamic_space_time": self.dynamic_space_time_pruning_enabled,
            },
        }
        return response

    def plan(self, raw_request: dict[str, Any]):
        import numpy as np
        import torch

        self.space_time_guide.reset(int(self._session.config.n_trajectory_samples))
        artifacts = super().plan(raw_request)
        results = self._session.last_plan_results
        source_indices = torch.as_tensor(
            artifacts.trajectory_arrays["top_k_candidate_indices"],
            dtype=torch.long,
            device=self._session.device,
        )
        top_k_times = results.candidate_timesteps.index_select(0, source_indices)
        top_k_times_numpy = validate_candidate_times(
            top_k_times.detach().cpu().numpy(),
            expected_candidates=int(source_indices.numel()),
            expected_horizon=int(results.q_trajs_pos_iter_0.shape[1]),
            duration_min=self.space_time_settings.duration_min,
            duration_max=self.space_time_settings.duration_max,
        )
        timing_control_points = (
            results.timing_control_points.index_select(0, source_indices).detach().cpu().numpy().astype(np.float64)
        )
        arrays = artifacts.trajectory_arrays
        arrays.update(
            artifact_schema_version=np.asarray(3, dtype=np.int64),
            timing_schema_version=np.asarray(TIMING_SCHEMA_VERSION, dtype=np.int64),
            best_trajectory_top_k_index=np.asarray(0, dtype=np.int64),
            top_k_time_from_start=top_k_times_numpy,
            time_from_start=top_k_times_numpy[0],
            timing_control_points=timing_control_points,
            candidate_durations=top_k_times_numpy[:, -1],
        )
        artifacts.result_payload["time_from_start"] = top_k_times_numpy[0].tolist()
        artifacts.result_payload["trajectory_artifact"].update(
            schema_version=3,
            timing_schema_version=TIMING_SCHEMA_VERSION,
            candidate_specific_time=True,
        )
        artifacts.result_payload["trajectory"] = {
            "duration_s": float(top_k_times_numpy[0, -1]),
            "duration_min_s": self.space_time_settings.duration_min,
            "duration_max_s": self.space_time_settings.duration_max,
            "timing_mode": self.space_time_settings.mode,
            "timing_schema_version": TIMING_SCHEMA_VERSION,
        }
        artifacts.result_payload["top_k_timing"] = {
            "durations_s": top_k_times_numpy[:, -1].tolist(),
            "candidate_specific_time": True,
        }
        artifacts.result_payload["space_time_guidance"] = {
            "settings": dict(self.space_time_settings.__dict__),
            "steps": self.space_time_guide.statistics,
            "full_candidate_specific_dense_check": True,
            "status": "full_candidate_specific_dense_validated",
        }
        artifacts.result_payload["dynamic_world"].update(
            mode=self.runtime_mode,
            fixed_timing=False,
            candidate_specific_time=True,
            timing_schema_version=TIMING_SCHEMA_VERSION,
        )
        return artifacts
