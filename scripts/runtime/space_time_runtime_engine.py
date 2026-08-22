"""Separate Phase-5 inference-only space-time MPD runtime entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from mpd.inference.space_time_guidance import (
    InferenceOnlySpaceTimeGuide,
    SpaceTimeGuidanceSettings,
)
from scripts.runtime.dynamic_runtime_engine import DynamicMpdRuntimeEngine
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
    ) -> None:
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
            # Establish the full candidate-specific correctness baseline first.
            capacity_buckets_enabled=False,
            shape_grouping_enabled=False,
            time_table_cache_enabled=False,
            fused_reduction_enabled=False,
            dynamic_guide_pruning_enabled=False,
            trajectory_schema_version=2,
            collision_spheres_float32=True,
            deduplicate_best_trajectory=True,
        )
        if self.planner.cost_guide is None:
            raise ValueError("Phase-5 runtime requires the configured MPD cost guide")
        self.space_time_settings = settings
        self.space_time_guide = InferenceOnlySpaceTimeGuide(
            self.planner.cost_guide,
            self.planning_task,
            self.planner.dataset,
            self.dynamic_field,
            settings,
            self.tensor_args,
        )
        self.planner.cost_guide = self.space_time_guide

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
        topk_times = self.space_time_guide.timing_spline.evaluate(
            timing_control_points,
            require_duration_bounds=True,
        ).time_from_start

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
            "full_candidate_specific_dense_check": False,
            "status": "guidance_enabled_pending_phase5d_full_revalidation",
        }
        artifacts.result_payload["dynamic_world"].update(
            fixed_timing=False,
            candidate_specific_time=True,
            timing_schema_version=1,
        )
        return artifacts
