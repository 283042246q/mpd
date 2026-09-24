"""Phase-5 snapshot and Phase-6 fixed-time Marvin bimanual runtime engine."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from mpd.bimanual.runtime_contract import BimanualRequest, ContractError
from mpd.inference.dynamic_collision import (
    DynamicWorldError,
    FixedCapacityDynamicWorld,
    StaticDynamicCollisionField,
)
from scripts.runtime.runtime_engine_marvin_bimanual import (
    MarvinBimanualRuntimeEngine,
    PlanArtifacts,
)
from scripts.runtime.infer_once import RuntimeContractError


class MarvinDynamicContractError(RuntimeContractError):
    status = "invalid_request"


def _snapshot_world(world: dict[str, Any], *, covariance_sigma: float) -> dict[str, Any]:
    """Freeze motion at the message stamp and conservatively fold uncertainty in."""
    if not isinstance(world, dict):
        raise DynamicWorldError("world must be a JSON object")
    if world.get("frame_id") != "world":
        raise DynamicWorldError("snapshot_no_time world frame must be 'world'")
    converted = dict(world)
    objects = []
    for index, source in enumerate(world.get("objects", ())):
        if not isinstance(source, dict):
            raise DynamicWorldError(f"objects[{index}] must be an object")
        item = dict(source)
        covariance = item.get("covariance_6x6", [0.0] * 36)
        if not isinstance(covariance, list) or len(covariance) != 36:
            raise DynamicWorldError("covariance_6x6 must contain 36 values")
        covariance = [float(value) for value in covariance]
        if not all(math.isfinite(value) for value in covariance):
            raise DynamicWorldError("covariance_6x6 contains NaN or Inf")
        max_position_variance = max(covariance[0], covariance[7], covariance[14], 0.0)
        inflation = item.get("inflation", {})
        if not isinstance(inflation, dict):
            raise DynamicWorldError("inflation must be an object")
        base = float(inflation.get("base_m", 0.0))
        item["linear_velocity"] = [0.0, 0.0, 0.0]
        item["covariance_6x6"] = [0.0] * 36
        item["inflation"] = {
            "mode": "linear",
            "base_m": base + covariance_sigma * math.sqrt(max_position_variance),
            "horizon_rate_m_s": 0.0,
        }
        objects.append(item)
    converted["objects"] = objects
    return converted


class MarvinBimanualDynamicRuntimeEngine(MarvinBimanualRuntimeEngine):
    """Resident static model plus a fixed-capacity, frozen obstacle snapshot."""

    def __init__(
        self,
        *args,
        max_dynamic_objects: int = 16,
        covariance_sigma: float = 3.0,
        process_acceleration_std_m_s2: float = 0.01,
        runtime_mode: str = "snapshot_no_time",
        **kwargs,
    ):
        if runtime_mode not in {"snapshot_no_time", "fixed_time_dynamic"}:
            raise ValueError("runtime_mode must be snapshot_no_time or fixed_time_dynamic")
        super().__init__(*args, **kwargs)
        self.runtime_mode = runtime_mode
        self.covariance_sigma = float(covariance_sigma)
        session = self._session
        session.accepted_runtime_modes = {runtime_mode}
        static_field = session.planning_task.get_collision_objects_field()
        if static_field is None:
            raise DynamicWorldError("dynamic runtime requires static Warehouse collision field")
        duration = float(session.config.trajectory_duration)
        self.dynamic_world = FixedCapacityDynamicWorld(
            max_dynamic_objects,
            trajectory_duration_s=duration,
            tensor_args=session.tensor_args,
            covariance_sigma=(0.0 if runtime_mode == "snapshot_no_time" else covariance_sigma),
            process_acceleration_std_m_s2=(
                0.0 if runtime_mode == "snapshot_no_time" else process_acceleration_std_m_s2
            ),
            capacity_buckets_enabled=True,
            shape_grouping_enabled=True,
            time_table_cache_enabled=True,
            fused_reduction_enabled=True,
        )
        # Dense validation uses the full Marvin sphere model.  The configured
        # guide may use the reduced foam_pika_100 collision robot, whose sphere
        # count and margins differ, so it needs a shape-compatible adapter.
        self.dynamic_field = StaticDynamicCollisionField(static_field, self.dynamic_world)
        session.planning_task.df_collision_objects = self.dynamic_field
        session.planning_task._collision_fields = [
            session.planning_task.df_collision_self,
            self.dynamic_field,
            session.planning_task.df_collision_ws_boundaries,
        ]
        if session.planner.cost_guide is not None:
            collision = session.planner.cost_guide.costs.get("CostTaskSpaceCollisionObjects")
            if collision is not None:
                guide_static_field = collision.cost.collision_objects_field
                self.guide_dynamic_field = StaticDynamicCollisionField(guide_static_field, self.dynamic_world)
                collision.cost.collision_objects_field = self.guide_dynamic_field
            else:
                self.guide_dynamic_field = self.dynamic_field
            # Dynamic obstacles must be queried at every fixed time point.  The
            # static broad-phase certificates cannot prove future-time safety.
            session.planner.cost_guide.gradient_pruning_enabled = False
        ranked = session.planner.dense_validation_config.get("ranked_early_exit", {})
        ranked["enabled"] = False
        session.planner.dense_validation_config["ranked_early_exit"] = ranked
        self.external_valid_until_unix_ns = 0

    def update_world(self, world: dict[str, Any]) -> int:
        if not isinstance(world, dict) or world.get("frame_id") != "world":
            raise DynamicWorldError("dynamic world frame must be 'world'")
        payload = (
            _snapshot_world(world, covariance_sigma=self.covariance_sigma)
            if self.runtime_mode == "snapshot_no_time"
            else world
        )
        version = self.dynamic_world.update(payload)
        self.external_valid_until_unix_ns = int(payload["valid_until_unix_ns"])
        return version

    def health(self):
        health = super().health()
        health["dynamic_world"] = {
            "mode": self.runtime_mode,
            "world_version": self.dynamic_world.world_version,
            "active_objects": self.dynamic_world.active_count,
            "max_objects": self.dynamic_world.max_objects,
            "frame_id": self.dynamic_world.frame_id,
            "valid_until_unix_ns": self.external_valid_until_unix_ns,
            "motion_frozen": self.runtime_mode == "snapshot_no_time",
            "motion_model": ("frozen" if self.runtime_mode == "snapshot_no_time" else "constant_velocity"),
            "candidate_specific_time": False,
            "trajectory_schema_version": 2,
        }
        return health

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        import numpy as np
        import torch
        import time

        from torch_robotics.torch_kinematics_tree.geometrics.utils import (
            link_pos_from_link_tensor,
        )

        try:
            request = BimanualRequest.from_dict(raw_request)
            if request.runtime_mode != self.runtime_mode:
                raise ContractError(f"dynamic engine accepts only {self.runtime_mode}")
            if request.world_version != self.dynamic_world.world_version:
                raise DynamicWorldError("request world_version is not the loaded latest snapshot")
            now = time.time_ns()
            if now >= self.external_valid_until_unix_ns:
                raise DynamicWorldError("dynamic snapshot expired before planning")
            if self.runtime_mode == "snapshot_no_time":
                # Objects are frozen, so internal validity only needs to cover the
                # fixed trajectory grid; external validity remains a result gate.
                self.dynamic_world.valid_until_unix_ns = max(
                    self.dynamic_world.valid_until_unix_ns,
                    now + int((float(self._session.config.trajectory_duration) + 1.0) * 1e9),
                )
            requested_start = raw_request.get("_trajectory_start_unix_ns")
            if requested_start is None:
                requested_start = max(now, self.dynamic_world.stamp_unix_ns)
            if isinstance(requested_start, bool) or not isinstance(requested_start, int):
                raise DynamicWorldError("trajectory_start_unix_ns must be an integer")
            trajectory_start_unix_ns = requested_start
            self.dynamic_world.set_plan_start(
                trajectory_start_unix_ns,
                world_version=request.world_version,
            )
            artifacts = super().plan(raw_request)
            dynamic_export_started = time.perf_counter()
            if time.time_ns() >= self.external_valid_until_unix_ns:
                raise DynamicWorldError("dynamic snapshot expired after planning")
            artifacts.result_payload["dynamic_world"] = {
                "mode": self.runtime_mode,
                "world_version": request.world_version,
                "valid_until_unix_ns": self.external_valid_until_unix_ns,
                "active_objects": self.dynamic_world.active_count,
                "trajectory_start_unix_ns": trajectory_start_unix_ns,
                "fixed_timing": True,
                "motion_model": ("frozen" if self.runtime_mode == "snapshot_no_time" else "constant_velocity"),
            }
            session = self._session
            results = session.last_plan_results
            if int(results.dense_validation_candidates_checked) != int(results.q_trajs_pos_iter_0.shape[0]) or not bool(
                results.dense_validation_complete
            ):
                raise DynamicWorldError("dynamic final DenseCheck did not evaluate every candidate")
            top_positions = np.asarray(artifacts.trajectory_arrays["top_k_positions"], dtype=np.float64)
            if top_positions.ndim != 3 or top_positions.shape[-1] != 14:
                raise DynamicWorldError("top-K positions must have shape [K,T,14]")
            q = torch.as_tensor(top_positions, **session.tensor_args)
            poses = torch.stack(session.robot.fk_collision_spheres(q.reshape(-1, 14)), dim=-3)
            sphere_positions = link_pos_from_link_tensor(poses)[..., :3].reshape(
                top_positions.shape[0],
                top_positions.shape[1],
                poses.shape[-3],
                3,
            )
            top_spheres = sphere_positions.detach().cpu().numpy().astype(np.float64)
            sphere_radii = session.robot.link_collision_spheres_radii.detach().cpu().numpy().astype(np.float64)
            if "top_k_object_path_pose_xyzw" in artifacts.trajectory_arrays:
                payload_poses = np.asarray(
                    artifacts.trajectory_arrays["top_k_object_path_pose_xyzw"],
                    dtype=np.float64,
                )
                payload_size = np.asarray(
                    artifacts.trajectory_arrays["payload_size_xyz"],
                    dtype=np.float64,
                )
                if payload_poses.shape[:2] != top_positions.shape[:2] or (
                    payload_poses.shape[-1] != 7 or payload_size.shape != (3,)
                ):
                    raise DynamicWorldError("cooperative payload artifact is inconsistent")
                top_spheres = np.concatenate((top_spheres, payload_poses[..., None, :3]), axis=2)
                sphere_radii = np.concatenate((sphere_radii, [0.5 * float(np.linalg.norm(payload_size))]))
            if not np.allclose(
                artifacts.trajectory_arrays["positions"],
                top_positions[0],
                rtol=0.0,
                atol=1e-7,
            ):
                raise DynamicWorldError("selected trajectory is not top-K candidate zero")
            artifacts.trajectory_arrays.update(
                artifact_schema_version=np.asarray(2, dtype=np.int64),
                best_trajectory_top_k_index=np.asarray(0, dtype=np.int64),
                top_k_collision_sphere_positions=top_spheres,
                collision_sphere_positions=top_spheres[0],
                collision_sphere_radii=sphere_radii,
                trajectory_start_unix_ns=np.asarray(trajectory_start_unix_ns, dtype=np.int64),
            )
            artifacts.result_payload["trajectory_artifact"] = {
                "schema_version": 2,
                "top_k_count": int(top_positions.shape[0]),
                "best_trajectory_top_k_index": 0,
                "trajectory_start_unix_ns": trajectory_start_unix_ns,
            }
            artifacts.result_payload.setdefault("resident_runtime", {}).setdefault("timing_s", {})[
                "dynamic_collision_export"
            ] = (time.perf_counter() - dynamic_export_started)
            return artifacts
        except DynamicWorldError as error:
            raise MarvinDynamicContractError(str(error)) from error
