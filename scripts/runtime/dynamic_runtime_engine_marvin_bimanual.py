"""Phase-5 latest-snapshot Marvin bimanual runtime engine."""
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.covariance_sigma = float(covariance_sigma)
        session = self._session
        static_field = session.planning_task.get_collision_objects_field()
        if static_field is None:
            raise DynamicWorldError("dynamic runtime requires static Warehouse collision field")
        duration = float(session.config.trajectory_duration)
        self.dynamic_world = FixedCapacityDynamicWorld(
            max_dynamic_objects,
            trajectory_duration_s=duration,
            tensor_args=session.tensor_args,
            covariance_sigma=0.0,
            process_acceleration_std_m_s2=0.0,
            capacity_buckets_enabled=True,
            shape_grouping_enabled=True,
            time_table_cache_enabled=True,
            fused_reduction_enabled=True,
        )
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
                collision.cost.collision_objects_field = self.dynamic_field
        ranked = session.planner.dense_validation_config.get("ranked_early_exit", {})
        ranked["enabled"] = False
        session.planner.dense_validation_config["ranked_early_exit"] = ranked
        self.external_valid_until_unix_ns = 0

    def update_world(self, world: dict[str, Any]) -> int:
        frozen = _snapshot_world(world, covariance_sigma=self.covariance_sigma)
        version = self.dynamic_world.update(frozen)
        self.external_valid_until_unix_ns = int(frozen["valid_until_unix_ns"])
        return version

    def health(self):
        health = super().health()
        health["dynamic_world"] = {
            "mode": "snapshot_no_time",
            "world_version": self.dynamic_world.world_version,
            "active_objects": self.dynamic_world.active_count,
            "max_objects": self.dynamic_world.max_objects,
            "frame_id": self.dynamic_world.frame_id,
            "valid_until_unix_ns": self.external_valid_until_unix_ns,
            "motion_frozen": True,
        }
        return health

    def plan(self, raw_request: dict[str, Any]) -> PlanArtifacts:
        try:
            request = BimanualRequest.from_dict(raw_request)
            if request.runtime_mode != "snapshot_no_time":
                raise ContractError("Phase-5 engine only accepts snapshot_no_time")
            if request.world_version != self.dynamic_world.world_version:
                raise DynamicWorldError("request world_version is not the loaded latest snapshot")
            now = time.time_ns()
            if now >= self.external_valid_until_unix_ns:
                raise DynamicWorldError("dynamic snapshot expired before planning")
            # Objects are frozen, so internal validity only needs to cover the fixed
            # trajectory grid; external validity remains a planning-result gate.
            self.dynamic_world.valid_until_unix_ns = max(
                self.dynamic_world.valid_until_unix_ns,
                now + int((float(self._session.config.trajectory_duration) + 1.0) * 1e9),
            )
            self.dynamic_world.set_plan_start(
                max(now, self.dynamic_world.stamp_unix_ns),
                world_version=request.world_version,
            )
            artifacts = super().plan(raw_request)
            if time.time_ns() >= self.external_valid_until_unix_ns:
                raise DynamicWorldError("dynamic snapshot expired after planning")
            artifacts.result_payload["dynamic_world"] = {
                "mode": "snapshot_no_time",
                "world_version": request.world_version,
                "valid_until_unix_ns": self.external_valid_until_unix_ns,
                "active_objects": self.dynamic_world.active_count,
            }
            return artifacts
        except DynamicWorldError as error:
            raise MarvinDynamicContractError(str(error)) from error
