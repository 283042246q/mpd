#!/usr/bin/env python3
"""Serve Phase-4 dynamic MPD without changing the Phase-3 entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
from typing import Any

from scripts.runtime.dynamic_runtime_engine import DynamicMpdRuntimeEngine
from scripts.runtime.infer_once import DEFAULT_CONFIG_PATH
from scripts.runtime.infer_server import ResidentPlannerService
from scripts.runtime.ipc_protocol import PROTOCOL_SCHEMA_VERSION, ProtocolError


def _add_boolean_switch(parser: argparse.ArgumentParser, name: str, *, default: bool) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true")
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


class DynamicResidentPlannerService(ResidentPlannerService):
    def _update_world(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.state != "READY":
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "NOT_READY",
                "state": self.state,
            }
        snapshot = message.get("world")
        if not isinstance(snapshot, dict):
            raise ProtocolError("world must be a JSON object")
        if not self._plan_lock.acquire(blocking=False):
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "BUSY",
                "reason": "planner_or_world_update_active",
            }
        try:
            version = self._engine.update_world(snapshot)
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "OK",
                "world_version": version,
            }
        finally:
            self._plan_lock.release()

    def _plan(self, message: dict[str, Any]) -> dict[str, Any]:
        world_version = self._require_nonnegative_integer(message, "world_version")
        trajectory_start = message.get("trajectory_start_unix_ns")
        if isinstance(trajectory_start, bool) or not isinstance(trajectory_start, int):
            raise ProtocolError("trajectory_start_unix_ns must be a non-negative integer")
        if self._engine is not None and world_version != self._engine.dynamic_world.world_version:
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "STALE",
                "reason": "world_version_not_loaded",
                "request_seq": message.get("request_seq"),
                "world_version": world_version,
                "loaded_world_version": self._engine.dynamic_world.world_version,
            }
        dynamic_message = dict(message)
        request = dict(message.get("request", {}))
        request["_dynamic_world_version"] = world_version
        request["_trajectory_start_unix_ns"] = trajectory_start
        dynamic_message["request"] = request
        return super()._plan(dynamic_message)

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}.")
        if message.get("op") == "update_world":
            return self._update_world(message)
        return super().dispatch(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Phase-4 dynamic MPD service.")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-dynamic-objects", type=int, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.0)
    parser.add_argument("--process-acceleration-std", type=float, default=0.01)
    _add_boolean_switch(parser, "capacity-buckets", default=True)
    _add_boolean_switch(parser, "shape-grouping", default=True)
    _add_boolean_switch(parser, "time-table-cache", default=True)
    _add_boolean_switch(parser, "fused-reduction", default=True)
    _add_boolean_switch(parser, "dynamic-guide-pruning", default=True)
    parser.add_argument("--trajectory-schema-version", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--trajectory-compression", choices=("none", "zlib"), default="none"
    )
    _add_boolean_switch(parser, "collision-spheres-float32", default=True)
    _add_boolean_switch(parser, "deduplicate-best-trajectory", default=True)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    def engine_factory(state_callback):
        return DynamicMpdRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=state_callback,
            max_dynamic_objects=args.max_dynamic_objects,
            covariance_sigma=args.covariance_sigma,
            process_acceleration_std_m_s2=args.process_acceleration_std,
            capacity_buckets_enabled=args.capacity_buckets,
            shape_grouping_enabled=args.shape_grouping,
            time_table_cache_enabled=args.time_table_cache,
            fused_reduction_enabled=args.fused_reduction,
            dynamic_guide_pruning_enabled=args.dynamic_guide_pruning,
            trajectory_schema_version=args.trajectory_schema_version,
            collision_spheres_float32=args.collision_spheres_float32,
            deduplicate_best_trajectory=args.deduplicate_best_trajectory,
        )

    service = DynamicResidentPlannerService(
        args.socket,
        args.output_root,
        engine_factory,
        trajectory_compression=args.trajectory_compression == "zlib",
    )
    service.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
