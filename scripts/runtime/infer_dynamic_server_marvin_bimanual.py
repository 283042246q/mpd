#!/usr/bin/env python3
"""Length-framed Unix-socket service for Phase-5 snapshot replanning."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.inference.inference_marvin_bimanual import DEFAULT_CONFIG
from scripts.runtime.dynamic_runtime_engine_marvin_bimanual import MarvinBimanualDynamicRuntimeEngine
from mpd.inference.dynamic_collision import DynamicWorldError
from scripts.runtime.infer_server import ResidentPlannerService
from scripts.runtime.ipc_protocol import PROTOCOL_SCHEMA_VERSION, ProtocolError


class MarvinDynamicPlannerService(ResidentPlannerService):
    def _update_world(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.state != "READY":
            return {"schema_version": 1, "status": "NOT_READY", "state": self.state}
        world = message.get("world")
        if not isinstance(world, dict):
            raise ProtocolError("world must be a JSON object")
        if not self._plan_lock.acquire(blocking=False):
            return {"schema_version": 1, "status": "BUSY", "reason": "planner_or_update_active"}
        try:
            try:
                version = self._engine.update_world(world)
            except DynamicWorldError as error:
                return {
                    "schema_version": 1,
                    "status": "INVALID_REQUEST",
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            return {"schema_version": 1, "status": "OK", "world_version": version}
        finally:
            self._plan_lock.release()

    def _plan(self, message):
        version = self._require_nonnegative_integer(message, "world_version")
        loaded = 0 if self._engine is None else self._engine.dynamic_world.world_version
        if version != loaded:
            return {
                "schema_version": 1,
                "status": "STALE",
                "reason": "world_version_not_latest",
                "request_seq": message.get("request_seq"),
                "world_version": version,
                "loaded_world_version": loaded,
            }
        return super()._plan(message)

    def dispatch(self, message):
        if message.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError("invalid schema_version")
        if message.get("op") == "update_world":
            return self._update_world(message)
        return super().dispatch(message)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-dynamic-objects", type=int, default=16)
    parser.add_argument("--covariance-sigma", type=float, default=3.0)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)

    def factory(callback):
        return MarvinBimanualDynamicRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=callback,
            max_dynamic_objects=args.max_dynamic_objects,
            covariance_sigma=args.covariance_sigma,
        )

    MarvinDynamicPlannerService(args.socket, args.output_root, factory).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
