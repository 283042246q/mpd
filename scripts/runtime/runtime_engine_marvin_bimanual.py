"""Thread-safe latest-only static Marvin runtime engine."""
from __future__ import annotations

import time
from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.inference.inference_marvin_bimanual import plan


class MarvinBimanualRuntimeEngine:
    def __init__(self, *, planner=plan):
        self._planner = planner
        self._generation = 0
        self._latest_world_version = 0

    def update_world_version(self, world_version: int):
        if world_version <= self._latest_world_version:
            raise ValueError("world_version must increase monotonically")
        self._latest_world_version = world_version

    def plan_once(self, request: BimanualRequest):
        if request.world_version < self._latest_world_version:
            raise ValueError("request uses a stale world_version")
        if request.deadline_monotonic_ns and time.monotonic_ns() >= request.deadline_monotonic_ns:
            raise TimeoutError("planning deadline has expired")
        self._generation += 1
        return self._planner(request)

