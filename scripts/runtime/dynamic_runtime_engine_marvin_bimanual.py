from __future__ import annotations

from dataclasses import dataclass
from mpd.bimanual.runtime_contract import BimanualRequest
try:
    from .runtime_engine_marvin_bimanual import MarvinBimanualRuntimeEngine
except ImportError:
    from runtime_engine_marvin_bimanual import MarvinBimanualRuntimeEngine


@dataclass
class DynamicWorldState:
    world_version: int = 0
    valid_until_unix_ns: int = 0
    objects: tuple[dict, ...] = ()


class MarvinBimanualDynamicRuntimeEngine(MarvinBimanualRuntimeEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world = DynamicWorldState()

    def update_world(self, world: dict):
        version = world.get("world_version")
        if not isinstance(version, int) or version <= self.world.world_version:
            raise ValueError("world_version must increase monotonically")
        self.world = DynamicWorldState(version, int(world.get("valid_until_unix_ns", 0)), tuple(world.get("objects", ())))
        self._latest_world_version = version

    def plan_latest(self, request: BimanualRequest):
        if request.world_version != self.world.world_version:
            raise ValueError("request world_version is not the latest snapshot")
        return self.plan_once(request)
