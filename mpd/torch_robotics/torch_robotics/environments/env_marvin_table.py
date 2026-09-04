"""Minimal table environment descriptor for Marvin adapters."""
from dataclasses import dataclass, field


@dataclass
class EnvMarvinTable:
    name: str = "EnvMarvinTable"
    planning_frame: str = "world"
    obstacles: list[dict] = field(default_factory=list)
    dynamic_objects: list[dict] = field(default_factory=list)

    def update_dynamic_world(self, objects):
        self.dynamic_objects = list(objects)
