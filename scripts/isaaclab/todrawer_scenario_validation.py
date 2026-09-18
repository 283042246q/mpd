#!/usr/bin/env python3
"""Shared geometry and contract checks for random ToDrawer scenarios."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_ENVIRONMENT_SOURCE = (
    REPO_ROOT / "mpd" / "torch_robotics" / "torch_robotics" / "environments" / "env_open_drawer_shelf.py"
)
STATIC_BOX_CONSTANTS = (
    "DRAWER_CABINET_BOXES",
    "OPEN_BOTTOM_DRAWER_BOXES",
    "ADJACENT_SHELF_BOXES",
)
ROBOT_BASE_EXCLUSION_MIN = (-0.183, -0.125, -0.030)
ROBOT_BASE_EXCLUSION_MAX = (0.101, 0.125, 0.171)
DESIGN_EPISODE_DURATION_S = 35.0
TRAJECTORY_SAMPLE_DT_S = 0.05
# Static furniture is visual/context geometry for this benchmark. Dynamic
# objects may pass through it; only the robot-base exclusion is a hard reject.
MINIMUM_STATIC_CLEARANCE_M = 0.0
STATIC_CLEARANCE_WARNING_M = 0.02


@dataclass(frozen=True)
class AxisAlignedBox:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryClearance:
    robot_base_m: float
    static_environment_m: float


def _finite_vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(float(entry) for entry in value)
    if not all(math.isfinite(entry) for entry in result):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _literal_static_specs(path: Path = STATIC_ENVIRONMENT_SOURCE) -> dict[str, Any]:
    """Read the authoritative box constants without importing torch/robot modules."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.as_posix())
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in STATIC_BOX_CONSTANTS:
                values[target.id] = ast.literal_eval(value)
    missing = set(STATIC_BOX_CONSTANTS) - set(values)
    if missing:
        raise ValueError(f"missing static environment constants: {sorted(missing)}")
    return values


def load_static_environment_boxes(static_scene: str | Path | None = None) -> tuple[AxisAlignedBox, ...]:
    """Load boxes from a replay scene, or from EnvOpenDrawerShelf's source."""

    if static_scene is not None:
        source = Path(static_scene)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != "mpd_isaaclab_scene":
            raise ValueError(f"{source}: unsupported static-scene schema")
        boxes = []
        for index, item in enumerate(payload.get("obstacles", [])):
            if item.get("type") != "box":
                continue
            boxes.append(
                AxisAlignedBox(
                    str(item.get("name", f"box-{index}")),
                    _finite_vector(item.get("position"), 3, "static box position"),
                    _finite_vector(item.get("size"), 3, "static box size"),
                )
            )
        if not boxes:
            raise ValueError(f"{source}: static scene contains no boxes")
        return tuple(boxes)

    boxes = []
    for group_name, spec in _literal_static_specs().items():
        centers, sizes = spec.get("centers"), spec.get("sizes")
        if not isinstance(centers, list) or not isinstance(sizes, list) or len(centers) != len(sizes):
            raise ValueError(f"{group_name} centers/sizes are invalid")
        for index, (center, size) in enumerate(zip(centers, sizes)):
            boxes.append(
                AxisAlignedBox(
                    f"{group_name}:{index}",
                    _finite_vector(center, 3, f"{group_name}.centers[{index}]"),
                    _finite_vector(size, 3, f"{group_name}.sizes[{index}]"),
                )
            )
    return tuple(boxes)


def object_position_at(item: dict[str, Any], elapsed_s: float) -> list[float]:
    """Evaluate the same deterministic motion equation used by world_demo_node."""

    relative_time = float(elapsed_s) - float(item["crossing_time_s"])
    motion = item.get("motion", {"type": "constant_velocity"})
    motion_type = motion["type"]
    displacement = float(item["speed_m_s"]) * relative_time
    if motion_type == "constant_acceleration":
        displacement += 0.5 * float(motion["longitudinal_acceleration_m_s2"]) * relative_time**2
    if motion_type in {"smooth_speed_variation", "curved_speed_variation"}:
        amplitude = float(motion["speed_variation_amplitude_m_s"])
        frequency = float(motion["speed_variation_angular_frequency_rad_s"])
        phase = float(motion["speed_variation_phase_rad"])
        displacement += amplitude / frequency * (math.sin(frequency * relative_time + phase) - math.sin(phase))
    position = [
        float(anchor) + float(direction) * displacement
        for anchor, direction in zip(item["anchor_position"], item["direction"])
    ]
    if motion_type in {"sinusoidal_curve", "curved_speed_variation"}:
        amplitude = float(motion["lateral_amplitude_m"])
        frequency = float(motion["lateral_angular_frequency_rad_s"])
        phase = float(motion["lateral_phase_rad"])
        lateral = amplitude * (math.sin(frequency * relative_time + phase) - math.sin(phase))
        position = [
            value + float(direction) * lateral for value, direction in zip(position, motion["lateral_direction"])
        ]
    return position


def _object_half_extent(item: dict[str, Any]) -> tuple[float, float, float]:
    local_sdf = item["local_sdf"]
    if local_sdf.get("type") != "box":
        raise ValueError("ToDrawer random validation currently requires box objects")
    size = _finite_vector(local_sdf.get("size_xyz"), 3, "local_sdf.size_xyz")
    padding = float(item["inflation"]["base_m"])
    return tuple(0.5 * value + padding for value in size)


def trajectory_clearances(
    item: dict[str, Any],
    *,
    static_boxes: tuple[AxisAlignedBox, ...] | None = None,
    duration_s: float = DESIGN_EPISODE_DURATION_S,
    sample_dt_s: float = TRAJECTORY_SAMPLE_DT_S,
) -> TrajectoryClearance:
    """Sample the complete object trajectory against robot base and furniture."""

    if duration_s <= 0.0 or sample_dt_s <= 0.0:
        raise ValueError("trajectory duration and sample dt must be positive")
    boxes = load_static_environment_boxes() if static_boxes is None else static_boxes
    half_extent = _object_half_extent(item)
    base_center = tuple(
        0.5 * (lower + upper) for lower, upper in zip(ROBOT_BASE_EXCLUSION_MIN, ROBOT_BASE_EXCLUSION_MAX)
    )
    base_half = tuple(0.5 * (upper - lower) for lower, upper in zip(ROBOT_BASE_EXCLUSION_MIN, ROBOT_BASE_EXCLUSION_MAX))
    sample_count = math.ceil(duration_s / sample_dt_s)
    times = [duration_s * index / sample_count for index in range(sample_count + 1)]
    crossing_time = float(item["crossing_time_s"])
    if 0.0 <= crossing_time <= duration_s:
        times.append(crossing_time)
    positions = np.asarray(
        [object_position_at(item, elapsed_s) for elapsed_s in times],
        dtype=np.float64,
    )
    object_half = np.asarray(half_extent, dtype=np.float64)
    base_gaps = np.maximum(
        np.abs(positions - np.asarray(base_center))
        - object_half
        - np.asarray(base_half),
        0.0,
    )
    minimum_base = float(np.linalg.norm(base_gaps, axis=1).min())
    box_centers = np.asarray([box.center for box in boxes], dtype=np.float64)
    box_halves = 0.5 * np.asarray([box.size for box in boxes], dtype=np.float64)
    static_gaps = np.maximum(
        np.abs(positions[:, None, :] - box_centers[None, :, :])
        - object_half[None, None, :]
        - box_halves[None, :, :],
        0.0,
    )
    minimum_static = float(np.linalg.norm(static_gaps, axis=2).min())
    return TrajectoryClearance(minimum_base, minimum_static)


def validate_trajectory_clearance(
    item: dict[str, Any],
    *,
    static_boxes: tuple[AxisAlignedBox, ...] | None = None,
) -> TrajectoryClearance:
    clearance = trajectory_clearances(item, static_boxes=static_boxes)
    if clearance.robot_base_m <= 0.0:
        raise ValueError(f"{item.get('id', '<object>')} intersects the robot-base exclusion")
    return clearance
