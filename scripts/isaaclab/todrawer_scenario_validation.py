#!/usr/bin/env python3
"""Shared geometry and contract checks for random ToDrawer scenarios."""

from __future__ import annotations

import ast
import contextlib
from dataclasses import dataclass
from functools import lru_cache
import io
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
INITIAL_FRANKA_Q = (0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0)
INITIAL_FRANKA_CLEARANCE_SAMPLE_DT_S = 0.02
# These boxes keep randomized anchors inside the ToDrawer work volume while
# allowing the complete incoming path to remain clear of the parked arm.  The
# exact geometry is still accepted/rejected by the 56-sphere test below.
STARTUP_SAFE_ANCHOR_BOUNDS = {
    "A0": ((0.22, 0.24, 0.54), (0.34, 0.42, 0.69)),
    "A1": ((-0.02, 0.24, 0.50), (0.12, 0.42, 0.66)),
    "A2": ((-0.25, 0.24, 0.48), (-0.11, 0.42, 0.64)),
    "S0": ((0.48, -0.48, 0.48), (0.62, -0.32, 0.62)),
    "S1": ((0.38, -0.64, 0.62), (0.54, -0.46, 0.78)),
}


@dataclass(frozen=True)
class AxisAlignedBox:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]


@dataclass(frozen=True)
class TrajectoryClearance:
    robot_base_m: float
    static_environment_m: float


@dataclass(frozen=True)
class InitialFrankaClearance:
    minimum_m: float
    time_s: float
    sphere_index: int


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


@lru_cache(maxsize=1)
def initial_franka_collision_spheres() -> tuple[np.ndarray, np.ndarray]:
    """Return all 56 MPD collision spheres at the benchmark's initial q."""

    # Keep this import lazy so lightweight report readers do not initialize
    # torch_robotics.  The benchmark runtime already requires this dependency.
    import torch

    # torch_robotics still references removed NumPy aliases in some versions.
    for name, value in (("int", int), ("float", float), ("bool", bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    from torch_robotics.robots import RobotPanda

    tensor_args = {"device": "cpu", "dtype": torch.float64}
    with contextlib.redirect_stdout(io.StringIO()):
        robot = RobotPanda(gripper=True, tensor_args=tensor_args)
    q = torch.as_tensor([INITIAL_FRANKA_Q], **tensor_args)
    poses = robot.fk_collision_spheres(q)
    centers = (
        torch.stack(poses)
        .transpose(0, 1)[0, :, :3, 3]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    radii = (
        robot.link_collision_spheres_radii.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=True)
    )
    if centers.shape != (56, 3) or radii.shape != (56,):
        raise ValueError(
            "unexpected Franka collision-sphere geometry: "
            f"centers={centers.shape}, radii={radii.shape}"
        )
    centers.setflags(write=False)
    radii.setflags(write=False)
    return centers, radii


def initial_franka_trajectory_clearance(
    item: dict[str, Any],
    *,
    end_s: float | None = None,
    sample_dt_s: float = INITIAL_FRANKA_CLEARANCE_SAMPLE_DT_S,
) -> InitialFrankaClearance:
    """Check the full moving box against every parked-Franka collision sphere.

    The interval includes both endpoints and defaults to ``0..crossing_time``.
    Object base inflation is part of its effective geometry.  Horizon-rate
    inflation is intentionally not accumulated from episode start: the runtime
    applies it from each observation snapshot into that request's future.
    """

    crossing_time = float(item["crossing_time_s"])
    interval_end = crossing_time if end_s is None else float(end_s)
    if not math.isfinite(interval_end) or interval_end < 0.0 or sample_dt_s <= 0.0:
        raise ValueError("initial-Franka clearance interval is invalid")
    sample_count = max(1, math.ceil(interval_end / sample_dt_s))
    times = np.linspace(0.0, interval_end, sample_count + 1, dtype=np.float64)
    if 0.0 <= crossing_time <= interval_end and not np.any(np.isclose(times, crossing_time, atol=1e-12)):
        times = np.sort(np.append(times, crossing_time))
    positions = np.asarray(
        [object_position_at(item, float(elapsed_s)) for elapsed_s in times],
        dtype=np.float64,
    )
    half_extent = np.asarray(_object_half_extent(item), dtype=np.float64)
    sphere_centers, sphere_radii = initial_franka_collision_spheres()
    gaps = np.maximum(
        np.abs(positions[:, None, :] - sphere_centers[None, :, :])
        - half_extent[None, None, :],
        0.0,
    )
    signed = np.linalg.norm(gaps, axis=2) - sphere_radii[None, :]
    flat_index = int(np.argmin(signed))
    time_index, sphere_index = np.unravel_index(flat_index, signed.shape)
    return InitialFrankaClearance(
        minimum_m=float(signed[time_index, sphere_index]),
        time_s=float(times[time_index]),
        sphere_index=int(sphere_index),
    )


def validate_initial_franka_clearance(
    item: dict[str, Any],
    *,
    end_s: float | None = None,
    minimum_clearance_m: float = 0.0,
) -> InitialFrankaClearance:
    clearance = initial_franka_trajectory_clearance(item, end_s=end_s)
    if clearance.minimum_m <= minimum_clearance_m:
        interval_end = (
            float(item["crossing_time_s"]) if end_s is None else float(end_s)
        )
        raise ValueError(
            f"{item.get('id', '<object>')} intersects the initial Franka "
            f"before t={interval_end:.3f}s: "
            f"clearance={clearance.minimum_m:.6f}m at "
            f"t={clearance.time_s:.3f}s sphere={clearance.sphere_index}"
        )
    return clearance


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
