"""Pure-Python timeline model for Phase-4 IsaacLab replays.

The renderer intentionally consumes recorded planning decisions instead of
re-running handoff or collision logic.  This keeps a video faithful to the
world versions, accepted plans, rejected segments, and brake events that were
actually recorded by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "mpd_dynamic_replay"
SCHEMA_VERSION = 1
PLAN_STATUSES = frozenset(("accepted", "superseded", "rejected", "braking"))
EVENT_TYPES = frozenset(("handoff", "brake"))

COLOR_GRAY = (0.45, 0.45, 0.45, 0.75)
COLOR_BLUE = (0.10, 0.35, 1.00, 1.00)
COLOR_GREEN = (0.10, 0.90, 0.25, 1.00)
COLOR_YELLOW = (1.00, 0.80, 0.05, 0.30)
COLOR_RED = (1.00, 0.05, 0.05, 1.00)
COLOR_PURPLE = (0.70, 0.15, 0.95, 1.00)


class DynamicReplayError(ValueError):
    """The dynamic replay manifest or one of its trajectories is invalid."""


@dataclass(frozen=True)
class TrajectoryData:
    """Joint-space trajectory loaded from one Phase-4 ``trajectory.npz``."""

    positions: np.ndarray
    times_s: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.times_s[-1])


@dataclass(frozen=True)
class PlanRecord:
    """One recorded planning result and its execution interval."""

    plan_id: str
    trajectory_path: Path
    trajectory: TrajectoryData
    created_s: float
    start_s: float
    status: str
    active_from_s: float | None
    active_until_s: float | None
    collision_segments_s: tuple[tuple[float, float], ...]

    @property
    def trajectory_end_s(self) -> float:
        return self.start_s + self.trajectory.duration_s


@dataclass(frozen=True)
class WorldSnapshot:
    """Recorded dynamic-world snapshot at replay-relative time ``time_s``."""

    time_s: float
    world_version: int
    valid_until_s: float
    objects: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ReplayEvent:
    """Recorded handoff or controlled-braking event."""

    event_type: str
    time_s: float
    duration_s: float
    plan_id: str | None
    position: tuple[float, float, float] | None
    reason: str | None


@dataclass(frozen=True)
class DynamicReplayManifest:
    """Validated multi-plan replay description."""

    path: Path
    frame_id: str
    env_name: str
    duration_s: float
    static_scene: dict[str, Any]
    plans: tuple[PlanRecord, ...]
    world_snapshots: tuple[WorldSnapshot, ...]
    events: tuple[ReplayEvent, ...]
    initial_q: np.ndarray


@dataclass(frozen=True)
class PredictedObject:
    """One object pose and conservative inflation at a requested time."""

    object_id: str
    local_sdf: dict[str, Any]
    position: np.ndarray
    orientation_xyzw: np.ndarray
    inflation_m: float


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise DynamicReplayError(f"{name} must be a number") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise DynamicReplayError(f"{name} is outside its valid range")
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise DynamicReplayError(f"{name} must contain {size} finite numbers")
    return array


def _load_trajectory(path: Path) -> TrajectoryData:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "positions" in data:
                positions = np.asarray(data["positions"], dtype=np.float64)
            else:
                schema_version = int(
                    np.asarray(data["artifact_schema_version"]).item()
                )
                if schema_version != 2:
                    raise DynamicReplayError(
                        f"{path}: omitted positions require trajectory schema v2"
                    )
                best_index = int(
                    np.asarray(data["best_trajectory_topk_index"]).item()
                )
                topk_positions = np.asarray(data["topk_positions"], dtype=np.float64)
                if best_index < 0 or best_index >= len(topk_positions):
                    raise DynamicReplayError(
                        f"{path}: best trajectory top-K index is out of range"
                    )
                positions = topk_positions[best_index]
            times_s = np.asarray(data["time_from_start"], dtype=np.float64)
    except DynamicReplayError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise DynamicReplayError(f"cannot load trajectory {path}: {error}") from error
    if positions.ndim != 2 or positions.shape[0] < 2 or positions.shape[1] not in (7, 9):
        raise DynamicReplayError(f"{path}: positions must have shape [H,7] or [H,9]")
    if times_s.shape != (positions.shape[0],) or not np.isfinite(positions).all():
        raise DynamicReplayError(f"{path}: trajectory arrays are inconsistent")
    if not np.isfinite(times_s).all() or times_s[0] < 0.0 or np.any(np.diff(times_s) <= 0.0):
        raise DynamicReplayError(f"{path}: time_from_start must be finite and strictly increasing")
    return TrajectoryData(positions=positions, times_s=times_s)


def _validate_local_sdf(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DynamicReplayError(f"{name} must be an object")
    shape_type = value.get("type")
    if shape_type == "sphere":
        _finite(value.get("radius"), f"{name}.radius", minimum=1e-9)
    elif shape_type == "box":
        size = _vector(value.get("size_xyz"), 3, f"{name}.size_xyz")
        if np.any(size <= 0.0):
            raise DynamicReplayError(f"{name}.size_xyz must be positive")
    elif shape_type == "capsule":
        _finite(value.get("radius"), f"{name}.radius", minimum=1e-9)
        _finite(value.get("length"), f"{name}.length", minimum=1e-9)
    else:
        raise DynamicReplayError(f"{name}.type must be sphere, box, or capsule")
    return dict(value)


def _validate_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
        raise DynamicReplayError(f"{name}.id must be a non-empty string")
    local_sdf = _validate_local_sdf(value.get("local_sdf"), f"{name}.local_sdf")
    pose = value.get("pose")
    if not isinstance(pose, dict):
        raise DynamicReplayError(f"{name}.pose must be an object")
    position = _vector(pose.get("position"), 3, f"{name}.pose.position")
    orientation = _vector(
        pose.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
        4,
        f"{name}.pose.orientation_xyzw",
    )
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-9:
        raise DynamicReplayError(f"{name}.pose.orientation_xyzw has zero norm")
    velocity = _vector(value.get("linear_velocity", [0.0, 0.0, 0.0]), 3, f"{name}.linear_velocity")
    covariance = np.asarray(value.get("covariance_6x6", [0.0] * 36), dtype=np.float64)
    if covariance.shape == (36,):
        covariance = covariance.reshape(6, 6)
    if covariance.shape != (6, 6) or not np.isfinite(covariance).all():
        raise DynamicReplayError(f"{name}.covariance_6x6 must contain 36 finite numbers")
    inflation = value.get("inflation", {})
    if not isinstance(inflation, dict) or inflation.get("mode", "linear") not in ("linear", "covariance"):
        raise DynamicReplayError(f"{name}.inflation.mode must be linear or covariance")
    base = _finite(inflation.get("base_m", 0.0), f"{name}.inflation.base_m", minimum=0.0)
    rate = _finite(
        inflation.get("horizon_rate_m_s", 0.0),
        f"{name}.inflation.horizon_rate_m_s",
        minimum=0.0,
    )
    return {
        "id": value["id"],
        "local_sdf": local_sdf,
        "position": position,
        "orientation_xyzw": orientation / norm,
        "linear_velocity": velocity,
        "covariance_6x6": covariance,
        "inflation_mode": inflation.get("mode", "linear"),
        "base_inflation_m": base,
        "horizon_inflation_rate_m_s": rate,
        "covariance_sigma": _finite(value.get("covariance_sigma", 3.0), f"{name}.covariance_sigma", minimum=0.0),
        "process_acceleration_std_m_s2": _finite(
            value.get("process_acceleration_std_m_s2", 0.01),
            f"{name}.process_acceleration_std_m_s2",
            minimum=0.0,
        ),
    }


def load_dynamic_replay_manifest(path: str | Path) -> DynamicReplayManifest:
    """Load and validate a replay manifest and all referenced NPZ trajectories."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DynamicReplayError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or raw.get("schema_version") != SCHEMA_VERSION:
        raise DynamicReplayError(f"manifest must use {SCHEMA!r} schema version {SCHEMA_VERSION}")
    frame_id = raw.get("frame_id", "fr3_link0")
    if not isinstance(frame_id, str) or not frame_id:
        raise DynamicReplayError("frame_id must be a non-empty string")
    env_name = raw.get("env_name", "EnvWarehouseExtraObjectsV00")
    if not isinstance(env_name, str):
        raise DynamicReplayError("env_name must be a string")
    static_scene = raw.get("static_scene", {})
    if not isinstance(static_scene, dict):
        raise DynamicReplayError("static_scene must be an object")

    plans_raw = raw.get("plans")
    if not isinstance(plans_raw, list) or not plans_raw:
        raise DynamicReplayError("plans must be a non-empty list")
    plans = []
    plan_ids = set()
    for index, item in enumerate(plans_raw):
        name = f"plans[{index}]"
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
            raise DynamicReplayError(f"{name}.id must be a non-empty string")
        plan_id = item["id"]
        if plan_id in plan_ids:
            raise DynamicReplayError(f"duplicate plan id {plan_id!r}")
        plan_ids.add(plan_id)
        trajectory_value = item.get("trajectory")
        if not isinstance(trajectory_value, str) or not trajectory_value:
            raise DynamicReplayError(f"{name}.trajectory must be a path string")
        trajectory_path = (manifest_path.parent / trajectory_value).resolve()
        trajectory = _load_trajectory(trajectory_path)
        created_s = _finite(item.get("created_s"), f"{name}.created_s", minimum=0.0)
        start_s = _finite(item.get("start_s"), f"{name}.start_s", minimum=0.0)
        status = item.get("status")
        if status not in PLAN_STATUSES:
            raise DynamicReplayError(f"{name}.status must be one of {sorted(PLAN_STATUSES)}")
        active_from_s = item.get("active_from_s")
        active_until_s = item.get("active_until_s")
        if active_from_s is not None:
            active_from_s = _finite(active_from_s, f"{name}.active_from_s", minimum=0.0)
        if active_until_s is not None:
            active_until_s = _finite(active_until_s, f"{name}.active_until_s", minimum=0.0)
        if (active_from_s is None) != (active_until_s is None):
            raise DynamicReplayError(f"{name} must provide both active interval endpoints or neither")
        if status == "rejected" and active_from_s is not None:
            raise DynamicReplayError(f"{name}: rejected plans cannot have an active interval")
        if active_from_s is not None:
            if (
                active_until_s <= active_from_s
                or active_from_s < start_s
                or active_until_s > start_s + trajectory.duration_s
            ):
                raise DynamicReplayError(f"{name}: active interval is outside the trajectory duration")
        segments = []
        for segment_index, segment in enumerate(item.get("collision_segments_s", [])):
            values = _vector(segment, 2, f"{name}.collision_segments_s[{segment_index}]")
            begin, end = float(values[0]), float(values[1])
            if begin < 0.0 or end <= begin or end > trajectory.duration_s:
                raise DynamicReplayError(f"{name}: collision segment is outside the trajectory duration")
            segments.append((begin, end))
        plans.append(
            PlanRecord(
                plan_id=plan_id,
                trajectory_path=trajectory_path,
                trajectory=trajectory,
                created_s=created_s,
                start_s=start_s,
                status=status,
                active_from_s=active_from_s,
                active_until_s=active_until_s,
                collision_segments_s=tuple(segments),
            )
        )
    plans.sort(key=lambda plan: (plan.created_s, plan.plan_id))
    active_plans = sorted(
        (plan for plan in plans if plan.active_from_s is not None),
        key=lambda plan: plan.active_from_s,
    )
    for previous, current in zip(active_plans, active_plans[1:]):
        if current.active_from_s < previous.active_until_s:
            raise DynamicReplayError(f"active intervals overlap for {previous.plan_id!r} and {current.plan_id!r}")

    snapshots_raw = raw.get("world_snapshots")
    if not isinstance(snapshots_raw, list) or not snapshots_raw:
        raise DynamicReplayError("world_snapshots must be a non-empty list")
    snapshots = []
    last_world_version = -1
    last_snapshot_time_s = -1.0
    for index, item in enumerate(snapshots_raw):
        name = f"world_snapshots[{index}]"
        if not isinstance(item, dict):
            raise DynamicReplayError(f"{name} must be an object")
        time_s = _finite(item.get("time_s"), f"{name}.time_s", minimum=0.0)
        valid_until_s = _finite(item.get("valid_until_s"), f"{name}.valid_until_s", minimum=time_s)
        if index == 0 and time_s != 0.0:
            raise DynamicReplayError("the first world snapshot must start at time_s=0")
        if time_s <= last_snapshot_time_s:
            raise DynamicReplayError("world snapshot time_s must increase in manifest order")
        if valid_until_s <= time_s:
            raise DynamicReplayError(f"{name}.valid_until_s must be later than time_s")
        last_snapshot_time_s = time_s
        version = item.get("world_version")
        if isinstance(version, bool) or not isinstance(version, int) or version <= last_world_version:
            raise DynamicReplayError("world_version must increase in manifest order")
        last_world_version = version
        objects_raw = item.get("objects", [])
        if not isinstance(objects_raw, list):
            raise DynamicReplayError(f"{name}.objects must be a list")
        objects = tuple(_validate_object(value, f"{name}.objects[{i}]") for i, value in enumerate(objects_raw))
        object_ids = [value["id"] for value in objects]
        if len(object_ids) != len(set(object_ids)):
            raise DynamicReplayError(f"{name} contains duplicate object ids")
        snapshots.append(WorldSnapshot(time_s, version, valid_until_s, objects))
    events = []
    for index, item in enumerate(raw.get("events", [])):
        name = f"events[{index}]"
        if not isinstance(item, dict) or item.get("type") not in EVENT_TYPES:
            raise DynamicReplayError(f"{name}.type must be handoff or brake")
        plan_id = item.get("plan_id")
        if plan_id is not None and plan_id not in plan_ids:
            raise DynamicReplayError(f"{name}.plan_id references an unknown plan")
        position = item.get("position")
        position_tuple = None if position is None else tuple(_vector(position, 3, f"{name}.position").tolist())
        reason = item.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise DynamicReplayError(f"{name}.reason must be a string")
        events.append(
            ReplayEvent(
                event_type=item["type"],
                time_s=_finite(item.get("time_s"), f"{name}.time_s", minimum=0.0),
                duration_s=_finite(item.get("duration_s", 0.8), f"{name}.duration_s", minimum=0.0),
                plan_id=plan_id,
                position=position_tuple,
                reason=reason,
            )
        )
    events.sort(key=lambda event: event.time_s)

    default_duration = max(
        max(plan.trajectory_end_s for plan in plans),
        max((event.time_s + event.duration_s for event in events), default=0.0),
    )
    duration_s = _finite(raw.get("duration_s", default_duration), "duration_s", minimum=1e-9)
    for current, following in zip(snapshots, snapshots[1:]):
        if current.valid_until_s < following.time_s:
            raise DynamicReplayError(f"world snapshot {current.world_version} expires before the next snapshot")
    if snapshots[-1].valid_until_s < duration_s:
        raise DynamicReplayError("the final world snapshot does not cover duration_s")
    initial_q_raw = raw.get("initial_q")
    initial_q = (
        plans[0].trajectory.positions[0].copy()
        if initial_q_raw is None
        else _vector(
            initial_q_raw,
            plans[0].trajectory.positions.shape[1],
            "initial_q",
        )
    )
    if any(plan.trajectory.positions.shape[1] != initial_q.shape[0] for plan in plans):
        raise DynamicReplayError("all plans must use the same trajectory DoF")
    return DynamicReplayManifest(
        path=manifest_path,
        frame_id=frame_id,
        env_name=env_name,
        duration_s=duration_s,
        static_scene=dict(static_scene),
        plans=tuple(plans),
        world_snapshots=tuple(snapshots),
        events=tuple(events),
        initial_q=initial_q,
    )


def active_plan_at(manifest: DynamicReplayManifest, time_s: float) -> PlanRecord | None:
    """Return the newest plan whose recorded active interval contains ``time_s``."""

    active = [
        plan
        for plan in manifest.plans
        if plan.active_from_s is not None and plan.active_from_s <= time_s < plan.active_until_s
    ]
    return max(active, key=lambda plan: (plan.active_from_s, plan.created_s), default=None)


def latest_pending_plan_at(manifest: DynamicReplayManifest, time_s: float) -> PlanRecord | None:
    """Return the newest accepted plan waiting for its future handoff."""

    pending = [
        plan
        for plan in manifest.plans
        if plan.status in ("accepted", "superseded")
        and plan.created_s <= time_s
        and plan.active_from_s is not None
        and time_s < plan.active_from_s
    ]
    return max(pending, key=lambda plan: (plan.created_s, plan.plan_id), default=None)


def trajectory_position_at(plan: PlanRecord, time_s: float) -> np.ndarray:
    """Interpolate one plan at replay-relative time ``time_s``."""

    local_time = np.clip(time_s - plan.start_s, plan.trajectory.times_s[0], plan.trajectory.times_s[-1])
    return np.asarray(
        [
            np.interp(local_time, plan.trajectory.times_s, plan.trajectory.positions[:, index])
            for index in range(plan.trajectory.positions.shape[1])
        ],
        dtype=np.float64,
    )


def robot_position_at(manifest: DynamicReplayManifest, time_s: float) -> np.ndarray:
    """Return the recorded commanded robot configuration at ``time_s``."""

    active = active_plan_at(manifest, time_s)
    if active is not None:
        return trajectory_position_at(active, time_s)
    completed = [plan for plan in manifest.plans if plan.active_until_s is not None and plan.active_until_s <= time_s]
    if completed:
        last = max(completed, key=lambda plan: plan.active_until_s)
        return trajectory_position_at(last, last.active_until_s)
    return manifest.initial_q.copy()


def world_snapshot_at(manifest: DynamicReplayManifest, time_s: float) -> WorldSnapshot:
    """Return the newest recorded world snapshot not newer than ``time_s``."""

    eligible = [snapshot for snapshot in manifest.world_snapshots if snapshot.time_s <= time_s]
    return eligible[-1] if eligible else manifest.world_snapshots[0]


def predict_object(item: dict[str, Any], snapshot_time_s: float, query_time_s: float) -> PredictedObject:
    """Apply the Phase-4 constant-velocity and inflation model."""

    dt = max(0.0, float(query_time_s) - float(snapshot_time_s))
    position = item["position"] + item["linear_velocity"] * dt
    if item["inflation_mode"] == "linear":
        inflation = item["base_inflation_m"] + item["horizon_inflation_rate_m_s"] * dt
    else:
        covariance = item["covariance_6x6"]
        p_pp = covariance[:3, :3]
        p_pv = covariance[:3, 3:]
        p_vp = covariance[3:, :3]
        p_vv = covariance[3:, 3:]
        propagated = p_pp + dt * (p_pv + p_vp) + dt * dt * p_vv
        process_variance = item["process_acceleration_std_m_s2"] ** 2
        propagated = propagated + np.eye(3) * process_variance * dt**3 / 3.0
        eigenvalue = max(0.0, float(np.linalg.eigvalsh(propagated).max()))
        inflation = item["base_inflation_m"] + item["covariance_sigma"] * math.sqrt(eigenvalue)
    return PredictedObject(
        object_id=item["id"],
        local_sdf=item["local_sdf"],
        position=np.asarray(position, dtype=np.float64),
        orientation_xyzw=np.asarray(item["orientation_xyzw"], dtype=np.float64),
        inflation_m=float(inflation),
    )


def plan_base_color(plan: PlanRecord, manifest: DynamicReplayManifest, time_s: float) -> tuple[float, ...] | None:
    """Return the requested gray/blue/green/red plan color at ``time_s``."""

    if time_s < plan.created_s:
        return None
    if plan.status in ("rejected", "braking"):
        return COLOR_RED
    active = active_plan_at(manifest, time_s)
    if active is not None and active.plan_id == plan.plan_id:
        return COLOR_BLUE
    pending = latest_pending_plan_at(manifest, time_s)
    if pending is not None and pending.plan_id == plan.plan_id:
        return COLOR_GREEN
    return COLOR_GRAY


def segment_color(
    plan: PlanRecord,
    segment_start_s: float,
    base_color: tuple[float, ...],
) -> tuple[float, ...]:
    """Override collision/rejection trajectory segments in red."""

    if plan.status in ("rejected", "braking"):
        return COLOR_RED
    if any(begin <= segment_start_s < end for begin, end in plan.collision_segments_s):
        return COLOR_RED
    return base_color


def brake_event_at(manifest: DynamicReplayManifest, time_s: float) -> ReplayEvent | None:
    """Return the newest brake event whose flash window contains ``time_s``."""

    events = [
        event
        for event in manifest.events
        if event.event_type == "brake" and event.time_s <= time_s < event.time_s + event.duration_s
    ]
    return events[-1] if events else None
