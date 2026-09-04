"""Strict, JSON-compatible contract shared by offline MPD and ROS2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

JOINT_NAMES = tuple([f"Joint{i}_L" for i in range(1, 8)] + [f"Joint{i}_R" for i in range(1, 8)])
TASK_MODES = {"left_only", "right_only", "dual_independent", "cooperative_rigid"}
RUNTIME_MODES = {"snapshot_no_time", "fixed_time_dynamic", "inference_time_optimized"}
SCHEMA = "marvin_bimanual_request/v1"
RESULT_SCHEMA = "marvin_bimanual_result/v1"


class ContractError(ValueError):
    pass


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError(f"{field} must contain exactly {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ContractError(f"{field} contains NaN or Inf")
    return result


def _pose(value: Any, field: str) -> tuple[float, ...]:
    pose = value if isinstance(value, Mapping) else None
    if pose is None:
        raise ContractError(f"{field} must be an object")
    position = _vector(pose.get("position"), 3, f"{field}.position")
    orientation = _vector(pose.get("orientation_xyzw"), 4, f"{field}.orientation_xyzw")
    norm = math.sqrt(sum(item * item for item in orientation))
    if norm < 1e-9:
        raise ContractError(f"{field}.orientation_xyzw has zero norm")
    return position + tuple(item / norm for item in orientation)


@dataclass(frozen=True)
class BimanualRequest:
    request_id: str
    task_mode: str
    q_start: tuple[float, ...]
    q_goal: tuple[float, ...] | None
    joint_names: tuple[str, ...] = JOINT_NAMES
    left_goal_pose: tuple[float, ...] | None = None
    right_goal_pose: tuple[float, ...] | None = None
    object_goal_pose: tuple[float, ...] | None = None
    grasp_profile: str | None = None
    scene: Mapping[str, Any] = None
    world_version: int = 0
    deadline_monotonic_ns: int = 0
    runtime_mode: str = "snapshot_no_time"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BimanualRequest":
        if not isinstance(value, Mapping):
            raise ContractError("request must be a JSON object")
        if value.get("schema") not in (None, SCHEMA):
            raise ContractError(f"schema must be {SCHEMA}")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ContractError("request_id must be a non-empty string")
        mode = value.get("task_mode")
        if mode not in TASK_MODES:
            raise ContractError(f"task_mode must be one of {sorted(TASK_MODES)}")
        names = tuple(value.get("joint_names", JOINT_NAMES))
        if names != JOINT_NAMES:
            raise ContractError("joint_names must match Marvin's canonical left-then-right order")
        q_start = _vector(value.get("q_start"), 14, "q_start")
        q_goal = None if value.get("q_goal") is None else _vector(value.get("q_goal"), 14, "q_goal")
        runtime_mode = value.get("runtime_mode", "snapshot_no_time")
        if runtime_mode not in RUNTIME_MODES:
            raise ContractError(f"runtime_mode must be one of {sorted(RUNTIME_MODES)}")
        world_version = value.get("world_version", 0)
        deadline = value.get("deadline_monotonic_ns", 0)
        if isinstance(world_version, bool) or not isinstance(world_version, int) or world_version < 0:
            raise ContractError("world_version must be a non-negative integer")
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 0:
            raise ContractError("deadline_monotonic_ns must be a non-negative integer")
        scene = value.get("scene", {})
        if not isinstance(scene, Mapping):
            raise ContractError("scene must be an object")
        poses = {
            key: None if value.get(key) is None else _pose(value[key], key)
            for key in ("left_goal_pose", "right_goal_pose", "object_goal_pose")
        }
        if mode == "cooperative_rigid" and poses["object_goal_pose"] is None and q_goal is None:
            raise ContractError("cooperative_rigid requires object_goal_pose or q_goal")
        if mode == "dual_independent" and q_goal is None and poses["left_goal_pose"] is None and poses["right_goal_pose"] is None:
            raise ContractError("dual_independent requires q_goal or both arm goal poses")
        return cls(
            request_id=request_id,
            task_mode=mode,
            q_start=q_start,
            q_goal=q_goal,
            joint_names=names,
            left_goal_pose=poses["left_goal_pose"],
            right_goal_pose=poses["right_goal_pose"],
            object_goal_pose=poses["object_goal_pose"],
            grasp_profile=value.get("grasp_profile"),
            scene=scene,
            world_version=world_version,
            deadline_monotonic_ns=deadline,
            runtime_mode=runtime_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "request_id": self.request_id,
            "task_mode": self.task_mode,
            "joint_names": list(self.joint_names),
            "q_start": list(self.q_start),
            "q_goal": None if self.q_goal is None else list(self.q_goal),
            "left_goal_pose": self.left_goal_pose,
            "right_goal_pose": self.right_goal_pose,
            "object_goal_pose": self.object_goal_pose,
            "grasp_profile": self.grasp_profile,
            "scene": dict(self.scene or {}),
            "world_version": self.world_version,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
            "runtime_mode": self.runtime_mode,
        }


def validate_result(value: Mapping[str, Any], *, request: BimanualRequest | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("result must be an object")
    if value.get("schema") not in (None, RESULT_SCHEMA):
        raise ContractError(f"schema must be {RESULT_SCHEMA}")
    if value.get("status") not in {"ok", "error", "cancelled", "deadline_exceeded"}:
        raise ContractError("result.status is invalid")
    names = tuple(value.get("joint_names", ()))
    if names != JOINT_NAMES:
        raise ContractError("result joint_names do not match canonical order")
    positions = value.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ContractError("result.positions must be a non-empty trajectory")
    for index, point in enumerate(positions):
        _vector(point, 14, f"positions[{index}]")
    times = _vector(value.get("time_from_start"), len(positions), "time_from_start")
    if times[0] < 0 or any(right <= left for left, right in zip(times, times[1:])):
        raise ContractError("time_from_start must be strictly increasing and non-negative")
    if request is not None and value.get("request_id") not in (None, request.request_id):
        raise ContractError("result request_id does not match request")
    return dict(value)

