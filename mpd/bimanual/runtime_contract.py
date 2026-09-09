"""Strict JSON contract shared by Marvin MPD, Isaac Lab, and ROS 2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


JOINT_NAMES = tuple(
    [f"Joint{i}_L" for i in range(1, 8)]
    + [f"Joint{i}_R" for i in range(1, 8)]
)
TASK_MODES = {"left_only", "right_only", "dual_independent", "cooperative_rigid"}
RUNTIME_MODES = {
    "snapshot_no_time",
    "fixed_time_dynamic",
    "inference_time_optimized",
}
SCHEMA = "marvin_bimanual_request/v2"
RESULT_SCHEMA = "marvin_bimanual_result/v2"
SUCCESS_STATUS = "success"
RESULT_STATUSES = {
    SUCCESS_STATUS,
    "no_valid_trajectory",
    "invalid_request",
    "stale",
    "deadline_exceeded",
    "fault",
}
ROBOT_MODEL = "marvin_bimanual"
PLANNING_FRAME = "world"
SCENE_ID = "EnvWarehouseMarvinBimanual"
SCENE_VERSION = "marvin_warehouse_v2"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ContractError(ValueError):
    """The request or result does not satisfy the public runtime contract."""


def _vector(value: Any, size: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ContractError(f"{field} must contain exactly {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ContractError(f"{field} contains NaN or Inf")
    return result


def _optional_vector(value: Any, size: int, field: str) -> tuple[float, ...] | None:
    return None if value is None else _vector(value, size, field)


def _sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise ContractError(f"{field} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _pose(value: Any, field: str, planning_frame: str) -> tuple[float, ...]:
    pose = value if isinstance(value, Mapping) else None
    if pose is None:
        raise ContractError(f"{field} must be an object")
    frame_id = pose.get("frame_id", planning_frame)
    if frame_id != planning_frame:
        raise ContractError(f"{field}.frame_id must equal planning_frame={planning_frame!r}")
    if "pose_xyzw" in pose:
        values = _vector(pose["pose_xyzw"], 7, f"{field}.pose_xyzw")
        position, orientation = values[:3], values[3:]
    else:
        # Compatibility spelling for hand-written requests. Serialization
        # always emits pose_xyzw so there is one canonical wire format.
        position = _vector(pose.get("position"), 3, f"{field}.position")
        orientation = _vector(
            pose.get("orientation_xyzw"), 4, f"{field}.orientation_xyzw"
        )
    norm = math.sqrt(sum(item * item for item in orientation))
    if norm < 1e-9:
        raise ContractError(f"{field} quaternion has zero norm")
    return position + tuple(item / norm for item in orientation)


def _pose_dict(values: tuple[float, ...] | None, frame_id: str) -> dict[str, Any] | None:
    if values is None:
        return None
    return {"frame_id": frame_id, "pose_xyzw": list(values)}


@dataclass(frozen=True)
class BimanualRequest:
    request_id: str
    task_mode: str
    q_start: tuple[float, ...]
    q_goal: tuple[float, ...] | None
    joint_names: tuple[str, ...] = JOINT_NAMES
    q_velocity_start: tuple[float, ...] = (0.0,) * 14
    q_acceleration_start: tuple[float, ...] = (0.0,) * 14
    left_goal_pose: tuple[float, ...] | None = None
    right_goal_pose: tuple[float, ...] | None = None
    object_goal_pose: tuple[float, ...] | None = None
    grasp_profile: str | None = None
    robot_model: str = ROBOT_MODEL
    planning_frame: str = PLANNING_FRAME
    scene_id: str = SCENE_ID
    scene_version: str = SCENE_VERSION
    scene_hash: str | None = None
    robot_model_hash: str | None = None
    checkpoint_hash: str | None = None
    scene: Mapping[str, Any] | None = None
    world_version: int = 0
    deadline_unix_ns: int = 0
    runtime_mode: str = "snapshot_no_time"
    seed: int = 12345

    @property
    def active_ee_mask(self) -> tuple[float, float]:
        if self.task_mode == "left_only":
            return (1.0, 0.0)
        if self.task_mode == "right_only":
            return (0.0, 1.0)
        return (1.0, 1.0)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BimanualRequest":
        if not isinstance(value, Mapping):
            raise ContractError("request must be a JSON object")
        if value.get("schema") != SCHEMA:
            raise ContractError(f"schema must be {SCHEMA}")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ContractError("request_id must be a non-empty string")
        mode = value.get("task_mode")
        if mode not in TASK_MODES:
            raise ContractError(f"task_mode must be one of {sorted(TASK_MODES)}")
        names = tuple(value.get("joint_names", ()))
        if names != JOINT_NAMES:
            raise ContractError("joint_names must match Marvin's canonical left-then-right order")
        robot_model = value.get("robot_model")
        if robot_model != ROBOT_MODEL:
            raise ContractError(f"robot_model must be {ROBOT_MODEL!r}")
        planning_frame = value.get("planning_frame")
        if planning_frame != PLANNING_FRAME:
            raise ContractError(f"planning_frame must be {PLANNING_FRAME!r}")
        scene_id = value.get("scene_id")
        if scene_id != SCENE_ID:
            raise ContractError(f"scene_id must be {SCENE_ID!r}")
        scene_version = value.get("scene_version")
        if scene_version != SCENE_VERSION:
            raise ContractError(f"scene_version must be {SCENE_VERSION!r}")

        q_start = _vector(value.get("q_start"), 14, "q_start")
        q_goal = _optional_vector(value.get("q_goal"), 14, "q_goal")
        zero = (0.0,) * 14
        q_velocity_start = _vector(
            value.get("q_velocity_start", zero), 14, "q_velocity_start"
        )
        q_acceleration_start = _vector(
            value.get("q_acceleration_start", zero), 14, "q_acceleration_start"
        )

        runtime_mode = value.get("runtime_mode", "snapshot_no_time")
        if runtime_mode not in RUNTIME_MODES:
            raise ContractError(f"runtime_mode must be one of {sorted(RUNTIME_MODES)}")
        world_version = value.get("world_version", 0)
        deadline = value.get("deadline_unix_ns", 0)
        seed = value.get("seed", 12345)
        for item, field, minimum, maximum in (
            (world_version, "world_version", 0, 2**64 - 1),
            (deadline, "deadline_unix_ns", 0, 2**63 - 1),
            (seed, "seed", 0, 2**32 - 1),
        ):
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or not minimum <= item <= maximum
            ):
                raise ContractError(f"{field} must be an integer in [{minimum}, {maximum}]")

        scene = value.get("scene", {})
        if not isinstance(scene, Mapping):
            raise ContractError("scene must be an object")
        poses = {
            key: (
                None
                if value.get(key) is None
                else _pose(value[key], key, planning_frame)
            )
            for key in ("left_goal_pose", "right_goal_pose", "object_goal_pose")
        }
        if mode == "dual_independent" and q_goal is None and (
            poses["left_goal_pose"] is None or poses["right_goal_pose"] is None
        ):
            raise ContractError(
                "dual_independent requires q_goal or both left_goal_pose and right_goal_pose"
            )
        if mode == "left_only" and q_goal is None and poses["left_goal_pose"] is None:
            raise ContractError("left_only requires q_goal or left_goal_pose")
        if mode == "right_only" and q_goal is None and poses["right_goal_pose"] is None:
            raise ContractError("right_only requires q_goal or right_goal_pose")
        if mode == "cooperative_rigid" and poses["object_goal_pose"] is None and q_goal is None:
            raise ContractError("cooperative_rigid requires object_goal_pose or q_goal")

        grasp_profile = value.get("grasp_profile")
        if grasp_profile is not None and (
            not isinstance(grasp_profile, str) or not grasp_profile.strip()
        ):
            raise ContractError("grasp_profile must be a non-empty string when provided")

        return cls(
            request_id=request_id,
            task_mode=mode,
            q_start=q_start,
            q_goal=q_goal,
            joint_names=names,
            q_velocity_start=q_velocity_start,
            q_acceleration_start=q_acceleration_start,
            left_goal_pose=poses["left_goal_pose"],
            right_goal_pose=poses["right_goal_pose"],
            object_goal_pose=poses["object_goal_pose"],
            grasp_profile=grasp_profile,
            robot_model=robot_model,
            planning_frame=planning_frame,
            scene_id=scene_id,
            scene_version=scene_version,
            scene_hash=_sha256(value.get("scene_hash"), "scene_hash"),
            robot_model_hash=_sha256(value.get("robot_model_hash"), "robot_model_hash"),
            checkpoint_hash=_sha256(value.get("checkpoint_hash"), "checkpoint_hash"),
            scene=scene,
            world_version=world_version,
            deadline_unix_ns=deadline,
            runtime_mode=runtime_mode,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "request_id": self.request_id,
            "task_mode": self.task_mode,
            "runtime_mode": self.runtime_mode,
            "robot_model": self.robot_model,
            "planning_frame": self.planning_frame,
            "joint_names": list(self.joint_names),
            "q_start": list(self.q_start),
            "q_goal": None if self.q_goal is None else list(self.q_goal),
            "q_velocity_start": list(self.q_velocity_start),
            "q_acceleration_start": list(self.q_acceleration_start),
            "left_goal_pose": _pose_dict(self.left_goal_pose, self.planning_frame),
            "right_goal_pose": _pose_dict(self.right_goal_pose, self.planning_frame),
            "object_goal_pose": _pose_dict(self.object_goal_pose, self.planning_frame),
            "grasp_profile": self.grasp_profile,
            "scene_id": self.scene_id,
            "scene_version": self.scene_version,
            "scene_hash": self.scene_hash,
            "robot_model_hash": self.robot_model_hash,
            "checkpoint_hash": self.checkpoint_hash,
            "scene": dict(self.scene or {}),
            "world_version": self.world_version,
            "deadline_unix_ns": self.deadline_unix_ns,
            "seed": self.seed,
        }


def validate_result(
    value: Mapping[str, Any], *, request: BimanualRequest | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("result must be an object")
    if value.get("schema") != RESULT_SCHEMA:
        raise ContractError(f"schema must be {RESULT_SCHEMA}")
    status = value.get("status")
    if status not in RESULT_STATUSES:
        raise ContractError("result.status is invalid")
    if request is not None and value.get("request_id") != request.request_id:
        raise ContractError("result request_id does not match request")
    if status != SUCCESS_STATUS:
        error = value.get("error")
        if not isinstance(error, Mapping) or not isinstance(error.get("message"), str):
            raise ContractError("failed result must contain error.message")
        return dict(value)

    names = tuple(value.get("joint_names", ()))
    if names != JOINT_NAMES:
        raise ContractError("result joint_names do not match canonical order")
    positions = value.get("positions")
    if not isinstance(positions, list) or len(positions) < 2:
        raise ContractError("result.positions must contain at least two trajectory points")
    for index, point in enumerate(positions):
        _vector(point, 14, f"positions[{index}]")
    times = _vector(value.get("time_from_start"), len(positions), "time_from_start")
    if abs(times[0]) > 1e-9 or any(
        right <= left for left, right in zip(times, times[1:])
    ):
        raise ContractError("time_from_start must start at zero and increase strictly")
    for key in ("velocities", "accelerations"):
        rows = value.get(key)
        if not isinstance(rows, list) or len(rows) != len(positions):
            raise ContractError(f"result.{key} must match positions")
        for index, point in enumerate(rows):
            _vector(point, 14, f"{key}[{index}]")
    if request is not None:
        if value.get("world_version") != request.world_version:
            raise ContractError("result world_version does not match request")
        start_error = max(
            abs(actual - expected)
            for actual, expected in zip(positions[0], request.q_start)
        )
        if start_error > 1e-5:
            raise ContractError(
                f"result start differs from request by {start_error:.6g} rad"
            )
    validation = value.get("validation")
    if not isinstance(validation, Mapping) or validation.get("valid") is not True:
        raise ContractError("successful result must contain validation.valid=true")
    return dict(value)
