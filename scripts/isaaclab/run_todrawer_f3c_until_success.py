#!/usr/bin/env python3
"""Run one selectable ToDrawer planner mode per category until each succeeds.

A scenario succeeds only when the ROS log records reaching the goal and no
controlled-braking request occurred before that goal timestamp.  Failed
attempts are retained and retried with a new deterministic planner seed and a
new mode-timed obstacle realization (anchor, direction, speed, size, and
crossing time).  Every
attempt with a replay manifest is rendered in IsaacLab, including failed
attempts; only a successful attempt advances to the next scenario.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[2]
if SCRIPT_REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, SCRIPT_REPO_ROOT.as_posix())

from scripts.isaaclab.benchmark_todrawer_random import (
    BASE_CROSSINGS,
    CATEGORIES,
    DEFAULT_ANCHOR_JITTER_M,
    DEFAULT_FACTORIZED_C_CHECKPOINT,
    DEFAULT_FACTORIZED_TAU_R_CHECKPOINT,
    DENSE_ANCHOR_JITTER_M,
    FACTORIZED_MODE_SPECS,
    GENERATION_REVISION,
    MAX_GENERATION_RESAMPLE_ATTEMPTS,
    REPO_ROOT,
    SAFE_CONTROL_CROSSINGS,
    _random_object,
    generate_suite,
)
from scripts.isaaclab.analyze_todrawer_mode_motion_start import (
    measure_manifest_motion_start,
)
from scripts.isaaclab.todrawer_scenario_validation import (
    STARTUP_SAFE_ANCHOR_BOUNDS,
    initial_franka_trajectory_clearance,
    load_static_environment_boxes,
    static_interaction_clearance,
    validate_initial_franka_clearance,
    validate_trajectory_clearance,
)


PIPELINE = REPO_ROOT / "scripts/isaaclab/run_dynamic_demo_pipeline.sh"
REPLAY = REPO_ROOT / "scripts/isaaclab/replay_mpd_trajectory.py"
DEFAULT_AIRUNTIME_ROOT = Path("/home/eric/Projects/physical_ai_runtime")
DEFAULT_ISAACLAB_ROOT = Path("/home/eric/IsaacLab")
DEFAULT_ISAAC_PREFIX = Path("/home/eric/anaconda3/envs/env_isaaclab")
GOAL_PATTERN = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?)\].*goal reached; holding position"
)
BRAKE_PATTERN = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?)\].*controlled braking requested"
)
START_COLLISION_PATTERN = re.compile(
    r"\[(?P<time>\d+(?:\.\d+)?)\].*q_pos_start is in collision"
)
MAX_PLANNER_SEED = 2_147_483_647
DEFAULT_MODE = "f3_c"
SUPPORTED_MODES = (
    "phase4",
    "phase4_aligned",
    "joint",
    "f1_c",
    "f2_c",
    "f3_c",
    "f1_tau_r",
    "f2_tau_r",
    "f3_tau_r",
)


@dataclass(frozen=True)
class RunnerMode:
    name: str
    phase: str
    timing_mode: str | None = None
    factorized_method: str | None = None
    factorized_representation: str | None = None


@dataclass(frozen=True)
class AttemptAssessment:
    success: bool
    reason: str
    pipeline_returncode: int
    manifest_available: bool
    goal_reached: bool
    goal_timestamp_s: float | None
    brake_timestamps_s: list[float]
    brakes_before_goal: list[float]
    start_collision_timestamps_s: list[float]
    start_collisions_before_goal: list[float]
    premotion_audit_passed: bool | None
    measured_motion_start_from_world_s: float | None
    minimum_premotion_clearance_m: float | None
    minimum_premotion_clearance_object_id: str | None
    minimum_premotion_clearance_time_s: float | None
    premotion_audit_error: str | None


@dataclass(frozen=True)
class ModeTimingProfile:
    # Measured from existing successful replays at a 0.01 rad joint-motion
    # threshold. These are normal first-plan medians, not retry outliers.
    significant_motion_start_s: float
    expected_goal_s: float
    crossing_shift_min_s: float
    crossing_shift_max_s: float


MODE_TIMING_PROFILES = {
    "phase4": ModeTimingProfile(3.04, 12.70, 1.10, 1.40),
    "phase4_aligned": ModeTimingProfile(3.04, 12.70, 1.20, 1.50),
    "joint": ModeTimingProfile(4.02, 12.80, 1.65, 1.95),
    "f1_c": ModeTimingProfile(3.97, 8.80, 1.40, 1.70),
    "f2_c": ModeTimingProfile(3.98, 8.90, 1.50, 1.80),
    "f3_c": ModeTimingProfile(3.97, 8.90, 1.60, 1.90),
    "f1_tau_r": ModeTimingProfile(4.05, 10.20, 1.70, 2.00),
    "f2_tau_r": ModeTimingProfile(4.07, 10.80, 1.80, 2.10),
    "f3_tau_r": ModeTimingProfile(4.13, 11.00, 1.90, 2.20),
}
MINIMUM_CROSSING_AFTER_MOTION_START_S = 1.25
GOAL_CROSSING_RESERVE_S = 0.50
MINIMUM_INITIAL_FRANKA_CLEARANCE_M = 0.005
MODE_GENERATION_RESAMPLE_ATTEMPTS = 200
# Directions are compared as unoriented lines: d and -d describe the same
# motion line. A 35 degree gap is visibly meaningful and cannot be satisfied
# by the ordinary +/-0.18 rad direction jitter alone.
MINIMUM_DIRECTION_LINE_SEPARATION_RAD = math.radians(35.0)
FORCED_DIRECTION_LINE_SEPARATION_RANGE_RAD = (
    math.radians(50.0),
    math.radians(60.0),
)
ATTEMPT_GENERATION_REVISION = "actual-premotion-safe-distinct-direction-lines-v3"


def mode_contract(mode: str) -> RunnerMode:
    if mode == "phase4":
        return RunnerMode(mode, "phase4")
    if mode == "phase4_aligned":
        return RunnerMode(mode, "phase4_aligned")
    if mode == "joint":
        return RunnerMode(mode, "phase5", timing_mode="phase5_joint")
    if mode in SUPPORTED_MODES:
        method, representation = FACTORIZED_MODE_SPECS[mode]
        return RunnerMode(
            mode,
            "factorized",
            factorized_method=method,
            factorized_representation=representation,
        )
    raise ValueError(
        f"unsupported mode {mode!r}; expected one of {', '.join(SUPPORTED_MODES)}"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _run_streaming(command: list[str], *, cwd: Path, log_path: Path, env=None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end="", flush=True)
        return int(process.wait())


def _audit_actual_premotion_clearance(
    attempt_dir: Path,
) -> tuple[bool | None, float | None, float | None, str | None, float | None, str | None]:
    """Check every obstacle against parked Franka until measured robot motion."""

    scenario_path = attempt_dir / "scenario.json"
    manifest_path = attempt_dir / "episode/replay-manifest.json"
    if not scenario_path.is_file() or not manifest_path.is_file():
        return None, None, None, None, None, None
    try:
        motion = measure_manifest_motion_start(
            manifest_path,
            threshold_rad=0.01,
        )
        if motion is None:
            raise ValueError("replay has no measured 0.01 rad robot motion start")
        motion_start_s = float(motion["motion_start_from_world_s"])
        scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
        objects = list(scenario.get("objects", []))
        if not objects:
            raise ValueError("attempt scenario contains no dynamic objects")
        best_clearance = None
        best_object_id = None
        for item in objects:
            clearance = initial_franka_trajectory_clearance(
                item,
                end_s=motion_start_s,
            )
            if best_clearance is None or clearance.minimum_m < best_clearance.minimum_m:
                best_clearance = clearance
                best_object_id = str(item.get("id", "<object>"))
        assert best_clearance is not None
        return (
            best_clearance.minimum_m > MINIMUM_INITIAL_FRANKA_CLEARANCE_M,
            motion_start_s,
            best_clearance.minimum_m,
            best_object_id,
            best_clearance.time_s,
            None,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return False, None, None, None, None, str(error)


def assess_attempt(attempt_dir: Path, pipeline_returncode: int) -> AttemptAssessment:
    manifest_available = (attempt_dir / "episode/replay-manifest.json").is_file()
    ros_log = attempt_dir / "ros-replan.log"
    text = ros_log.read_text(encoding="utf-8", errors="replace") if ros_log.is_file() else ""
    goal_matches = list(GOAL_PATTERN.finditer(text))
    brake_timestamps = [float(match.group("time")) for match in BRAKE_PATTERN.finditer(text)]
    start_collision_timestamps = [
        float(match.group("time")) for match in START_COLLISION_PATTERN.finditer(text)
    ]
    goal_timestamp = float(goal_matches[0].group("time")) if goal_matches else None
    brakes_before_goal = (
        []
        if goal_timestamp is None
        else [timestamp for timestamp in brake_timestamps if timestamp <= goal_timestamp]
    )
    start_collisions_before_goal = (
        []
        if goal_timestamp is None
        else [
            timestamp
            for timestamp in start_collision_timestamps
            if timestamp <= goal_timestamp
        ]
    )
    (
        premotion_audit_passed,
        measured_motion_start_s,
        minimum_premotion_clearance_m,
        minimum_premotion_clearance_object_id,
        minimum_premotion_clearance_time_s,
        premotion_audit_error,
    ) = _audit_actual_premotion_clearance(attempt_dir)

    if pipeline_returncode != 0:
        reason = f"pipeline exited with status {pipeline_returncode}"
    elif not manifest_available:
        reason = "replay manifest missing"
    elif goal_timestamp is None:
        reason = "goal was not reached"
    elif start_collisions_before_goal:
        reason = "planner observed q_pos_start in collision before reaching the goal"
    elif premotion_audit_passed is False:
        if premotion_audit_error is not None:
            reason = f"pre-motion clearance audit failed: {premotion_audit_error}"
        else:
            reason = (
                "dynamic obstacle violated parked-Franka clearance before measured "
                "robot motion"
            )
    elif brakes_before_goal:
        reason = "controlled braking occurred before reaching the goal"
    else:
        reason = "goal reached with no earlier brake or collision safety violation"

    return AttemptAssessment(
        success=(
            pipeline_returncode == 0
            and manifest_available
            and goal_timestamp is not None
            and not start_collisions_before_goal
            and premotion_audit_passed is not False
            and not brakes_before_goal
        ),
        reason=reason,
        pipeline_returncode=int(pipeline_returncode),
        manifest_available=manifest_available,
        goal_reached=goal_timestamp is not None,
        goal_timestamp_s=goal_timestamp,
        brake_timestamps_s=brake_timestamps,
        brakes_before_goal=brakes_before_goal,
        start_collision_timestamps_s=start_collision_timestamps,
        start_collisions_before_goal=start_collisions_before_goal,
        premotion_audit_passed=premotion_audit_passed,
        measured_motion_start_from_world_s=measured_motion_start_s,
        minimum_premotion_clearance_m=minimum_premotion_clearance_m,
        minimum_premotion_clearance_object_id=(
            minimum_premotion_clearance_object_id
        ),
        minimum_premotion_clearance_time_s=minimum_premotion_clearance_time_s,
        premotion_audit_error=premotion_audit_error,
    )


def planner_seed(suite_seed: int, scenario_index: int, attempt_index: int) -> int:
    """Return a stable seed sequence; attempt_index is zero-based."""

    return int((suite_seed + 1009 * scenario_index + 104729 * attempt_index) % MAX_PLANNER_SEED)


def anchor_seed(suite_seed: int, scenario_index: int, attempt_index: int) -> int:
    """Return a deterministic seed independent of the planner seed sequence."""

    return int(
        (
            suite_seed
            + 15_485_863 * (scenario_index + 1)
            + 32_452_843 * (attempt_index + 1)
        )
        % MAX_PLANNER_SEED
    )


def resample_attempt_anchors(
    scenario: dict[str, Any],
    *,
    seed: int,
    anchor_jitter_m: float = DEFAULT_ANCHOR_JITTER_M,
    dense_anchor_jitter_m: float = DENSE_ANCHOR_JITTER_M,
) -> dict[str, Any]:
    """Copy a scenario and deterministically resample only its anchor positions."""

    if anchor_jitter_m < 0.0 or dense_anchor_jitter_m < 0.0:
        raise ValueError("anchor jitter ranges must be non-negative")
    sampled = copy.deepcopy(scenario)
    category = str(sampled["category"])
    half_range_m = (
        dense_anchor_jitter_m if category == "inflated_dense" else anchor_jitter_m
    )
    anchors = {
        str(spec["id"]): tuple(float(value) for value in spec["anchor"])
        for spec in BASE_CROSSINGS + SAFE_CONTROL_CROSSINGS
    }
    rng = random.Random(seed)
    static_boxes = load_static_environment_boxes()
    sampled_positions = []
    for item in sampled["objects"]:
        anchor_id = str(item["anchor_id"])
        if anchor_id not in anchors:
            raise ValueError(f"unknown anchor_id {anchor_id!r}")
        nominal = anchors[anchor_id]
        last_error: Exception | None = None
        for sample_index in range(1, MAX_GENERATION_RESAMPLE_ATTEMPTS + 1):
            candidate_position = [
                value + rng.uniform(-half_range_m, half_range_m) for value in nominal
            ]
            item["anchor_position"] = candidate_position
            try:
                clearance = validate_trajectory_clearance(item, static_boxes=static_boxes)
            except ValueError as error:
                last_error = error
                continue
            item["minimum_robot_base_clearance_m"] = clearance.robot_base_m
            item["minimum_static_environment_clearance_m"] = (
                clearance.static_environment_m
            )
            item["attempt_anchor_sample_index"] = sample_index
            sampled_positions.append(
                {
                    "object_id": item["id"],
                    "anchor_id": anchor_id,
                    "nominal_position": list(nominal),
                    "sampled_position": candidate_position,
                    "sample_index": sample_index,
                }
            )
            break
        else:
            raise RuntimeError(
                f"failed to sample a base-safe anchor for {item['id']} after "
                f"{MAX_GENERATION_RESAMPLE_ATTEMPTS} attempts: {last_error}"
            )
    sampled["attempt_anchor_sampling"] = {
        "mode": "uniform_per_axis_about_nominal_anchor",
        "seed": int(seed),
        "half_range_m": float(half_range_m),
        "positions": sampled_positions,
    }
    return sampled


def _sample_startup_safe_anchor(rng: random.Random, anchor_id: str) -> list[float]:
    try:
        lower, upper = STARTUP_SAFE_ANCHOR_BOUNDS[anchor_id]
    except KeyError as error:
        raise ValueError(f"unknown anchor_id {anchor_id!r}") from error
    return [rng.uniform(low, high) for low, high in zip(lower, upper)]


def direction_line_separation_rad(
    first: list[float] | tuple[float, ...],
    second: list[float] | tuple[float, ...],
) -> float:
    """Return the acute angle between two unoriented 3-D direction lines."""

    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm <= 1.0e-12 or second_norm <= 1.0e-12:
        raise ValueError("motion direction must be non-zero")
    cosine = sum(
        float(left) * float(right) for left, right in zip(first, second)
    ) / (first_norm * second_norm)
    # abs makes opposite vectors equivalent, as required for a direction line.
    return math.acos(min(1.0, max(0.0, abs(cosine))))


def _sample_distinct_xy_direction(
    rng: random.Random,
    reference: list[float] | tuple[float, ...],
) -> list[float]:
    """Sample an XY direction on a visibly different unoriented line."""

    reference_x, reference_y = float(reference[0]), float(reference[1])
    if math.hypot(reference_x, reference_y) <= 1.0e-12:
        raise ValueError("reference direction must have a non-zero XY component")
    reference_angle = math.atan2(reference_y, reference_x)
    separation = rng.uniform(*FORCED_DIRECTION_LINE_SEPARATION_RANGE_RAD)
    if rng.random() < 0.5:
        separation = -separation
    angle = reference_angle + separation
    direction = [math.cos(angle), math.sin(angle), 0.0]
    if rng.random() < 0.5:
        direction = [-value for value in direction]
    return direction


def _replace_primary_direction(item: dict[str, Any], direction: list[float]) -> None:
    """Replace a primary motion direction and keep curved motion orthogonal."""

    item["direction"] = direction
    motion = item.get("motion", {})
    if motion.get("type") in {"sinusoidal_curve", "curved_speed_variation"}:
        motion["lateral_direction"] = [-direction[1], direction[0], 0.0]


def _direction_line_contract(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe and verify the per-attempt unoriented direction-line contract."""

    required_lines = min(2, len(objects))
    best_pair: tuple[int, int] | None = None
    best_separation = 0.0
    for left in range(len(objects)):
        for right in range(left + 1, len(objects)):
            separation = direction_line_separation_rad(
                objects[left]["direction"], objects[right]["direction"]
            )
            if separation > best_separation:
                best_pair = (left, right)
                best_separation = separation
    satisfied = required_lines == 1 or (
        best_pair is not None
        and best_separation + 1.0e-12 >= MINIMUM_DIRECTION_LINE_SEPARATION_RAD
    )
    return {
        "required_distinct_lines": required_lines,
        "opposite_vectors_share_line": True,
        "minimum_separation_deg": math.degrees(
            MINIMUM_DIRECTION_LINE_SEPARATION_RAD
        ),
        "maximum_realized_pair_separation_deg": math.degrees(best_separation),
        "witness_object_ids": (
            []
            if best_pair is None
            else [objects[best_pair[0]]["id"], objects[best_pair[1]]["id"]]
        ),
        "satisfied": satisfied,
    }


def _success_record_uses_current_attempt_policy(record: dict[str, Any]) -> bool:
    sampling = record.get("attempt_sampling", {})
    line_contract = sampling.get("direction_line_contract", {})
    return bool(
        sampling.get("revision") == ATTEMPT_GENERATION_REVISION
        and line_contract.get("satisfied") is True
        and line_contract.get("opposite_vectors_share_line") is True
    )


def _require_current_attempt_policy(
    payload: dict[str, Any], *, expected_mode: str
) -> None:
    """Fail closed if an attempt was not materialized by the shared v3 policy."""

    if not _success_record_uses_current_attempt_policy(payload):
        revision = payload.get("attempt_sampling", {}).get("revision")
        raise RuntimeError(
            f"mode {expected_mode} produced attempt generation revision "
            f"{revision!r}; expected {ATTEMPT_GENERATION_REVISION!r}"
        )
    actual_mode = payload.get("mode_timing_profile", {}).get("mode")
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"mode {expected_mode} produced timing profile for {actual_mode!r}"
        )


def resample_mode_attempt_scenario(
    scenario: dict[str, Any],
    *,
    mode: str,
    seed: int,
    goal_reserve_s: float = GOAL_CROSSING_RESERVE_S,
    minimum_static_clearance_m: float | None = None,
) -> dict[str, Any]:
    """Create one mode-timed, startup-safe realization of a suite template."""

    if mode not in MODE_TIMING_PROFILES:
        raise ValueError(f"unsupported mode timing profile {mode!r}")
    if goal_reserve_s < 0.0:
        raise ValueError("goal reserve must be non-negative")
    if minimum_static_clearance_m is not None and minimum_static_clearance_m < 0.0:
        raise ValueError("minimum static clearance must be non-negative")
    profile = MODE_TIMING_PROFILES[mode]
    rng = random.Random(seed)
    sampled = copy.deepcopy(scenario)
    category = str(sampled["category"])
    original_objects = list(sampled["objects"])
    if not original_objects:
        raise ValueError("scenario contains no objects")

    original_times = [float(item["crossing_time_s"]) for item in original_objects]
    shift_lower = max(
        profile.crossing_shift_min_s,
        profile.significant_motion_start_s
        + MINIMUM_CROSSING_AFTER_MOTION_START_S
        - min(original_times),
    )
    shift_upper = min(
        profile.crossing_shift_max_s,
        profile.expected_goal_s - goal_reserve_s - max(original_times),
    )
    if shift_lower > shift_upper:
        raise ValueError(
            f"mode {mode} has no crossing-time interval for {sampled['id']}: "
            f"shift=[{shift_lower:.3f}, {shift_upper:.3f}]"
        )
    crossing_shift_s = rng.uniform(shift_lower, shift_upper)
    crossing_specs = BASE_CROSSINGS + SAFE_CONTROL_CROSSINGS
    corridor_index = {
        str(spec["id"]): index for index, spec in enumerate(crossing_specs)
    }
    static_boxes = load_static_environment_boxes()
    parked_franka_protection_until_s = profile.expected_goal_s - goal_reserve_s
    realized_objects = []
    nominal_directions = []
    for item in original_objects:
        spec = crossing_specs[corridor_index[str(item["anchor_id"])]]
        nominal_directions.append(spec["direction"])
    nominally_distinct = any(
        direction_line_separation_rad(left, right)
        >= MINIMUM_DIRECTION_LINE_SEPARATION_RAD
        for index, left in enumerate(nominal_directions)
        for right in nominal_directions[index + 1 :]
    )
    # If all nominal corridors are parallel, put the second orientation on an
    # outer A0/A2 lane instead of the central A1 lane.  This substantially
    # increases the chance that the complete line stays clear of parked Franka.
    diversity_object_index = None
    diversity_reference_direction = None
    if len(original_objects) >= 2 and not nominally_distinct:
        anchor_ids = [str(item["anchor_id"]) for item in original_objects]
        preferred_anchor = next(
            (anchor_id for anchor_id in ("A2", "A0") if anchor_id in anchor_ids),
            anchor_ids[0],
        )
        diversity_object_index = anchor_ids.index(preferred_anchor)
        reference_index = next(
            index
            for index in range(len(original_objects))
            if index != diversity_object_index
        )
        diversity_reference_direction = nominal_directions[reference_index]
    for object_index, original in enumerate(original_objects):
        crossing_time_s = float(original["crossing_time_s"]) + crossing_shift_s
        last_error: Exception | None = None
        for resample_index in range(1, MODE_GENERATION_RESAMPLE_ATTEMPTS + 1):
            item = _random_object(
                rng,
                scenario_index=int(str(sampled["id"]).rsplit("-", 1)[-1]),
                object_index=object_index,
                corridor_index=corridor_index[str(original["anchor_id"])],
                category=category,
                crossing_time_s=crossing_time_s,
                schedule_role=str(original["schedule_role"]),
                motion_type=str(original["motion_model"]),
            )
            item["anchor_position"] = _sample_startup_safe_anchor(
                rng, str(original["anchor_id"])
            )
            if object_index == diversity_object_index:
                assert diversity_reference_direction is not None
                _replace_primary_direction(
                    item,
                    _sample_distinct_xy_direction(
                        rng, diversity_reference_direction
                    ),
                )
            try:
                clearance = validate_trajectory_clearance(
                    item, static_boxes=static_boxes
                )
                if minimum_static_clearance_m is not None:
                    static_clearance = static_interaction_clearance(
                        item,
                        static_boxes=static_boxes,
                    )
                    if static_clearance <= minimum_static_clearance_m:
                        raise ValueError(
                            f"{item['id']} intersects or approaches static furniture "
                            f"in its interaction window: "
                            f"clearance={static_clearance:.6f}m <= "
                            f"{minimum_static_clearance_m:.6f}m"
                        )
                    item["minimum_static_interaction_clearance_m"] = (
                        static_clearance
                    )
                initial_clearance = validate_initial_franka_clearance(
                    item,
                    end_s=parked_franka_protection_until_s,
                    minimum_clearance_m=MINIMUM_INITIAL_FRANKA_CLEARANCE_M,
                )
            except ValueError as error:
                last_error = error
                continue
            item["minimum_robot_base_clearance_m"] = clearance.robot_base_m
            item["minimum_static_environment_clearance_m"] = (
                clearance.static_environment_m
            )
            item["minimum_initial_franka_clearance_m"] = (
                initial_clearance.minimum_m
            )
            item["minimum_initial_franka_clearance_time_s"] = (
                initial_clearance.time_s
            )
            item["minimum_initial_franka_clearance_sphere_index"] = (
                initial_clearance.sphere_index
            )
            item["generation_resample_attempts"] = resample_index
            realized_objects.append(item)
            break
        else:
            raise RuntimeError(
                f"failed to generate startup-safe {category} object "
                f"{object_index} for {mode} after "
                f"{MODE_GENERATION_RESAMPLE_ATTEMPTS} attempts: {last_error}"
            )

    direction_line_contract = _direction_line_contract(realized_objects)
    if not direction_line_contract["satisfied"]:
        raise RuntimeError(
            f"failed to realize two distinct unoriented motion lines for "
            f"{sampled['id']}"
        )
    sampled["objects"] = realized_objects
    sampled["anchor_schedule"] = [
        {
            "anchor_id": item["anchor_id"],
            "schedule_role": item["schedule_role"],
            "crossing_time_s": item["crossing_time_s"],
        }
        for item in realized_objects
    ]
    sampled["primary_crossing_window_s"] = [
        min(float(item["crossing_time_s"]) for item in realized_objects),
        max(float(item["crossing_time_s"]) for item in realized_objects),
    ]
    sampled["mode_timing_profile"] = {
        "mode": mode,
        "significant_motion_threshold_rad": 0.01,
        "significant_motion_start_s": profile.significant_motion_start_s,
        "expected_goal_s": profile.expected_goal_s,
        "goal_crossing_reserve_s": goal_reserve_s,
        "crossing_shift_s": crossing_shift_s,
        "crossing_shift_range_s": [shift_lower, shift_upper],
        "minimum_crossing_after_motion_start_s": (
            MINIMUM_CROSSING_AFTER_MOTION_START_S
        ),
    }
    sampled["attempt_sampling"] = {
        "revision": ATTEMPT_GENERATION_REVISION,
        "mode": "full_obstacle_resample_with_initial_franka_sweep_check",
        "seed": int(seed),
        "randomized_fields": [
            "anchor_position",
            "direction",
            "speed_m_s",
            "local_sdf.size_xyz",
            "crossing_time_s",
        ],
        "initial_franka_check_interval": "0..parked_franka_protection_until_s",
        "parked_franka_protection_until_s": parked_franka_protection_until_s,
        "minimum_initial_franka_clearance_m": (
            MINIMUM_INITIAL_FRANKA_CLEARANCE_M
        ),
        "minimum_static_environment_clearance_m": minimum_static_clearance_m,
        "maximum_generation_resample_attempts": (
            MODE_GENERATION_RESAMPLE_ATTEMPTS
        ),
        "direction_line_contract": direction_line_contract,
    }
    return sampled


def build_pipeline_command(
    *,
    mode: str,
    scenario_path: Path,
    attempt_dir: Path,
    seed: int,
    duration_sec: float,
    plan_rate_hz: float,
    factorized_c_checkpoint: Path,
    factorized_tau_r_checkpoint: Path,
) -> list[str]:
    contract = mode_contract(mode)
    command = [
        PIPELINE.as_posix(),
        "--profile",
        "to_drawer",
        "--phase",
        contract.phase,
        "--world-scenario-file",
        scenario_path.as_posix(),
        "--planner-seed",
        str(seed),
        "--duration-sec",
        str(duration_sec),
        "--plan-rate-hz",
        str(plan_rate_hz),
        "--output-dir",
        attempt_dir.as_posix(),
        "--allow-brake",
        "--skip-build",
        "--skip-render",
    ]
    if contract.timing_mode is not None:
        command.extend(("--timing-mode", contract.timing_mode))
    if contract.phase == "factorized":
        checkpoint = (
            factorized_c_checkpoint
            if contract.factorized_representation == "c"
            else factorized_tau_r_checkpoint
        )
        command.extend(
            (
                "--factorized-method",
                str(contract.factorized_method),
                "--factorized-timing-checkpoint",
                checkpoint.as_posix(),
                "--factorized-adapt-spatial-basis",
            )
        )
    return command


def build_render_command(
    *,
    isaaclab_root: Path,
    manifest: Path,
    video: Path,
    screenshot: Path,
    summary: Path,
    video_fps: float,
    width: int,
    height: int,
) -> list[str]:
    return [
        (isaaclab_root / "isaaclab.sh").as_posix(),
        "-p",
        REPLAY.as_posix(),
        "--manifest",
        manifest.as_posix(),
        "--output_video",
        video.as_posix(),
        "--screenshot_path",
        screenshot.as_posix(),
        "--output_json",
        summary.as_posix(),
        "--video_fps",
        str(video_fps),
        "--width",
        str(width),
        "--height",
        str(height),
        "--prediction_horizon_s",
        "3.0",
        "--prediction_samples",
        "10",
        "--enable_cameras",
    ]


def _next_attempt_index(run_dir: Path) -> int:
    indices = []
    for path in run_dir.glob("attempt-*"):
        try:
            indices.append(int(path.name[len("attempt-") :]))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def _materialize_suite(output_dir: Path, suite_seed: int) -> dict[str, Any]:
    suite_path = output_dir / "suite.json"
    if suite_path.is_file():
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        if (
            suite.get("suite_seed") != suite_seed
            or suite.get("scenario_count") != len(CATEGORIES)
            or suite.get("generation_policy", {}).get("revision")
            != GENERATION_REVISION
        ):
            raise ValueError(
                "existing suite.json does not match the requested seed/category count/"
                "generation revision; "
                "choose a new output directory"
            )
        return suite

    suite = generate_suite(len(CATEGORIES), suite_seed)
    for scenario in suite["scenarios"]:
        _write_json(output_dir / "scenarios" / f"{scenario['id']}.json", scenario)
    _write_json(suite_path, suite)
    return suite


def _build_once(airuntime_root: Path, output_dir: Path) -> int:
    return _run_streaming(
        ["pixi", "run", "build", "--packages-up-to", "mpd_dynamic_planner_adapter"],
        cwd=airuntime_root,
        log_path=output_dir / "ros-build.log",
    )


def _render_until_saved(
    *,
    command: list[str],
    attempt_dir: Path,
    video: Path,
    screenshot: Path,
    summary: Path,
    isaac_prefix: Path,
    retry_delay_sec: float,
    max_render_attempts: int,
) -> None:
    render_env = os.environ.copy()
    render_env.pop("PYTHONPATH", None)
    render_env.pop("LD_LIBRARY_PATH", None)
    render_env["CONDA_PREFIX"] = isaac_prefix.as_posix()
    render_attempt = 1
    while True:
        print(f"[render] attempt={render_attempt} video={video}", flush=True)
        status = _run_streaming(
            command,
            cwd=REPO_ROOT,
            log_path=attempt_dir / f"isaac-replay-attempt-{render_attempt:03d}.log",
            env=render_env,
        )
        if (
            status == 0
            and video.is_file()
            and video.stat().st_size > 0
            and screenshot.is_file()
            and summary.is_file()
        ):
            return
        if max_render_attempts and render_attempt >= max_render_attempts:
            raise RuntimeError(
                f"IsaacLab rendering did not succeed after {max_render_attempts} attempts"
            )
        render_attempt += 1
        time.sleep(retry_delay_sec)


def _publish_success_artifacts(
    *,
    attempt_video: Path,
    attempt_screenshot: Path,
    attempt_summary: Path,
    success_video: Path,
    success_screenshot: Path,
    success_summary: Path,
) -> None:
    """Copy the already-rendered successful replay to the stable video index."""

    success_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(attempt_video, success_video)
    shutil.copy2(attempt_screenshot, success_screenshot)
    shutil.copy2(attempt_summary, success_summary)


def _parser() -> argparse.ArgumentParser:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "scripts/isaaclab/logs/todrawer-f3c-until-success" / timestamp,
    )
    parser.add_argument("--suite-seed", type=int, default=20260829)
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default=DEFAULT_MODE,
        help=(
            "planner mode: phase4, phase4_aligned, joint, or one of the six "
            "learned factorized modes (default: f3_c)"
        ),
    )
    parser.add_argument(
        "--anchor-jitter-m",
        type=float,
        default=DEFAULT_ANCHOR_JITTER_M,
        help="legacy option retained for CLI compatibility; mode-safe sampling uses bounded work-volume anchors",
    )
    parser.add_argument(
        "--dense-anchor-jitter-m",
        type=float,
        default=DENSE_ANCHOR_JITTER_M,
        help="per-axis anchor half-range for inflated_dense (default: 0.020 m)",
    )
    parser.add_argument("--duration-sec", type=float, default=35.0)
    parser.add_argument("--plan-rate-hz", type=float, default=1.0)
    parser.add_argument(
        "--factorized-c-checkpoint", type=Path, default=DEFAULT_FACTORIZED_C_CHECKPOINT
    )
    parser.add_argument(
        "--factorized-tau-r-checkpoint",
        type=Path,
        default=DEFAULT_FACTORIZED_TAU_R_CHECKPOINT,
    )
    parser.add_argument("--retry-delay-sec", type=float, default=5.0)
    parser.add_argument(
        "--max-attempts-per-scenario",
        type=int,
        default=0,
        help="0 retries indefinitely (default); a positive value stops after that many attempts",
    )
    parser.add_argument(
        "--max-render-attempts",
        type=int,
        default=0,
        help="0 retries rendering indefinitely (default)",
    )
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--airuntime-root", type=Path, default=DEFAULT_AIRUNTIME_ROOT)
    parser.add_argument("--isaaclab-root", type=Path, default=DEFAULT_ISAACLAB_ROOT)
    parser.add_argument("--isaac-prefix", type=Path, default=DEFAULT_ISAAC_PREFIX)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write next-attempt scenarios and commands without running them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.duration_sec <= 0 or args.plan_rate_hz <= 0 or args.retry_delay_sec < 0:
        raise SystemExit("duration, plan rate, and retry delay must be valid positive values")
    if args.max_attempts_per_scenario < 0 or args.max_render_attempts < 0:
        raise SystemExit("maximum attempt counts must be non-negative")
    if args.anchor_jitter_m < 0 or args.dense_anchor_jitter_m < 0:
        raise SystemExit("anchor jitter ranges must be non-negative")

    output_dir = args.output_dir.expanduser().resolve()
    contract = mode_contract(args.mode)
    factorized_c_checkpoint = args.factorized_c_checkpoint.expanduser().resolve()
    factorized_tau_r_checkpoint = (
        args.factorized_tau_r_checkpoint.expanduser().resolve()
    )
    airuntime_root = args.airuntime_root.expanduser().resolve()
    isaaclab_root = args.isaaclab_root.expanduser().resolve()
    isaac_prefix = args.isaac_prefix.expanduser().resolve()
    if contract.factorized_representation == "c" and not factorized_c_checkpoint.is_file():
        raise SystemExit(
            f"factorized c checkpoint is not a file: {factorized_c_checkpoint}"
        )
    if (
        contract.factorized_representation == "tau_r"
        and not factorized_tau_r_checkpoint.is_file()
    ):
        raise SystemExit(
            "factorized tau_r checkpoint is not a file: "
            f"{factorized_tau_r_checkpoint}"
        )
    factorized_checkpoint = None
    if contract.factorized_representation == "c":
        factorized_checkpoint = factorized_c_checkpoint
    elif contract.factorized_representation == "tau_r":
        factorized_checkpoint = factorized_tau_r_checkpoint
    if not args.dry_run and not (isaaclab_root / "isaaclab.sh").is_file():
        raise SystemExit(f"IsaacLab launcher is missing: {isaaclab_root / 'isaaclab.sh'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    suite = _materialize_suite(output_dir, args.suite_seed)
    if not args.skip_build and not args.dry_run:
        if _build_once(airuntime_root, output_dir) != 0:
            return 1

    successful = []
    for scenario_index, scenario in enumerate(suite["scenarios"]):
        category = scenario["category"]
        scenario_id = scenario["id"]
        artifact_stem = f"{scenario_id}-{category}-{args.mode}"
        video = output_dir / "videos" / args.mode / f"{artifact_stem}.mp4"
        screenshot = output_dir / "videos" / args.mode / f"{artifact_stem}.png"
        replay_summary = output_dir / "videos" / args.mode / f"{artifact_stem}.json"
        success_path = output_dir / "successes" / args.mode / f"{scenario_id}.json"
        if success_path.is_file() and video.is_file() and video.stat().st_size > 0:
            record = json.loads(success_path.read_text(encoding="utf-8"))
            if _success_record_uses_current_attempt_policy(record):
                successful.append(record)
                print(
                    f"[skip] {scenario_id} category={category} mode={args.mode} "
                    "already has a successful video using the current attempt policy"
                )
                continue
            print(
                f"[rerun] {scenario_id} category={category} mode={args.mode} "
                f"has an old success that does not satisfy the shared "
                f"{ATTEMPT_GENERATION_REVISION} policy"
            )
        if args.mode == "f3_c":
            legacy_success_path = output_dir / "successes" / f"{scenario_id}.json"
            legacy_video = (
                output_dir / "videos" / f"{scenario_id}-{category}-f3-c.mp4"
            )
            if (
                legacy_success_path.is_file()
                and legacy_video.is_file()
                and legacy_video.stat().st_size > 0
            ):
                record = json.loads(legacy_success_path.read_text(encoding="utf-8"))
                if (
                    record.get("mode") == "f3_c"
                    and _success_record_uses_current_attempt_policy(record)
                ):
                    successful.append(record)
                    print(
                        f"[skip] {scenario_id} category={category} mode=f3_c "
                        "has a compatible legacy-path successful video"
                    )
                    continue

        run_dir = output_dir / "runs" / scenario_id / args.mode
        attempt_number = _next_attempt_index(run_dir)
        attempts_this_invocation = 0
        while True:
            attempt_index = attempt_number - 1
            seed = planner_seed(args.suite_seed, scenario_index, attempt_index)
            attempt_anchor_seed = anchor_seed(
                args.suite_seed, scenario_index, attempt_index
            )
            attempt_dir = run_dir / f"attempt-{attempt_number:03d}"
            attempt_scenario = resample_mode_attempt_scenario(
                scenario,
                mode=args.mode,
                seed=attempt_anchor_seed,
            )
            _require_current_attempt_policy(
                attempt_scenario,
                expected_mode=args.mode,
            )
            scenario_path = attempt_dir / "scenario.json"
            _write_json(scenario_path, attempt_scenario)
            command = build_pipeline_command(
                mode=args.mode,
                scenario_path=scenario_path,
                attempt_dir=attempt_dir,
                seed=seed,
                duration_sec=args.duration_sec,
                plan_rate_hz=args.plan_rate_hz,
                factorized_c_checkpoint=factorized_c_checkpoint,
                factorized_tau_r_checkpoint=factorized_tau_r_checkpoint,
            )
            run_spec = {
                "scenario_id": scenario_id,
                "category": category,
                "attempt": attempt_number,
                "planner_seed": seed,
                "anchor_seed": attempt_anchor_seed,
                "attempt_generation_revision": ATTEMPT_GENERATION_REVISION,
                "mode_timing_profile": attempt_scenario["mode_timing_profile"],
                "attempt_sampling": attempt_scenario["attempt_sampling"],
                "anchor_positions": {
                    item["id"]: item["anchor_position"]
                    for item in attempt_scenario["objects"]
                },
                "directions": {
                    item["id"]: item["direction"]
                    for item in attempt_scenario["objects"]
                },
                "scenario_path": scenario_path.as_posix(),
                "mode": args.mode,
                "phase": contract.phase,
                "timing_mode": contract.timing_mode,
                "factorized_method": contract.factorized_method,
                "factorized_representation": contract.factorized_representation,
                "factorized_timing_checkpoint": (
                    factorized_checkpoint.as_posix()
                    if factorized_checkpoint is not None
                    else None
                ),
                "factorized_spatial_basis_adapted": (
                    contract.phase == "factorized"
                ),
                "command": command,
            }
            _write_json(attempt_dir / "run-spec.json", run_spec)
            print(
                f"[run] {scenario_id} category={category} mode={args.mode} "
                f"attempt={attempt_number} planner_seed={seed} "
                f"geometry_seed={attempt_anchor_seed} "
                f"crossing_shift_s={attempt_scenario['mode_timing_profile']['crossing_shift_s']:.3f}",
                flush=True,
            )
            if args.dry_run:
                print("  " + " ".join(command))
                break

            status = _run_streaming(
                command,
                cwd=REPO_ROOT,
                log_path=attempt_dir / "pipeline.log",
            )
            assessment = assess_attempt(attempt_dir, status)
            print(f"[result] success={assessment.success} reason={assessment.reason}")
            manifest = attempt_dir / "episode/replay-manifest.json"
            attempt_video = attempt_dir / "replay.mp4"
            attempt_screenshot = attempt_dir / "replay-final.png"
            attempt_replay_summary = attempt_dir / "replay-summary.json"
            replay_rendered = False
            if assessment.manifest_available:
                render_command = build_render_command(
                    isaaclab_root=isaaclab_root,
                    manifest=manifest,
                    video=attempt_video,
                    screenshot=attempt_screenshot,
                    summary=attempt_replay_summary,
                    video_fps=args.video_fps,
                    width=args.width,
                    height=args.height,
                )
                _render_until_saved(
                    command=render_command,
                    attempt_dir=attempt_dir,
                    video=attempt_video,
                    screenshot=attempt_screenshot,
                    summary=attempt_replay_summary,
                    isaac_prefix=isaac_prefix,
                    retry_delay_sec=args.retry_delay_sec,
                    max_render_attempts=args.max_render_attempts,
                )
                replay_rendered = True
            else:
                print(
                    f"[replay] skipped {scenario_id} attempt={attempt_number}: "
                    "manifest is unavailable",
                    file=sys.stderr,
                )
            attempt_result = {
                **asdict(assessment),
                "replay_rendered": replay_rendered,
                "manifest": manifest.as_posix() if assessment.manifest_available else None,
                "video": attempt_video.as_posix() if replay_rendered else None,
                "screenshot": attempt_screenshot.as_posix() if replay_rendered else None,
                "replay_summary": (
                    attempt_replay_summary.as_posix() if replay_rendered else None
                ),
            }
            _write_json(attempt_dir / "attempt-result.json", attempt_result)
            if assessment.success:
                _publish_success_artifacts(
                    attempt_video=attempt_video,
                    attempt_screenshot=attempt_screenshot,
                    attempt_summary=attempt_replay_summary,
                    success_video=video,
                    success_screenshot=screenshot,
                    success_summary=replay_summary,
                )
                record = {
                    **run_spec,
                    "attempt_dir": attempt_dir.as_posix(),
                    "manifest": manifest.as_posix(),
                    "video": video.as_posix(),
                    "screenshot": screenshot.as_posix(),
                    "replay_summary": replay_summary.as_posix(),
                    "attempt_video": attempt_video.as_posix(),
                    "attempt_screenshot": attempt_screenshot.as_posix(),
                    "attempt_replay_summary": attempt_replay_summary.as_posix(),
                    "assessment": asdict(assessment),
                }
                _write_json(success_path, record)
                successful.append(record)
                summary_payload = {
                    "schema": "mpd_todrawer_until_success",
                    "schema_version": 3,
                    "mode": args.mode,
                    "attempt_generation_revision": ATTEMPT_GENERATION_REVISION,
                    "success_definition": (
                        "goal reached, no controlled brake or q-start collision at or "
                        "before goal, and parked-Franka clearance passed until measured "
                        "robot motion"
                    ),
                    "attempt_randomization": (
                        "planner seed and full obstacle realization vary per attempt; "
                        "multi-object attempts use at least two unoriented motion lines"
                    ),
                    "replay_policy": "render every attempt that has a manifest",
                    "successful": successful,
                }
                _write_json(output_dir / "summaries" / f"{args.mode}.json", summary_payload)
                _write_json(output_dir / "summary.json", summary_payload)
                print(
                    f"[success] {scenario_id} category={category} "
                    f"mode={args.mode} video={video}"
                )
                break

            attempts_this_invocation += 1
            if (
                args.max_attempts_per_scenario
                and attempts_this_invocation >= args.max_attempts_per_scenario
            ):
                print(
                    f"[stop] {scenario_id} mode={args.mode} did not succeed after "
                    f"{attempts_this_invocation} attempts",
                    file=sys.stderr,
                )
                return 2
            attempt_number += 1
            time.sleep(args.retry_delay_sec)

    if args.dry_run:
        return 0
    summary_payload = {
        "schema": "mpd_todrawer_until_success",
        "schema_version": 3,
        "mode": args.mode,
        "attempt_generation_revision": ATTEMPT_GENERATION_REVISION,
        "success_definition": (
            "goal reached, no controlled brake or q-start collision at or before goal, "
            "and parked-Franka clearance passed until measured robot motion"
        ),
        "attempt_randomization": (
            "planner seed and full obstacle realization vary per attempt; "
            "multi-object attempts use at least two unoriented motion lines"
        ),
        "replay_policy": "render every attempt that has a manifest",
        "successful": successful,
    }
    _write_json(output_dir / "summaries" / f"{args.mode}.json", summary_payload)
    _write_json(output_dir / "summary.json", summary_payload)
    print(
        f"[done] mode={args.mode} {len(successful)}/{len(CATEGORIES)} "
        f"categories saved under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
