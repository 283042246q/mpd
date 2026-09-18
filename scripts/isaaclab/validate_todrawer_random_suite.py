#!/usr/bin/env python3
"""Validate or statistically audit a random ToDrawer benchmark suite."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from scripts.isaaclab.benchmark_todrawer_random import (
    BASE_CROSSINGS,
    CATEGORIES,
    GENERATION_REVISION,
    MAX_GENERATION_RESAMPLE_ATTEMPTS,
    SAFE_CONTROL_CROSSINGS,
    generate_suite,
)
from scripts.isaaclab.todrawer_scenario_validation import (
    STATIC_CLEARANCE_WARNING_M,
    load_static_environment_boxes,
    object_position_at,
    trajectory_clearances,
)


MOTION_MODELS = {
    "constant_velocity",
    "constant_acceleration",
    "sinusoidal_curve",
    "smooth_speed_variation",
    "curved_speed_variation",
}
ANCHORS = {item["id"]: item for item in BASE_CROSSINGS + SAFE_CONTROL_CROSSINGS}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_tree(value: Any, path: str = "scenario") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{path} contains NaN or Inf")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{path}.{key}")


def validate_scenario(
    scenario: dict[str, Any],
    *,
    static_boxes=None,
) -> list[str]:
    """Raise on contract violations and return non-fatal clearance warnings."""

    scenario_id = str(scenario.get("id", "<unknown>"))
    _require(scenario.get("schema") == "mpd_todrawer_dynamic_scenario", f"{scenario_id}: bad schema")
    _require(scenario.get("schema_version") == 3, f"{scenario_id}: schema_version must be 3")
    _require(scenario.get("frame_id") == "fr3_link0", f"{scenario_id}: bad frame_id")
    category = scenario.get("category")
    _require(category in CATEGORIES, f"{scenario_id}: unknown category")
    clock = scenario.get("motion_clock", {})
    _require(clock.get("mode") == "first_robot_state_after_scenario_load", f"{scenario_id}: bad clock mode")
    _require(clock.get("independent_of_execution") is True, f"{scenario_id}: clock is execution-dependent")
    objects = scenario.get("objects")
    _require(isinstance(objects, list) and 1 <= len(objects) <= 3, f"{scenario_id}: object count")
    _finite_tree(scenario)
    warnings = []
    times = []
    anchor_ids = []
    roles = []
    boxes = load_static_environment_boxes() if static_boxes is None else static_boxes
    for item in objects:
        object_id = str(item.get("id", "<object>"))
        anchor_id = item.get("anchor_id")
        _require(anchor_id in ANCHORS, f"{scenario_id}/{object_id}: illegal anchor")
        _require(item.get("corridor_id") == anchor_id, f"{scenario_id}/{object_id}: corridor mismatch")
        _require(item.get("motion_model") in MOTION_MODELS, f"{scenario_id}/{object_id}: bad motion")
        direction = [float(value) for value in item.get("direction", [])]
        _require(len(direction) == 3, f"{scenario_id}/{object_id}: bad direction")
        _require(
            abs(math.sqrt(sum(value * value for value in direction)) - 1.0) <= 1e-7,
            f"{scenario_id}/{object_id}: direction is not unit",
        )
        at_crossing = object_position_at(item, float(item["crossing_time_s"]))
        crossing_error = math.dist(at_crossing, item["anchor_position"])
        _require(crossing_error < 1e-7, f"{scenario_id}/{object_id}: crossing error {crossing_error}")
        base = ANCHORS[anchor_id]["anchor"]
        jitter_limit = 0.020 if category == "inflated_dense" else 0.015
        _require(
            all(
                abs(float(actual) - float(expected)) <= jitter_limit + 1e-12
                for actual, expected in zip(item["anchor_position"], base)
            ),
            f"{scenario_id}/{object_id}: anchor jitter exceeds {jitter_limit}",
        )
        clearance = trajectory_clearances(item, static_boxes=boxes)
        _require(clearance.robot_base_m > 0.0, f"{scenario_id}/{object_id}: robot-base intersection")
        _require(
            abs(float(item.get("minimum_robot_base_clearance_m", -1.0)) - clearance.robot_base_m) <= 1e-9,
            f"{scenario_id}/{object_id}: stored base clearance mismatch",
        )
        _require(
            abs(float(item.get("minimum_static_environment_clearance_m", -1.0)) - clearance.static_environment_m)
            <= 1e-6,
            f"{scenario_id}/{object_id}: stored static clearance mismatch",
        )
        attempts = int(item.get("generation_resample_attempts", 0))
        _require(1 <= attempts <= MAX_GENERATION_RESAMPLE_ATTEMPTS, f"{scenario_id}/{object_id}: bad resample count")
        if clearance.static_environment_m < STATIC_CLEARANCE_WARNING_M:
            warnings.append(
                f"{scenario_id}/{object_id}: static clearance is only {clearance.static_environment_m:.4f} m"
            )
        times.append(float(item["crossing_time_s"]))
        anchor_ids.append(str(anchor_id))
        roles.append(str(item.get("schedule_role")))

    ordered = sorted(times)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    adjacent_pairs = ({"A0", "A1"}, {"A1", "A2"})
    if category in {"simultaneous_multi", "curved_crossing"}:
        _require(set(anchor_ids) in adjacent_pairs, f"{scenario_id}: non-adjacent simultaneous pair")
        _require(max(times) - min(times) <= 0.20 + 1e-12, f"{scenario_id}: simultaneous spread")
    elif category in {"staggered_multi", "accelerating_crossing"}:
        _require(all(0.70 - 1e-12 <= gap <= 1.10 + 1e-12 for gap in gaps), f"{scenario_id}: staggered gap")
        if len(objects) == 2:
            _require(set(anchor_ids) in adjacent_pairs, f"{scenario_id}: non-adjacent staggered pair")
    elif category == "fast_crossing" and len(objects) == 2:
        _require(set(anchor_ids) in adjacent_pairs, f"{scenario_id}: non-adjacent fast pair")
        _require(0.60 - 1e-12 <= gaps[0] <= 0.90 + 1e-12, f"{scenario_id}: fast gap")
    elif category == "uncertain_motion":
        _require(set(anchor_ids) in adjacent_pairs, f"{scenario_id}: non-adjacent uncertain pair")
        _require(0.65 - 1e-12 <= gaps[0] <= 1.00 + 1e-12, f"{scenario_id}: uncertain gap")
    elif category == "inflated_dense":
        _require(anchor_ids == ["A0", "A1", "A2"], f"{scenario_id}: dense anchors")
        _require(all(0.74 - 1e-12 <= gap <= 1.06 + 1e-12 for gap in gaps), f"{scenario_id}: dense gap")
    elif category == "mixed_motion_multi":
        near = [time for time, role in zip(times, roles) if role == "simultaneous"]
        delayed = [time for time, role in zip(times, roles) if role == "delayed"]
        _require(
            anchor_ids == ["A0", "A1", "A2"] and len(near) == 2 and len(delayed) == 1, f"{scenario_id}: mixed roles"
        )
        _require(max(near) - min(near) <= 0.20 + 1e-12, f"{scenario_id}: mixed near spread")
        _require(
            0.90 - 1e-12 <= delayed[0] - statistics.mean(near) <= 1.20 + 1e-12, f"{scenario_id}: mixed delayed gap"
        )
    elif category == "safe_control":
        _require(set(anchor_ids) <= {"S0", "S1"}, f"{scenario_id}: safe-control anchor")
        _require(all(4.40 - 1e-12 <= time <= 6.00 + 1e-12 for time in times), f"{scenario_id}: safe-control time")
    else:
        _require(all(4.0 <= time <= 6.5 for time in times), f"{scenario_id}: crossing outside primary window")
    return warnings


def validate_suite(payload: dict[str, Any], *, static_scene: Path | None = None) -> list[str]:
    _require(payload.get("schema") == "mpd_todrawer_random_suite", "bad suite schema")
    _require(payload.get("schema_version") == 4, "suite schema_version must be 4")
    _require(payload.get("generation_policy", {}).get("revision") == GENERATION_REVISION, "stale generation revision")
    scenarios = payload.get("scenarios")
    _require(isinstance(scenarios, list), "suite scenarios must be a list")
    _require(payload.get("scenario_count") == len(scenarios), "scenario_count mismatch")
    boxes = load_static_environment_boxes(static_scene)
    return [warning for scenario in scenarios for warning in validate_scenario(scenario, static_boxes=boxes)]


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "max": float(array.max()),
    }


def monte_carlo(count: int, seed: int) -> dict[str, Any]:
    suite = generate_suite(count, seed)
    validate_suite(suite)
    category_counts = Counter()
    anchor_counts = Counter()
    motion_roles = Counter()
    category_motion_roles = Counter()
    crossing_times = []
    static_clearances = []
    base_clearances = []
    attempts = Counter()
    simultaneous_order = Counter()
    for scenario in suite["scenarios"]:
        category_counts[scenario["category"]] += 1
        objects = scenario["objects"]
        for item in objects:
            anchor_counts[item["anchor_id"]] += 1
            motion_roles[f"{item['motion_model']}|{item['schedule_role']}"] += 1
            category_motion_roles[
                (scenario["category"], item["motion_model"], item["schedule_role"])
            ] += 1
            crossing_times.append(float(item["crossing_time_s"]))
            static_clearances.append(float(item["minimum_static_environment_clearance_m"]))
            base_clearances.append(float(item["minimum_robot_base_clearance_m"]))
            attempts[int(item["generation_resample_attempts"])] += 1
        simultaneous = [item for item in objects if item["schedule_role"] == "simultaneous"]
        if len(simultaneous) == 2:
            first = min(simultaneous, key=lambda item: item["crossing_time_s"])
            simultaneous_order[first["anchor_id"]] += 1
    if count >= 300:
        required_combinations = [
            ("inflated_dense", motion, role)
            for motion in (
                "constant_velocity",
                "constant_acceleration",
                "curved_speed_variation",
            )
            for role in ("earliest", "middle", "latest")
        ] + [
            ("mixed_motion_multi", motion, role)
            for motion in (
                "constant_acceleration",
                "sinusoidal_curve",
                "curved_speed_variation",
            )
            for role in ("simultaneous", "delayed")
        ]
        missing = [item for item in required_combinations if not category_motion_roles[item]]
        _require(not missing, f"motion model remains bound to temporal role: {missing}")
    anchor_warnings = []
    primary_total = sum(anchor_counts[name] for name in ("A0", "A1", "A2"))
    if primary_total:
        for name in ("A0", "A1", "A2"):
            share = anchor_counts[name] / primary_total
            if share < 0.25 or share > 0.42:
                anchor_warnings.append(
                    f"{name} usage share {share:.3%} is outside the broad 25%..42% audit band"
                )
    return {
        "scenario_count": count,
        "category_counts": dict(sorted(category_counts.items())),
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "motion_model_by_temporal_role": dict(sorted(motion_roles.items())),
        "simultaneous_first_anchor": dict(sorted(simultaneous_order.items())),
        "crossing_time_s": _percentiles(crossing_times),
        "static_clearance_m": _percentiles(static_clearances),
        "robot_base_clearance_m": _percentiles(base_clearances),
        "resample_attempts": {str(key): value for key, value in sorted(attempts.items())},
        "warnings": anchor_warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite", type=Path)
    source.add_argument("--scenario", type=Path)
    source.add_argument("--monte-carlo", type=int)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--static-scene", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.suite is not None:
        payload = json.loads(args.suite.read_text(encoding="utf-8"))
        warnings = validate_suite(payload, static_scene=args.static_scene)
        result = {
            "valid": True,
            "scenario_count": len(payload["scenarios"]),
            "warning_count": len(warnings),
            "warnings": warnings[:50],
            "warnings_truncated": len(warnings) > 50,
        }
    elif args.scenario is not None:
        payload = json.loads(args.scenario.read_text(encoding="utf-8"))
        warnings = validate_scenario(
            payload,
            static_boxes=load_static_environment_boxes(args.static_scene),
        )
        result = {
            "valid": True,
            "scenario_id": payload.get("id"),
            "warning_count": len(warnings),
            "warnings": warnings[:50],
            "warnings_truncated": len(warnings) > 50,
        }
    else:
        if args.monte_carlo < 1:
            parser.error("--monte-carlo must be positive")
        result = monte_carlo(args.monte_carlo, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
