#!/usr/bin/env python3
"""Measure when each ToDrawer mode visibly starts moving in recorded replays."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EXECUTED_STATUSES = {
    "accepted",
    "superseded",
    "interrupted",
    "interrupted_before_handoff",
}
DEFAULT_MODES = (
    "phase4",
    "phase4_aligned",
    "joint",
    "joint_corridor_a",
    "f1_c",
    "f1_c_corridor_a",
    "f2_c",
    "f3_c",
    "f1_tau_r",
    "f1_tau_r_corridor_a",
    "f2_tau_r",
    "f3_tau_r",
)


def measure_first_plan_completion(manifest_path: Path) -> float | None:
    """World start to the first successful MPD result created by the worker."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    world_start_ns = manifest.get("world_start_unix_ns")
    if world_start_ns is None:
        return None
    completed = []
    for result_path in (manifest_path.parent.parent / "planner-results").glob("*/result.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if result.get("status") != "success":
            continue
        created = result.get("created_unix_time")
        if isinstance(created, (int, float)) and math.isfinite(float(created)):
            completed.append(float(created))
    return min(completed) - int(world_start_ns) * 1e-9 if completed else None


def _trajectory_times(archive: Any) -> np.ndarray:
    for key in ("time_from_start", "times", "time_from_start_s"):
        if key in archive:
            return np.asarray(archive[key], dtype=np.float64)
    raise ValueError("trajectory archive has no supported time array")


def measure_manifest_motion_start(
    manifest_path: Path,
    *,
    threshold_rad: float = 0.01,
) -> dict[str, Any] | None:
    """Return the first threshold crossing for the first executed MPD command."""

    if threshold_rad <= 0.0:
        raise ValueError("motion threshold must be positive")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    initial_q = np.asarray(payload.get("initial_q"), dtype=np.float64)
    if initial_q.ndim != 1 or initial_q.size < 7:
        raise ValueError(f"{manifest_path}: initial_q must contain at least seven joints")
    initial_q = initial_q[:7]
    world_offset_s = 0.0
    episode_start_ns = payload.get("episode_start_unix_ns")
    world_start_ns = payload.get("world_start_unix_ns")
    if episode_start_ns is not None and world_start_ns is not None:
        world_offset_s = (int(episode_start_ns) - int(world_start_ns)) * 1e-9

    for index, plan in enumerate(payload.get("plans", [])):
        if plan.get("status") not in EXECUTED_STATUSES:
            continue
        trajectory_ref = plan.get("trajectory")
        start_s = plan.get("start_s")
        if trajectory_ref is None or start_s is None:
            continue
        trajectory_path = manifest_path.parent / str(trajectory_ref)
        if not trajectory_path.is_file():
            continue
        with np.load(trajectory_path, allow_pickle=False) as archive:
            positions = np.asarray(archive["positions"], dtype=np.float64)
            times = _trajectory_times(archive)
        if positions.ndim != 2 or positions.shape[0] != times.size or positions.shape[1] < 7:
            raise ValueError(f"{trajectory_path}: inconsistent trajectory arrays")
        displacement = np.max(np.abs(positions[:, :7] - initial_q[None, :]), axis=1)
        hits = np.flatnonzero(displacement >= threshold_rad)
        if hits.size == 0:
            continue
        hit = int(hits[0])
        timing = plan.get("phase_timing", {})
        measured_s = world_offset_s + float(start_s) + float(times[hit])
        return {
            "manifest": manifest_path.as_posix(),
            "plan_index": index,
            "plan_id": str(plan.get("id", f"plan-{index:04d}")),
            "threshold_rad": threshold_rad,
            "motion_start_from_world_s": measured_s,
            "maximum_joint_displacement_rad": float(displacement[hit]),
            "planning_submitted_from_world_s": (
                None
                if timing.get("planning_submitted_s") is None
                else world_offset_s + float(timing["planning_submitted_s"])
            ),
            "bridge_start_from_world_s": (
                None
                if timing.get("bridge_start_s") is None
                else world_offset_s + float(timing["bridge_start_s"])
            ),
            "handoff_from_world_s": (
                None
                if timing.get("handoff_s") is None
                else world_offset_s + float(timing["handoff_s"])
            ),
        }
    return None


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def analyze_logs(
    logs_root: Path,
    *,
    modes: tuple[str, ...] = DEFAULT_MODES,
    thresholds_rad: tuple[float, ...] = (0.01, 0.02),
) -> dict[str, Any]:
    mode_set = set(modes)
    measurements: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {f"{threshold:.6g}": [] for threshold in thresholds_rad}
        for mode in modes
    }
    skipped = defaultdict(int)
    first_plan_times: dict[str, list[float]] = {mode: [] for mode in modes}
    for manifest_path in logs_root.rglob("replay-manifest.json"):
        mode = next((part for part in manifest_path.parts if part in mode_set), None)
        if mode is None:
            continue
        try:
            first_plan = measure_first_plan_completion(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            first_plan = None
        if first_plan is not None and math.isfinite(first_plan):
            first_plan_times[mode].append(first_plan)
        for threshold in thresholds_rad:
            key = f"{threshold:.6g}"
            try:
                result = measure_manifest_motion_start(
                    manifest_path, threshold_rad=threshold
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                skipped[mode] += 1
                continue
            if result is not None and math.isfinite(result["motion_start_from_world_s"]):
                measurements[mode][key].append(result)

    summaries = {}
    for mode in modes:
        summaries[mode] = {
            key: _describe(
                [entry["motion_start_from_world_s"] for entry in entries]
            )
            for key, entries in measurements[mode].items()
        }
        summaries[mode]["skipped_measurements"] = skipped[mode]
        summaries[mode]["first_plan_completed_from_world_s"] = _describe(first_plan_times[mode])
    return {
        "schema": "mpd_todrawer_mode_motion_start_audit",
        "schema_version": 1,
        "logs_root": logs_root.resolve().as_posix(),
        "thresholds_rad": list(thresholds_rad),
        "modes": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("scripts/isaaclab/logs"))
    parser.add_argument("--modes", nargs="+", choices=DEFAULT_MODES, default=list(DEFAULT_MODES))
    parser.add_argument("--threshold-rad", type=float, nargs="+", default=[0.01, 0.02])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(value <= 0.0 for value in args.threshold_rad):
        raise SystemExit("motion thresholds must be positive")
    report = analyze_logs(
        args.logs_root,
        modes=tuple(args.modes),
        thresholds_rad=tuple(args.threshold_rad),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
