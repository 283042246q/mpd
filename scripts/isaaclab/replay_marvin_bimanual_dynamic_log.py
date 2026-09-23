#!/usr/bin/env python3
"""Deterministically materialize a Phase-5 dynamic replay timeline.

The output is intentionally simulator-neutral JSON. It can be paired with the
Phase-2 Isaac replay artifact: every recorded joint sample is associated with
the latest world generation that was valid at that timestamp.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def build_timeline(record: dict) -> dict:
    events = record.get("events")
    if record.get("schema") != "marvin_bimanual_dynamic_replay/v1" or not isinstance(events, list):
        raise ValueError("invalid Marvin dynamic replay schema")
    ordered = sorted(events, key=lambda item: (int(item["unix_ns"]), int(item["sequence"])))
    versions = [
        int(item["payload"]["world_version"])
        for item in ordered
        if item.get("type") == "world"
    ]
    if any(b <= a for a, b in zip(versions, versions[1:])):
        raise ValueError("recorded world versions must increase strictly")
    return {
        "schema": "marvin_bimanual_isaac_dynamic_timeline/v1",
        "request_id": record.get("request_id"),
        "events": ordered,
        "world_versions": versions,
        "deterministic_order": "unix_ns_then_sequence",
    }


def selected_plan(record: dict) -> dict:
    """Return the final selected artifact recorded by the latest-only adapter."""
    timeline = build_timeline(record)
    plans = [
        event["payload"]
        for event in timeline["events"]
        if event.get("type") == "plan_selected"
    ]
    if not plans:
        raise ValueError("dynamic replay contains no plan_selected event")
    plan = dict(plans[-1])
    required = {
        "result_path",
        "trajectory_path",
        "top_k_index",
        "trajectory_start_unix_ns",
    }
    missing = sorted(required - plan.keys())
    if missing:
        raise ValueError(f"plan_selected is missing {missing}")
    return plan


def predicted_world_objects(record: dict, unix_ns: int) -> list[dict]:
    """Predict the latest recorded constant-velocity world at ``unix_ns``."""
    if isinstance(unix_ns, bool) or not isinstance(unix_ns, int) or unix_ns < 0:
        raise ValueError("unix_ns must be a non-negative integer")
    timeline = build_timeline(record)
    worlds = [
        event["payload"]
        for event in timeline["events"]
        if event.get("type") == "world"
    ]
    if not worlds:
        return []
    eligible = [
        world for world in worlds if int(world.get("stamp_unix_ns", 0)) <= unix_ns
    ]
    world = eligible[-1] if eligible else worlds[0]
    stamp_ns = int(world["stamp_unix_ns"])
    elapsed = max(0.0, (unix_ns - stamp_ns) * 1e-9)
    predicted = []
    for item in world.get("objects", []):
        pose = item.get("pose") or {}
        position = [float(value) for value in pose.get("position", ())]
        velocity = [float(value) for value in item.get("linear_velocity", ())]
        if len(position) != 3 or len(velocity) != 3 or not all(
            math.isfinite(value) for value in position + velocity
        ):
            raise ValueError("dynamic world object has invalid position or velocity")
        predicted.append(
            {
                **item,
                "pose": {
                    **pose,
                    "position": [
                        value + elapsed * speed
                        for value, speed in zip(position, velocity)
                    ],
                },
            }
        )
    return predicted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    timeline = build_timeline(json.loads(args.record.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(timeline, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
