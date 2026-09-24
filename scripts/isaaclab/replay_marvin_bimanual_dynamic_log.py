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
        {**event["payload"], "selected_unix_ns": int(event["unix_ns"])}
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


def world_start_unix_ns(record: dict) -> int:
    """Return the explicit scenario start, falling back to the first snapshot."""
    timeline = build_timeline(record)
    starts = [
        int(event["payload"]["scenario_start_unix_ns"])
        for event in timeline["events"]
        if event.get("type") == "world_start"
    ]
    if starts:
        return min(starts)
    snapshots = [
        int(event["payload"]["stamp_unix_ns"])
        for event in timeline["events"]
        if event.get("type") == "world"
        and int(event["payload"].get("stamp_unix_ns", 0)) > 0
    ]
    if not snapshots:
        raise ValueError("dynamic replay contains no world start or world snapshot")
    return min(snapshots)


def candidate_visual_states(record: dict, unix_ns: int, candidate_count: int) -> list[str]:
    """Franka-compatible gray/blue/green/red states for the selected generation."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    timeline = build_timeline(record)
    plan = selected_plan(record)
    generation = plan.get("generation")
    states = ["hidden"] * candidate_count
    for event in timeline["events"]:
        if int(event["unix_ns"]) > int(unix_ns):
            break
        if event.get("type") != "candidate_revalidation":
            continue
        payload = event["payload"]
        if generation is not None and payload.get("generation") != generation:
            continue
        index = int(payload.get("candidate", -1))
        if 0 <= index < candidate_count:
            states[index] = "gray" if payload.get("safe") else "red"
    selected_index = int(plan["top_k_index"])
    if int(unix_ns) >= int(plan["selected_unix_ns"]):
        states[selected_index] = (
            "blue"
            if int(unix_ns) >= int(plan["trajectory_start_unix_ns"])
            else "green"
        )
    return states


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
    # The demo world is constant velocity from an explicit scenario start, so
    # back-projection before the first published observation is deterministic.
    elapsed = (unix_ns - stamp_ns) * 1e-9
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


def build_timing_report(
    record: dict,
    *,
    task_mode: str | None = None,
    runtime_mode: str | None = None,
    pipeline_stamps_ns: dict[str, int] | None = None,
) -> dict:
    """Summarize every attempt, including discarded and rejected plans.

    Nested worker/backend durations overlap with their parents and must not be
    added to the end-to-end wall-clock durations.
    """
    events = build_timeline(record)["events"]
    stamps = pipeline_stamps_ns or {}

    def seconds_between(first: int | None, last: int | None) -> float | None:
        if first is None or last is None:
            return None
        return max(0.0, (last - first) * 1e-9)

    def event_of_type(items: list[dict], event_type: str) -> dict | None:
        return next((item for item in items if item.get("type") == event_type), None)

    starts = [index for index, event in enumerate(events) if event.get("type") == "plan_start"]
    attempts = []
    for position, start_index in enumerate(starts):
        end_index = starts[position + 1] if position + 1 < len(starts) else len(events)
        group = events[start_index:end_index]
        start = group[0]
        selected = event_of_type(group, "plan_selected")
        discarded = event_of_type(group, "plan_discard")
        worker = event_of_type(group, "worker_return_timing")
        worker_failure = event_of_type(group, "worker_failure_timing")
        selection = event_of_type(group, "candidate_selection_timing")
        recorder = event_of_type(group, "replay_recorder_timing")
        terminal = selected or discarded or selection or worker_failure or worker
        selected_payload = selected["payload"] if selected else {}
        if selected:
            outcome = "selected"
        elif discarded:
            outcome = "discarded"
        elif selection:
            outcome = "rejected"
        elif worker_failure:
            outcome = "worker_failed"
        elif worker:
            outcome = "worker_returned"
        else:
            outcome = "incomplete"
        attempts.append(
            {
                "attempt": start["payload"].get("attempt", position + 1),
                "generation": start["payload"].get("generation"),
                "outcome": outcome,
                "plan_start_unix_ns": int(start["unix_ns"]),
                "plan_end_unix_ns": int(terminal["unix_ns"]) if terminal else None,
                "plan_wall_elapsed_s": seconds_between(
                    int(start["unix_ns"]),
                    int(terminal["unix_ns"]) if terminal else None,
                ),
                "worker_return_timing": worker["payload"] if worker else None,
                "worker_failure_timing": worker_failure["payload"] if worker_failure else None,
                "candidate_selection_timing_s": (selection["payload"].get("timing_s") if selection else None),
                "selected_plan_timing_s": selected_payload.get("timing_s"),
                "scheduled_wait_s": seconds_between(
                    int(selected["unix_ns"]) if selected else None,
                    (
                        int(selected_payload["trajectory_start_unix_ns"])
                        if selected and "trajectory_start_unix_ns" in selected_payload
                        else None
                    ),
                ),
                "trajectory_duration_s": selected_payload.get("trajectory_duration_s"),
                "recorder_timing": recorder["payload"] if recorder else None,
                "execution_timings": [
                    event["payload"] for event in group
                    if event.get("type") == "execution_timing"
                ],
                "controlled_brake_timings": [
                    event["payload"] for event in group
                    if event.get("type") == "controlled_brake"
                ],
            }
        )

    world_starts = [
        int(event["payload"]["scenario_start_unix_ns"]) for event in events if event.get("type") == "world_start"
    ]
    first_plan = events[starts[0]] if starts else None
    terminal_action = next(
        (event for event in reversed(events) if event.get("type") == "action_terminal_timing"),
        None,
    )
    warmup = event_of_type(events, "resident_warmup_timing")
    uploads = [event["payload"] for event in events if event.get("type") == "world_upload_timing"]
    stage_pairs = {
        "worker_startup": ("pipeline_start", "worker_ready"),
        "ros_build": ("ros_build_start", "ros_build_done"),
        "ros_launch_to_result": ("ros_launch", "ros_result"),
        "isaac_render": ("render_start", "render_end"),
        "pipeline_to_result": ("pipeline_start", "ros_result"),
        "pipeline_total": ("pipeline_start", "pipeline_end"),
    }
    return {
        "schema": "marvin_bimanual_dynamic_timing/v1",
        "request_id": record.get("request_id"),
        "task_mode": task_mode,
        "runtime_mode": runtime_mode,
        "clock_note": (
            "Per-component durations use monotonic clocks; replay and pipeline intervals "
            "use Unix timestamps. Nested durations overlap and must not be summed."
        ),
        "world_start_to_first_plan_s": seconds_between(
            min(world_starts) if world_starts else None,
            int(first_plan["unix_ns"]) if first_plan else None,
        ),
        "world_start_to_action_terminal_s": seconds_between(
            min(world_starts) if world_starts else None,
            int(terminal_action["unix_ns"]) if terminal_action else None,
        ),
        "resident_warmup_timing": warmup["payload"] if warmup else None,
        "world_upload_timings": uploads,
        "action_terminal_timing": terminal_action["payload"] if terminal_action else None,
        "pipeline_stamps_unix_ns": stamps,
        "pipeline_timing_s": {
            name: seconds_between(stamps.get(first), stamps.get(last)) for name, (first, last) in stage_pairs.items()
        },
        "attempts": attempts,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timing-output", type=Path)
    parser.add_argument("--task-mode")
    parser.add_argument("--runtime-mode")
    parser.add_argument("--pipeline-stamp", action="append", default=[], metavar="NAME=UNIX_NS")
    args = parser.parse_args(argv)
    if args.output is None and args.timing_output is None:
        parser.error("at least one of --output or --timing-output is required")
    record = json.loads(args.record.read_text(encoding="utf-8"))
    if args.output is not None:
        _write_json(args.output, build_timeline(record))
    if args.timing_output is not None:
        stamps = {}
        for item in args.pipeline_stamp:
            name, separator, value = item.partition("=")
            if not separator or not name or not value.isdecimal():
                parser.error(f"invalid --pipeline-stamp: {item}")
            stamps[name] = int(value)
        _write_json(
            args.timing_output,
            build_timing_report(
                record,
                task_mode=args.task_mode,
                runtime_mode=args.runtime_mode,
                pipeline_stamps_ns=stamps,
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
