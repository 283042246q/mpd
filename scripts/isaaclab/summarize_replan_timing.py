#!/usr/bin/env python3
"""Summarize Phase-4 handoff timing and command continuity from a replay manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXECUTED_STATUSES = frozenset(("accepted", "superseded"))


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def summarize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    plans = payload.get("plans")
    events = payload.get("events", [])
    if not isinstance(plans, list) or not isinstance(events, list):
        raise ValueError("manifest plans and events must be lists")

    executed = []
    for index, plan in enumerate(plans):
        if not isinstance(plan, dict) or plan.get("status") not in EXECUTED_STATUSES:
            continue
        timing = plan.get("phase_timing")
        if not isinstance(timing, dict):
            raise ValueError(f"executed plan {index} has no phase_timing")
        active_from = _finite(plan.get("active_from_s"), f"plans[{index}].active_from_s")
        active_until = _finite(plan.get("active_until_s"), f"plans[{index}].active_until_s")
        submitted = _finite(
            timing.get("planning_submitted_s"),
            f"plans[{index}].phase_timing.planning_submitted_s",
        )
        bridge_start = _finite(
            timing.get("bridge_start_s"),
            f"plans[{index}].phase_timing.bridge_start_s",
        )
        handoff = _finite(
            timing.get("handoff_s"), f"plans[{index}].phase_timing.handoff_s"
        )
        if not submitted <= bridge_start <= handoff <= active_until + 1e-6:
            raise ValueError(f"executed plan {index} has inconsistent phase timing")
        if abs(active_from - bridge_start) > 1e-5:
            raise ValueError(f"executed plan {index} does not start at bridge_start")
        executed.append(
            {
                "id": str(plan.get("id", f"plan-{index}")),
                "status": plan["status"],
                "planning_submitted_s": submitted,
                "bridge_start_s": bridge_start,
                "handoff_s": handoff,
                "active_until_s": active_until,
                "old_continuation_s": bridge_start - submitted,
                "bridge_s": handoff - bridge_start,
                "latest_mpd_realized_s": max(0.0, active_until - handoff),
                "latest_mpd_nominal_s": _finite(
                    timing.get("mpd_suffix_s"),
                    f"plans[{index}].phase_timing.mpd_suffix_s",
                ),
            }
        )
    executed.sort(key=lambda item: item["bridge_start_s"])

    gaps = []
    for old, new in zip(executed, executed[1:]):
        gap = new["bridge_start_s"] - old["active_until_s"]
        gaps.append(
            {
                "from": old["id"],
                "to": new["id"],
                "gap_s": max(0.0, gap),
                "overlap_s": max(0.0, -gap),
            }
        )

    totals = {
        "old_continuation_s": sum(item["old_continuation_s"] for item in executed),
        "quintic_bridge_s": sum(item["bridge_s"] for item in executed),
        "latest_mpd_realized_s": sum(item["latest_mpd_realized_s"] for item in executed),
        "latest_mpd_nominal_s": sum(item["latest_mpd_nominal_s"] for item in executed),
    }
    ratio_denominator = (
        totals["old_continuation_s"]
        + totals["quintic_bridge_s"]
        + totals["latest_mpd_realized_s"]
    )
    ratios = {
        key: (value / ratio_denominator if ratio_denominator > 0.0 else 0.0)
        for key, value in (
            ("old_continuation", totals["old_continuation_s"]),
            ("quintic_bridge", totals["quintic_bridge_s"]),
            ("latest_mpd_realized", totals["latest_mpd_realized_s"]),
        )
    }
    return {
        "executed_plan_count": len(executed),
        "handoff_event_count": sum(event.get("type") == "handoff" for event in events),
        "brake_event_count": sum(event.get("type") == "brake" for event in events),
        "maximum_command_gap_s": max((item["gap_s"] for item in gaps), default=0.0),
        "command_gaps": gaps,
        "phase_totals": totals,
        "phase_ratios": ratios,
        "plans": executed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-gap-s", type=float)
    parser.add_argument("--require-no-brake", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    summary = summarize_manifest(payload)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if (
        args.maximum_gap_s is not None
        and summary["maximum_command_gap_s"] > args.maximum_gap_s
    ):
        return 2
    if args.require_no_brake and summary["brake_event_count"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
