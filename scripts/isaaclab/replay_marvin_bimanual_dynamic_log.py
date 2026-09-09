#!/usr/bin/env python3
"""Deterministically materialize a Phase-5 dynamic replay timeline.

The output is intentionally simulator-neutral JSON. It can be paired with the
Phase-2 Isaac replay artifact: every recorded joint sample is associated with
the latest world generation that was valid at that timestamp.
"""
from __future__ import annotations

import argparse
import json
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
