#!/usr/bin/env python3
"""MPD-only Marvin static inference entrypoint.

When a checkpoint is supplied, the production diffusion worker can be plugged
into ``plan``.  The deterministic interpolation fallback makes the contract
and ROS integration testable without GPU/model files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml

from mpd.bimanual.runtime_contract import BimanualRequest, JOINT_NAMES, validate_result


def plan(request: BimanualRequest, *, points: int = 64, duration_s: float = 2.0) -> dict:
    goal = tuple(request.q_goal if request.q_goal is not None else request.q_start)
    start = tuple(request.q_start)
    count = max(2, int(points))
    trajectory = [[left + (right - left) * step / (count - 1) for left, right in zip(start, goal)] for step in range(count)]
    result = {"schema": "marvin_bimanual_result/v1", "request_id": request.request_id, "status": "ok", "joint_names": list(JOINT_NAMES), "positions": trajectory, "velocities": [], "accelerations": [], "time_from_start": [float(duration_s) * step / (len(trajectory) - 1) for step in range(len(trajectory))], "world_version": request.world_version, "validation": {"valid": True, "runtime_mode": request.runtime_mode}}
    return validate_result(result, request=request)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text()) if args.config else {}
    request = BimanualRequest.from_dict(json.loads(args.request.read_text()))
    result = plan(request, points=int(config.get("num_trajectory_samples", 64)))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
