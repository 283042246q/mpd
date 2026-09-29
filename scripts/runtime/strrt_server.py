#!/usr/bin/env python3
"""Resident ST-RRT* planner using the dynamic-world IPC contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.runtime.infer_dynamic_server import DynamicResidentPlannerService
from scripts.runtime.ipc_protocol import ProtocolError
from scripts.runtime.strrt_engine import StrrtEngine


class StrrtService(DynamicResidentPlannerService):
    def _plan(self, message: dict[str, Any]) -> dict[str, Any]:
        deadline = message.get("deadline_unix_ns")
        if isinstance(deadline, bool) or not isinstance(deadline, int) or deadline < 0:
            raise ProtocolError("ST-RRT* requires deadline_unix_ns")
        request = message.get("request")
        if not isinstance(request, dict):
            raise ProtocolError("request must be a JSON object")
        wrapped = dict(message)
        wrapped["request"] = {**request, "_deadline_unix_ns": deadline}
        return super()._plan(wrapped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--static-scene", required=True, type=Path)
    parser.add_argument("--target-pose-xyzw", nargs=7, type=float)
    parser.add_argument("--max-duration-s", type=float, default=14.0)
    parser.add_argument("--solve-budget-s", type=float, default=1.5)
    parser.add_argument("--edge-dt-s", type=float, default=0.02)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--planner-range", type=float, default=0.35)
    parser.add_argument("--worker-seed", type=int, default=0)
    args = parser.parse_args(argv)

    def create_engine(state_callback):
        return StrrtEngine(
            args.static_scene,
            target_pose_xyzw=args.target_pose_xyzw,
            max_duration_s=args.max_duration_s,
            solve_budget_s=args.solve_budget_s,
            edge_dt_s=args.edge_dt_s,
            max_joint_step_rad=args.max_joint_step_rad,
            planner_range=args.planner_range,
            worker_seed=args.worker_seed,
            state_callback=state_callback,
        )

    StrrtService(args.socket, args.output_root, create_engine, trajectory_compression=False).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
