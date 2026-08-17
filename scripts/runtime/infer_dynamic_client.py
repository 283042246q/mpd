#!/usr/bin/env python3
"""CLI for the separate Phase-4 dynamic MPD service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from scripts.runtime.infer_client import request
from scripts.runtime.infer_once import _load_request
from scripts.runtime.ipc_protocol import PROTOCOL_SCHEMA_VERSION


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send commands to infer_dynamic_server.py.")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("health")
    subparsers.add_parser("shutdown")
    update = subparsers.add_parser("update-world")
    update.add_argument("--world", required=True, type=Path)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--request", required=True, type=Path)
    plan.add_argument("--request-seq", required=True, type=int)
    plan.add_argument("--world-version", required=True, type=int)
    plan.add_argument("--trajectory-start-unix-ns", required=True, type=int)
    plan.add_argument("--deadline-sec", type=float, default=None)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    operation = args.operation.replace("-", "_")
    message = {"schema_version": PROTOCOL_SCHEMA_VERSION, "op": operation}
    if args.operation == "update-world":
        message["world"] = _json_object(args.world)
    elif args.operation == "plan":
        raw_request, _ = _load_request(args.request.expanduser().resolve())
        message.update(
            request_seq=args.request_seq,
            world_version=args.world_version,
            trajectory_start_unix_ns=args.trajectory_start_unix_ns,
            deadline_unix_ns=(
                None if args.deadline_sec is None else time.time_ns() + int(args.deadline_sec * 1_000_000_000)
            ),
            request=raw_request,
        )
    response = request(args.socket, message, args.timeout_sec)
    print(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if response.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
