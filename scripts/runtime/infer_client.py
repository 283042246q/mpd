#!/usr/bin/env python3
"""Command-line client for the resident MPD Unix-socket service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

from scripts.runtime.infer_once import _load_request
from scripts.runtime.ipc_protocol import PROTOCOL_SCHEMA_VERSION, receive_message, send_message


def request(socket_path: Path, message: dict, timeout_sec: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout_sec)
        stream.connect(socket_path.expanduser().resolve().as_posix())
        send_message(stream, message)
        return receive_message(stream)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send one command to infer_server.py.")
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    subparsers.add_parser("health")
    subparsers.add_parser("shutdown")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--request", required=True, type=Path)
    plan.add_argument("--request-seq", required=True, type=int)
    plan.add_argument("--world-version", type=int, default=0)
    plan.add_argument("--deadline-sec", type=float, default=None)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    message = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "op": args.operation,
    }
    if args.operation == "plan":
        raw_request, _ = _load_request(args.request.expanduser().resolve())
        message.update(
            request_seq=args.request_seq,
            world_version=args.world_version,
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
