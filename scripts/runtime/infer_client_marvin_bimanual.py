#!/usr/bin/env python3
"""CLI client for the resident Marvin bimanual worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mpd.bimanual.runtime_contract import BimanualRequest
from scripts.runtime.ipc_protocol import PROTOCOL_SCHEMA_VERSION, receive_message, send_message


def request(socket_path: Path, message: dict, timeout_s: float) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout_s)
        stream.connect(str(socket_path.expanduser().resolve()))
        send_message(stream, message)
        return receive_message(stream)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    commands = parser.add_subparsers(dest="op", required=True)
    commands.add_parser("health")
    commands.add_parser("shutdown")
    plan = commands.add_parser("plan")
    plan.add_argument("--request", required=True, type=Path)
    plan.add_argument("--request-seq", required=True, type=int)
    plan.add_argument("--deadline-s", type=float, default=None)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    message = {"schema_version": PROTOCOL_SCHEMA_VERSION, "op": args.op}
    if args.op == "plan":
        raw = json.loads(args.request.read_text(encoding="utf-8"))
        parsed = BimanualRequest.from_dict(raw)
        message.update(
            request_seq=args.request_seq,
            world_version=parsed.world_version,
            deadline_unix_ns=(
                parsed.deadline_unix_ns
                if args.deadline_s is None
                else time.time_ns() + int(args.deadline_s * 1e9)
            ),
            request=raw,
        )
    response = request(args.socket, message, args.timeout_s)
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if response.get("status") == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
