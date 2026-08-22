#!/usr/bin/env python3
"""Serve repeated MPD requests from one warm planner process."""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import socketserver
import stat
import threading
import time
import traceback
from typing import Any, Callable

from scripts.runtime.infer_once import (
    DEFAULT_CONFIG_PATH,
    RESULT_SCHEMA_VERSION,
    RuntimeContractError,
    _atomic_write_json,
    _atomic_write_npz,
)
from scripts.runtime.ipc_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    ProtocolError,
    receive_message,
    send_message,
)
from scripts.runtime.runtime_engine import MpdRuntimeEngine


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class ResidentPlannerService:
    """Thread-safe request dispatcher around one non-reentrant planner."""

    def __init__(
        self,
        socket_path: Path,
        output_root: Path,
        engine_factory: Callable[[Callable[[str], None]], Any],
        *,
        trajectory_compression: bool = True,
    ) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._state = "STARTING"
        self._state_lock = threading.Lock()
        self._plan_lock = threading.Lock()
        self._latest_request_seq = -1
        self._engine = None
        self._startup_error = None
        self._server = None
        self._engine_factory = engine_factory
        self.trajectory_compression = bool(trajectory_compression)

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def set_state(self, state: str) -> None:
        with self._state_lock:
            self._state = state

    def _initialize_engine(self) -> None:
        try:
            self._engine = self._engine_factory(self.set_state)
            self.set_state("READY")
        except Exception as error:  # startup must remain observable over health
            self._startup_error = error
            self.set_state("FAULT")
            traceback.print_exc()

    def health_response(self) -> dict[str, Any]:
        response = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "status": "OK",
            "state": self.state,
            "latest_request_seq": self._latest_request_seq,
        }
        if self._engine is not None:
            response["engine"] = self._engine.health()
        if self._startup_error is not None:
            response["error"] = {
                "type": type(self._startup_error).__name__,
                "message": str(self._startup_error),
            }
        return response

    @staticmethod
    def _require_nonnegative_integer(message: dict[str, Any], key: str) -> int:
        value = message.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProtocolError(f"{key} must be a non-negative integer.")
        return value

    def _plan(self, message: dict[str, Any]) -> dict[str, Any]:
        request_seq = self._require_nonnegative_integer(message, "request_seq")
        world_version = self._require_nonnegative_integer(message, "world_version")
        deadline_unix_ns = message.get("deadline_unix_ns")
        if deadline_unix_ns is not None and (
            isinstance(deadline_unix_ns, bool) or not isinstance(deadline_unix_ns, int) or deadline_unix_ns < 0
        ):
            raise ProtocolError("deadline_unix_ns must be a non-negative integer or null.")
        raw_request = message.get("request")
        if not isinstance(raw_request, dict):
            raise ProtocolError("request must be a JSON object.")

        if self.state != "READY":
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "NOT_READY",
                "state": self.state,
                "request_seq": request_seq,
                "world_version": world_version,
            }
        if deadline_unix_ns is not None and time.time_ns() >= deadline_unix_ns:
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "STALE",
                "reason": "deadline_expired_before_planning",
                "request_seq": request_seq,
                "world_version": world_version,
            }
        if request_seq <= self._latest_request_seq:
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "STALE",
                "reason": "request_seq_not_newer",
                "request_seq": request_seq,
                "world_version": world_version,
            }
        if not self._plan_lock.acquire(blocking=False):
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "BUSY",
                "request_seq": request_seq,
                "world_version": world_version,
            }

        output_dir = self.output_root / f"request-{request_seq:020d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_dir / "trajectory.npz"
        result_path = output_dir / "result.json"
        trajectory_path.unlink(missing_ok=True)
        self._latest_request_seq = request_seq
        self.set_state("PLANNING")
        started = time.perf_counter()
        try:
            artifacts = self._engine.plan(raw_request)
            _atomic_write_npz(
                trajectory_path,
                compressed=self.trajectory_compression,
                **artifacts.trajectory_arrays,
            )
            _atomic_write_json(result_path, artifacts.result_payload)
            finished_unix_ns = time.time_ns()
            if deadline_unix_ns is not None and finished_unix_ns >= deadline_unix_ns:
                return {
                    "schema_version": PROTOCOL_SCHEMA_VERSION,
                    "status": "STALE",
                    "reason": "deadline_expired_after_planning",
                    "request_seq": request_seq,
                    "world_version": world_version,
                    "result_path": result_path.as_posix(),
                }
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "OK",
                "request_seq": request_seq,
                "world_version": world_version,
                "result_path": result_path.as_posix(),
                "trajectory_path": trajectory_path.as_posix(),
                "elapsed_sec": time.perf_counter() - started,
                "engine_instance_id": self._engine.instance_id,
                "trajectory_artifact": artifacts.result_payload.get(
                    "trajectory_artifact",
                    {
                        "schema_version": 1,
                        "compression": "zlib" if self.trajectory_compression else "none",
                    },
                ),
            }
        except RuntimeContractError as error:
            trajectory_path.unlink(missing_ok=True)
            failure = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": error.status,
                "request_id": raw_request.get("request_id"),
                "trajectory_file": None,
                "error": {"type": type(error).__name__, "message": str(error)},
                "created_unix_time": time.time(),
            }
            _atomic_write_json(result_path, failure)
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "PLAN_FAILED",
                "request_seq": request_seq,
                "world_version": world_version,
                "result_path": result_path.as_posix(),
                "error": failure["error"],
            }
        except Exception as error:
            trajectory_path.unlink(missing_ok=True)
            self.set_state("FAULT")
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "FAULT",
                "request_seq": request_seq,
                "world_version": world_version,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        finally:
            if self.state == "PLANNING":
                self.set_state("READY")
            self._plan_lock.release()

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
            raise ProtocolError(f"schema_version must be {PROTOCOL_SCHEMA_VERSION}.")
        operation = message.get("op")
        if operation == "health":
            return self.health_response()
        if operation == "plan":
            return self._plan(message)
        if operation == "shutdown":
            self.set_state("STOPPING")
            if self._server is not None:
                threading.Thread(target=self._server.shutdown, daemon=True).start()
            return {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "status": "OK",
                "state": "STOPPING",
            }
        raise ProtocolError("op must be one of: health, plan, shutdown.")

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
                raise RuntimeError(f"Refusing to replace non-socket path: {self.socket_path}")
            self.socket_path.unlink()

        service = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                try:
                    response = service.dispatch(receive_message(self.request))
                except ProtocolError as error:
                    response = {
                        "schema_version": PROTOCOL_SCHEMA_VERSION,
                        "status": "INVALID_REQUEST",
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
                send_message(self.request, response)

        self._server = _ThreadingUnixServer(self.socket_path.as_posix(), Handler)
        self.socket_path.chmod(0o600)
        loader = threading.Thread(target=self._initialize_engine, name="mpd-engine-loader", daemon=True)
        loader.start()
        try:
            self._server.serve_forever(poll_interval=0.1)
        finally:
            self.set_state("STOPPING")
            self._server.server_close()
            loader.join(timeout=5.0)
            self.socket_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one persistent MPD planner behind a local Unix socket.")
    parser.add_argument("--socket", required=True, type=Path, help="Unix socket path.")
    parser.add_argument("--output-root", required=True, type=Path, help="Per-request output root.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    def engine_factory(state_callback):
        return MpdRuntimeEngine(
            config_path=args.config,
            runtime_output_root=args.output_root,
            device_text=args.device,
            state_callback=state_callback,
        )

    service = ResidentPlannerService(args.socket, args.output_root, engine_factory)
    service.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
