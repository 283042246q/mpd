"""Small length-framed JSON protocol for the resident MPD worker."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

PROTOCOL_SCHEMA_VERSION = 1
MAX_MESSAGE_BYTES = 1024 * 1024
_LENGTH = struct.Struct("!I")


class ProtocolError(ValueError):
    """Raised when a peer sends an invalid or incomplete frame."""


def _read_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            raise ProtocolError("Peer closed the socket before the frame completed.")
        chunks.extend(chunk)
    return bytes(chunks)


def encode_message(message: dict[str, Any]) -> bytes:
    if not isinstance(message, dict):
        raise ProtocolError("Protocol messages must be JSON objects.")
    try:
        payload = json.dumps(
            message,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"Message is not valid JSON: {error}") from error
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Message size must be in [1, {MAX_MESSAGE_BYTES}] bytes, got {len(payload)}.")
    return _LENGTH.pack(len(payload)) + payload


def send_message(stream: socket.socket, message: dict[str, Any]) -> None:
    stream.sendall(encode_message(message))


def receive_message(stream: socket.socket) -> dict[str, Any]:
    length = _LENGTH.unpack(_read_exact(stream, _LENGTH.size))[0]
    if length < 1 or length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"Message size must be in [1, {MAX_MESSAGE_BYTES}] bytes, got {length}.")
    payload = _read_exact(stream, length)
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"Frame payload must be valid UTF-8 JSON: {error}") from error
    if not isinstance(message, dict):
        raise ProtocolError("Protocol messages must be JSON objects.")
    return message
