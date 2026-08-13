import socket
import struct

import pytest

from scripts.runtime.ipc_protocol import (
    MAX_MESSAGE_BYTES,
    ProtocolError,
    encode_message,
    receive_message,
    send_message,
)


def test_round_trip_over_socket_pair():
    sender, receiver = socket.socketpair()
    try:
        message = {"schema_version": 1, "op": "health", "unicode": "规划"}
        send_message(sender, message)
        assert receive_message(receiver) == message
    finally:
        sender.close()
        receiver.close()


def test_rejects_oversized_frame_before_reading_payload():
    sender, receiver = socket.socketpair()
    try:
        sender.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        with pytest.raises(ProtocolError, match="Message size"):
            receive_message(receiver)
    finally:
        sender.close()
        receiver.close()


def test_rejects_non_finite_json():
    with pytest.raises(ProtocolError, match="valid JSON"):
        encode_message({"value": float("nan")})
