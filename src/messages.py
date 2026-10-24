"""Protocol message encoding and decoding (CONTEXT.md section 5).

Every frame on the wire is::

    [u32 header_len][u32 payload_len][header JSON, UTF-8][payload bytes]

The header is a JSON object. On ring frames the payload is the NumPy chunk as
raw float64 bytes; side-channel frames have an empty payload. This module holds
no protocol logic, only shapes, size limits and framing.
"""

import json
import socket
import struct

import numpy as np

# Gossip update types (section 5.2)
SUSPECT = "SUSPECT"
ALIVE = "ALIVE"
CONFIRM = "CONFIRM"
LEAVE = "LEAVE"
GOSSIP_TYPES = (SUSPECT, ALIVE, CONFIRM, LEAVE)

# Side-channel message types (section 5.3)
PING_REQ = "PING_REQ"
ACK = "ACK"
NACK = "NACK"

# Ring phases (section 4)
RS = "RS"
AG = "AG"

CHUNK_DTYPE = np.float64

_PREFIX = struct.Struct("!II")
MAX_HEADER_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = 1 << 30


def node_name(index):
    """Docker service name and protocol id of ring position `index`."""
    return f"gpu{index}"


def node_index(name):
    """Inverse of node_name: 'gpu3' -> 3. Plain integers are accepted too."""
    text = str(name)
    return int(text[3:]) if text.startswith("gpu") else int(text)


def gossip_update(type_, about, incarnation, from_, class_=None):
    if type_ not in GOSSIP_TYPES:
        raise ValueError(f"unknown gossip type {type_!r}")
    return {"type": type_, "about": about, "incarnation": int(incarnation),
            "from": from_, "class": class_}


def telemetry_entry(node_id, step, compute_t, wait_t, pred_lag):
    return {"node_id": node_id, "step": int(step), "compute_t": float(compute_t),
            "wait_t": float(wait_t), "pred_lag": float(pred_lag)}


def make_header(src, step, phase, chunk_idx, send_ts, compute_t, wait_t,
                telemetry, gossip, max_telemetry, max_gossip, fwd_wait=0.0):
    """Piggyback header for one ring chunk (section 5.1).

    `fwd_wait` [OUR CHOICE] is how long the sender was blocked on the receive
    just before this send, so the receiver can tell inherited delay from delay
    that originates at the sender. `telemetry` and `gossip` are truncated to
    their bounds here so a caller can never grow the header with N or with the
    gossip backlog.
    """
    return {
        "src": src,
        "step": int(step),
        "phase": phase,
        "chunk_idx": int(chunk_idx),
        "send_ts": float(send_ts),
        "compute_t": float(compute_t),
        "wait_t": float(wait_t),
        "fwd_wait": float(fwd_wait),
        "telemetry": list(telemetry)[:max_telemetry],
        "gossip": list(gossip)[:max_gossip],
    }


def hello(src):
    """First frame on a new ring connection, so the receiver knows its peer."""
    return {"type": "HELLO", "src": src}


def ping_req(about, requester, step_window):
    return {"type": PING_REQ, "about": about, "requester": requester,
            "step_window": [int(step_window[0]), int(step_window[1])]}


def ack(about, src, agree, evidence):
    return {"type": ACK, "about": about, "src": src, "agree": bool(agree),
            "evidence": evidence}


def nack(about, src, reason):
    return {"type": NACK, "about": about, "src": src, "reason": reason}


def encode_chunk(array):
    return np.ascontiguousarray(array, dtype=CHUNK_DTYPE).tobytes()


def decode_chunk(payload):
    return np.frombuffer(payload, dtype=CHUNK_DTYPE).copy()


def encode_frame(header, payload=b""):
    body = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_HEADER_BYTES:
        raise ValueError(f"header too large: {len(body)} bytes")
    return _PREFIX.pack(len(body), len(payload)) + body + payload


def decode_frame(data):
    """Decode one complete frame held in `data`. Returns (header, payload)."""
    header_len, payload_len = _PREFIX.unpack_from(data, 0)
    start = _PREFIX.size
    header = json.loads(data[start:start + header_len].decode("utf-8"))
    payload = bytes(data[start + header_len:start + header_len + payload_len])
    return header, payload


def send_frame(sock, header, payload=b""):
    sock.sendall(encode_frame(header, payload))


def recv_frame(sock):
    """Read one frame from a stream socket. Raises ConnectionError on EOF."""
    header_len, payload_len = _PREFIX.unpack(_recv_exact(sock, _PREFIX.size))
    if header_len > MAX_HEADER_BYTES or payload_len > MAX_PAYLOAD_BYTES:
        raise ConnectionError("frame exceeds size limits")
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    payload = _recv_exact(sock, payload_len) if payload_len else b""
    return header, payload


def _recv_exact(sock, size):
    buf = bytearray(size)
    view = memoryview(buf)
    got = 0
    while got < size:
        n = sock.recv_into(view[got:], size - got)
        if n == 0:
            raise ConnectionError("peer closed the connection")
        got += n
    return bytes(buf)


def request(host, port, message, timeout):
    """One short-lived side-channel exchange: send `message`, read one reply.

    Returns the reply header, or None on any connect/read failure or timeout
    (the caller counts None as a missing reply).
    """
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            send_frame(sock, message)
            reply, _ = recv_frame(sock)
            return reply
    except (OSError, ConnectionError, ValueError):
        return None
