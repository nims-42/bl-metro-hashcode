"""Node entry point and training event loop (CONTEXT.md 3.1, 3.2 L1-L2, 4, 7, 8).

Run with `python -m src.node`. Every parameter has a default in `Config` and can
be overridden by an environment variable of the same name in upper case
(NODE_ID, N, RUN_ID, K, ALPHA, FAULT_TYPE, ...).

Each training step:
  1. compute gradients on the local shard (NumPy stand-in), timing compute_t;
  2. ring all-reduce: N-1 reduce-scatter + N-1 all-gather steps, every chunk
     carrying the piggyback header;
  3. update weights, hand the step's timings to the SWIM layer, log metrics.
"""

import os
import queue
import random
import socket
import threading
import time
from dataclasses import dataclass, fields

import numpy as np

from src import messages as msg
from src.lifeguard import Lifeguard
from src.membership import Membership
from src.metrics import Metrics
from src.swim import Swim

COMPUTE_FAULT = "compute"
NETWORK_FAULT = "network"


@dataclass
class Config:
    node_id: int = 0
    n: int = 5
    run_id: str = "local"
    results_dir: str = "/results"
    steps: int = 300
    seed: int = 0

    # Training stand-in
    grad_dim: int = 20000
    batch: int = 8
    lr: float = 0.01
    compute_ms: float = 100.0      # simulated GPU time per step [OUR CHOICE]
    jitter_ms: float = 0.0         # uniform random extra compute on every node

    # Detection (section 7)
    detection: bool = True         # False: plain all-reduce, for overhead runs
    k: int = 2
    lateness_factor: float = 1.1
    jitter_threshold: float = 0.10
    n_consec: int = 3
    window: int = 20               # steps of history used for medians [OUR CHOICE]
    warmup_steps: int = 10         # startup skew is not lateness [OUR CHOICE]
    lhm_max: int = 8
    base_probe_timeout: float = 0.5
    probe_interval: float = 1.0
    k_confirm: int = 3
    alpha: float = 5.0
    beta: float = 6.0
    gossip_lambda: float = 3.0
    gossip_max: int = 6

    # Network
    ring_port: int = 7000
    ping_port: int = 7001
    host_template: str = "gpu{i}"  # Docker DNS name of ring position i
    port_stride: int = 0           # >0 gives each node its own ports (all on one host)
    connect_timeout: float = 60.0

    # Fault injection (section 8)
    fault_node: str = ""           # node ids this fault applies to, e.g. "2" or "gpu2,gpu3"
    fault_type: str = ""           # compute | network
    fault_delay_ms: float = 0.0    # compute: per step; network: per sent chunk
    fault_start_step: int = 0
    fault_end_step: int = -1       # -1: until the end of the run

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        values = {}
        for f in fields(cls):
            raw = env.get(f.name.upper())
            if raw is None or raw == "":
                continue
            if f.type is bool or f.type == "bool":
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif f.name == "node_id":
                values[f.name] = msg.node_index(raw)
            elif f.type in (int, "int"):
                values[f.name] = int(raw)
            elif f.type in (float, "float"):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        return cls(**values)

    def validate(self):
        if self.n < 3:
            raise ValueError(f"N must be at least 3, got {self.n}")
        if not 0 <= self.node_id < self.n:
            raise ValueError(f"NODE_ID {self.node_id} outside 0..{self.n - 1}")
        if not 1 <= self.k <= self.n - 2:
            raise ValueError(f"k must satisfy 1 <= k <= N-2, got k={self.k}, N={self.n}")
        if self.fault_type not in ("", COMPUTE_FAULT, NETWORK_FAULT):
            raise ValueError(f"FAULT_TYPE must be compute or network, got {self.fault_type!r}")
        return self

    @property
    def name(self):
        return msg.node_name(self.node_id)

    def ids(self):
        return [msg.node_name(i) for i in range(self.n)]

    def host(self, index):
        return self.host_template.format(i=index)

    def ring_port_of(self, index):
        return self.ring_port + index * self.port_stride

    def ping_port_of(self, index):
        return self.ping_port + index * self.port_stride

    def fault_at(self, step):
        """The fault type active on this node at `step`, or None."""
        if not self.fault_type or not self.fault_node:
            return None
        targets = {msg.node_index(t) for t in self.fault_node.split(",") if t.strip()}
        if self.node_id not in targets:
            return None
        if step < self.fault_start_step:
            return None
        if self.fault_end_step >= 0 and step >= self.fault_end_step:
            return None
        return self.fault_type


def ring_allreduce(vec, n, idx, send, recv, on_header=None, clock=time.monotonic,
                   prev_last_wait=0.0):
    """Sum `vec` across the ring (section 4).

    `send(phase, chunk_idx, array, fwd_wait)` must not block (frames are
    queued); `fwd_wait` is this node's blocked time on the receive just
    before that send (`prev_last_wait` for the first send of the step).
    `recv()` blocks for the predecessor's next frame and returns
    (header, array).

    Returns (summed vector, lags, excess): `lags` is the blocked time per
    received chunk (their sum is wait_t); `excess` is the part of each that
    originates at the predecessor, lag minus the predecessor's own `fwd_wait`
    for that chunk (their sum is pred_lag).
    """
    d = vec.size
    size = -(-d // n)
    chunks = np.zeros(size * n, dtype=msg.CHUNK_DTYPE)
    chunks[:d] = vec
    chunks = chunks.reshape(n, size)
    lags = []
    excess = []

    def last_wait():
        return lags[-1] if lags else prev_last_wait

    def receive(expected_phase, expected_idx):
        t0 = clock()
        header, data = recv()
        lags.append(clock() - t0)
        excess.append(max(0.0, lags[-1] - header.get("fwd_wait", 0.0)))
        if header.get("phase") != expected_phase or header.get("chunk_idx") != expected_idx:
            raise RuntimeError(f"ring out of order: got {header.get('phase')}/"
                               f"{header.get('chunk_idx')}, want {expected_phase}/{expected_idx}")
        if on_header is not None:
            on_header(header)
        return data

    for s in range(n - 1):
        send(msg.RS, (idx - s) % n, chunks[(idx - s) % n], last_wait())
        r = (idx - s - 1) % n
        chunks[r] += receive(msg.RS, r)
    for s in range(n - 1):
        send(msg.AG, (idx + 1 - s) % n, chunks[(idx + 1 - s) % n], last_wait())
        r = (idx - s) % n
        chunks[r] = receive(msg.AG, r)
    return chunks.reshape(-1)[:d].copy(), lags, excess


class Trainer:
    """NumPy stand-in for a training step: linear regression on a local shard."""

    def __init__(self, cfg, rng):
        self.x = rng.standard_normal((cfg.batch, cfg.grad_dim))
        self.y = rng.standard_normal(cfg.batch)
        self.w = np.zeros(cfg.grad_dim)
        self.lr = cfg.lr

    def gradient(self):
        return self.x.T @ (self.x @ self.w - self.y) / len(self.y)

    def update(self, avg_grad):
        self.w -= self.lr * avg_grad


class TcpRing:
    """Persistent TCP links: one to the successor, one from the predecessor.
    Sends go through a queue and a sender thread, which is where the
    network-fault delay is applied."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.idx = cfg.node_id
        self.succ_idx = (self.idx + 1) % cfg.n
        self._out = None
        self._in = None
        self._queue = queue.Queue()
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._error = None

    def connect(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", self.cfg.ring_port_of(self.idx)))
        listener.listen(1)
        accepted = {}
        acceptor = threading.Thread(target=lambda: accepted.update(sock=listener.accept()[0]),
                                    daemon=True)
        acceptor.start()

        self._out = _connect_with_retry(self.cfg.host(self.succ_idx),
                                        self.cfg.ring_port_of(self.succ_idx),
                                        self.cfg.connect_timeout)
        msg.send_frame(self._out, msg.hello(self.cfg.name))

        acceptor.join(self.cfg.connect_timeout)
        listener.close()
        if "sock" not in accepted:
            raise TimeoutError("predecessor never connected")
        self._in = accepted["sock"]
        _tune(self._in)
        hello, _ = msg.recv_frame(self._in)
        expected = msg.node_name((self.idx - 1) % self.cfg.n)
        if hello.get("src") != expected:
            raise RuntimeError(f"expected predecessor {expected}, got {hello.get('src')}")
        self._sender.start()

    def send(self, header, payload, delay=0.0):
        if self._error:
            raise self._error
        self._queue.put((msg.encode_frame(header, payload), delay))

    def recv(self):
        return msg.recv_frame(self._in)

    def flush(self):
        self._queue.join()

    def close(self):
        self._queue.put(None)
        for sock in (self._out, self._in):
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                sock.close()

    def _send_loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                frame, delay = item
                if delay > 0:
                    time.sleep(delay)
                self._out.sendall(frame)
            except OSError as exc:
                self._error = exc
            finally:
                self._queue.task_done()


class PingServer:
    """TCP side channel for PING_REQ (section 5.3): one request per connection."""

    def __init__(self, cfg, handler):
        self._handler = handler
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", cfg.ping_port_of(cfg.node_id)))
        self._sock.listen(16)
        self._timeout = cfg.base_probe_timeout * (cfg.lhm_max + 1)
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn:
            try:
                conn.settimeout(self._timeout)
                request, _ = msg.recv_frame(conn)
                if request.get("type") == msg.PING_REQ:
                    msg.send_frame(conn, self._handler(request))
            except (OSError, ConnectionError, ValueError):
                pass

    def close(self):
        self._sock.close()


def _tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _connect_with_retry(host, port, timeout):
    """Retry with backoff until the peer is up, so start order does not matter."""
    deadline = time.monotonic() + timeout
    delay = 0.05
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            sock.settimeout(None)
            _tune(sock)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def build_swim(cfg, metrics, request_fn=None, rng=None, spawn=None, clock=time.monotonic):
    ids = cfg.ids()
    lifeguard = Lifeguard(cfg.n, lhm_max=cfg.lhm_max, base_probe_timeout=cfg.base_probe_timeout,
                          probe_interval=cfg.probe_interval, alpha=cfg.alpha, beta=cfg.beta,
                          k_confirm=cfg.k_confirm, lateness_factor=cfg.lateness_factor,
                          jitter_threshold=cfg.jitter_threshold, clock=clock)
    membership = Membership(cfg.name, ids, clock=clock)
    return Swim(cfg, ids, membership, lifeguard, metrics, request_fn=request_fn, rng=rng,
                spawn=spawn, clock=clock)


def run(cfg):
    cfg.validate()
    rng = np.random.default_rng(cfg.seed * 1000 + cfg.node_id)
    jitter_rng = random.Random(cfg.seed * 1000 + cfg.node_id)
    metrics = Metrics(cfg.name, cfg.n, cfg.run_id, cfg.results_dir)

    def request(peer, message, timeout):
        i = msg.node_index(peer)
        return msg.request(cfg.host(i), cfg.ping_port_of(i), message, timeout)

    swim = build_swim(cfg, metrics, request_fn=request, rng=random.Random(cfg.seed + cfg.node_id))
    server = PingServer(cfg, swim.handle_ping_req)
    ring = TcpRing(cfg)
    ring.connect()
    trainer = Trainer(cfg, rng)
    clock = time.monotonic
    prev_wait = 0.0
    prev_last_wait = 0.0
    prev_fault = None

    try:
        for step in range(cfg.steps):
            if step == cfg.steps - 1:
                swim.leave()
            fault = cfg.fault_at(step)
            if fault != prev_fault:
                metrics.event("FAULT_START" if fault else "FAULT_END", cfg.name,
                              fault or prev_fault, cfg.name, delay_ms=cfg.fault_delay_ms)
                prev_fault = fault
            delay = cfg.fault_delay_ms / 1000.0

            step_start = clock()
            grad = trainer.gradient()
            simulated = cfg.compute_ms + jitter_rng.uniform(0, cfg.jitter_ms)
            time.sleep(simulated / 1000.0)
            if fault == COMPUTE_FAULT:
                time.sleep(delay)
            compute_t = clock() - step_start
            send_delay = delay if fault == NETWORK_FAULT else 0.0

            def send(phase, chunk_idx, array, fwd_wait, step=step, compute_t=compute_t):
                header = swim.build_header(step, phase, chunk_idx, compute_t, prev_wait,
                                           fwd_wait)
                ring.send(header, msg.encode_chunk(array), send_delay)

            def recv():
                header, payload = ring.recv()
                return header, msg.decode_chunk(payload)

            total, lags, excess = ring_allreduce(grad, cfg.n, cfg.node_id, send, recv,
                                                 on_header=swim.on_header, clock=clock,
                                                 prev_last_wait=prev_last_wait)
            trainer.update(total / cfg.n)
            wait_t, pred_lag = sum(lags), sum(excess)
            step_t = clock() - step_start

            swim.end_step(step, compute_t, wait_t, pred_lag, step_t)
            states, lhm = swim.snapshot()
            metrics.write_step(step, compute_t, wait_t, pred_lag, step_t, lhm, states,
                               fault=fault, late_streak=swim.late_streak,
                               chunk_lags=[round(x, 6) for x in lags])
            prev_wait, prev_last_wait = wait_t, lags[-1]
        ring.flush()
    finally:
        ring.close()
        server.close()
        metrics.close()


def main():
    run(Config.from_env())


if __name__ == "__main__":
    main()
