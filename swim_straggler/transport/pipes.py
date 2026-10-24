"""Pipe transport: ranks are threads in one process. Used for tests; gloo comes in Milestone 2."""

import queue
import threading
from multiprocessing import Pipe

import torch

from swim_straggler.transport.base import Transport


class PipeTransport(Transport):
    def __init__(self, rank, world_size, recv_conn, send_conn):
        self.rank, self.world_size = rank, world_size
        self._recv, self._send = recv_conn, send_conn
        self._outbox = queue.Queue()
        # Sending happens on a helper thread so a full pipe can never deadlock the ring.
        threading.Thread(target=self._send_loop, daemon=True).start()

    def _send_loop(self):
        while True:
            self._send.send_bytes(self._outbox.get())
            self._outbox.task_done()

    def send_next(self, payload):
        self._outbox.put(payload.numpy().tobytes())  # tobytes copies, so later edits are safe

    def recv_prev(self, numel):
        data = self._recv.recv_bytes()
        assert len(data) == numel * 4, f"expected {numel * 4} bytes, got {len(data)}"
        return torch.frombuffer(bytearray(data), dtype=torch.float32)

    def flush(self):
        self._outbox.join()


def run_ranks(world_size, fn, timeout=30.0):
    """Run `fn(transport)` on every rank (one thread each); return the results in rank order."""
    pipes = [Pipe(duplex=False) for _ in range(world_size)]  # pipes[i] carries rank i -> i+1
    transports = [
        PipeTransport(r, world_size, pipes[(r - 1) % world_size][0], pipes[r][1]) for r in range(world_size)
    ]
    results, errors = [None] * world_size, []

    def work(r):
        try:
            results[r] = fn(transports[r])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"rank {r}: {exc!r}")

    threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    if errors:
        raise RuntimeError("; ".join(errors))
    if any(t.is_alive() for t in threads):
        raise TimeoutError("ranks did not finish (deadlock?)")
    return results
