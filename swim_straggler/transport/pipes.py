"""Pipe transport: every rank is a thread in one process.

Used by the tests and as a fallback when torch.distributed is not wanted.
"""

import queue
import threading
from multiprocessing import Pipe

import torch

from swim_straggler.transport.base import Transport

BYTES_PER_FLOAT32 = 4


class PipeTransport(Transport):
    def __init__(self, rank, world_size, receive_connection, send_connection):
        self.rank = rank
        self.world_size = world_size
        self._receive_connection = receive_connection
        self._send_connection = send_connection
        self._outgoing_messages = queue.Queue()
        # A helper thread does the actual sending, so a full pipe can never block the ring.
        sender_thread = threading.Thread(target=self._send_forever, daemon=True)
        sender_thread.start()

    def _send_forever(self):
        while True:
            message_bytes = self._outgoing_messages.get()
            self._send_connection.send_bytes(message_bytes)
            self._outgoing_messages.task_done()

    def send_to_next(self, payload):
        # tobytes() copies the data, so the caller may modify the tensor straight away.
        self._outgoing_messages.put(payload.detach().numpy().tobytes())

    def receive_from_previous(self, number_of_elements):
        message_bytes = self._receive_connection.recv_bytes()
        expected_bytes = number_of_elements * BYTES_PER_FLOAT32
        if len(message_bytes) != expected_bytes:
            raise RuntimeError(f"rank {self.rank}: expected {expected_bytes} bytes, received {len(message_bytes)}")
        return torch.frombuffer(bytearray(message_bytes), dtype=torch.float32)

    def flush(self):
        self._outgoing_messages.join()


def run_ranks_in_threads(world_size, rank_function, timeout_seconds=60.0):
    """Run rank_function(transport) once per rank, each in its own thread.

    Returns the results in rank order. Raises if any rank fails or if the ranks do not
    finish within the timeout (which usually means a deadlock).
    """
    # pipes[i] carries messages from rank i to rank i + 1.
    pipes = [Pipe(duplex=False) for _ in range(world_size)]
    transports = []
    for rank in range(world_size):
        receive_connection = pipes[(rank - 1) % world_size][0]
        send_connection = pipes[rank][1]
        transports.append(PipeTransport(rank, world_size, receive_connection, send_connection))

    results = [None] * world_size
    error_messages = []

    def run_one_rank(rank):
        try:
            results[rank] = rank_function(transports[rank])
        except Exception as error:
            error_messages.append(f"rank {rank}: {error!r}")

    threads = [threading.Thread(target=run_one_rank, args=(rank,), daemon=True) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout_seconds)

    if error_messages:
        raise RuntimeError("; ".join(error_messages))
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("the ranks did not finish in time (deadlock?)")
    return results
