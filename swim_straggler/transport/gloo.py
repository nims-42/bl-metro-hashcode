"""torch.distributed transport using the gloo backend (one process per worker)."""

import os
import sys
from datetime import timedelta

import torch
from torch import distributed as torch_distributed

from swim_straggler.transport.base import Transport

LOOPBACK_ADDRESSES = ("127.0.0.1", "localhost")


def prefer_loopback_network_interface(master_address):
    """On Linux, tell gloo to use the loopback interface when every worker runs on this computer.

    Without this, gloo sometimes fails on machines (for example WSL) where the host name
    does not resolve to a usable network address.
    """
    if master_address in LOOPBACK_ADDRESSES and sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


class GlooTransport(Transport):
    def __init__(self, rank, world_size, master_address, master_port, timeout_seconds=120):
        prefer_loopback_network_interface(master_address)
        torch_distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_address}:{master_port}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
        self.rank = rank
        self.world_size = world_size
        self._unfinished_sends = []

    def send_to_next(self, payload):
        # The copy must stay alive until the send finishes, so it is kept next to its request.
        payload_copy = payload.detach().clone()
        request = torch_distributed.isend(payload_copy, dst=self.next_rank)
        self._unfinished_sends.append((request, payload_copy))

    def receive_from_previous(self, number_of_elements):
        buffer = torch.empty(number_of_elements, dtype=torch.float32)
        torch_distributed.recv(buffer, src=self.previous_rank)
        return buffer

    def flush(self):
        for request, _payload_copy in self._unfinished_sends:
            request.wait()
        self._unfinished_sends = []

    def close(self):
        if torch_distributed.is_initialized():
            torch_distributed.destroy_process_group()
