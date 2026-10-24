"""Small fixed-size header sent with every ring chunk, encoded as a float32 tensor.

Layout: [sender, step, compute_time, wait_time, transfer_time, *gossip].
Times are durations measured locally by the sender, never timestamps.
sender/step are stored as float32, so they are exact only up to 2**24.
"""

from dataclasses import dataclass

import torch

N_FIELDS = 5


@dataclass(frozen=True)
class Header:
    sender: int
    step: int
    compute_time: float = 0.0
    wait_time: float = 0.0
    transfer_time: float = 0.0
    gossip: tuple = ()  # opaque here; defined by the detector later


class HeaderCodec:
    def __init__(self, gossip_len=0):
        self.gossip_len = gossip_len
        self.size = N_FIELDS + gossip_len

    def encode(self, h):
        if len(h.gossip) > self.gossip_len:
            raise ValueError("gossip block too long")
        gossip = list(h.gossip) + [0.0] * (self.gossip_len - len(h.gossip))
        fields = [h.sender, h.step, h.compute_time, h.wait_time, h.transfer_time]
        return torch.tensor(fields + gossip, dtype=torch.float32)

    def decode(self, t):
        if t.numel() != self.size:
            raise ValueError(f"expected {self.size} floats, got {t.numel()}")
        v = t.tolist()
        return Header(int(v[0]), int(v[1]), v[2], v[3], v[4], tuple(v[N_FIELDS:]))
