"""Small fixed-size header sent with every ring chunk, encoded as a float32 tensor.

Layout: [sender, step, compute_time, wait_time, transfer_time, *gossip]

All times are durations in seconds, measured by the sender on its own clock.
They are never timestamps, because clocks of different processes are not synchronised.
The sender and step numbers are stored as float32, so they are exact only up to 2**24.
"""

from dataclasses import dataclass

import torch

NUMBER_OF_FIXED_FIELDS = 5


@dataclass(frozen=True)
class Header:
    sender: int
    step: int
    compute_time: float = 0.0
    wait_time: float = 0.0
    transfer_time: float = 0.0  # placeholder until the link-delay detector exists
    gossip: tuple = ()  # opaque here; the detector defines its content later


class HeaderCodec:
    def __init__(self, gossip_length=0):
        self.gossip_length = gossip_length
        self.size = NUMBER_OF_FIXED_FIELDS + gossip_length

    def encode(self, header):
        if len(header.gossip) > self.gossip_length:
            raise ValueError("gossip block is longer than the codec allows")
        padding = [0.0] * (self.gossip_length - len(header.gossip))
        values = [header.sender, header.step, header.compute_time, header.wait_time, header.transfer_time]
        values += list(header.gossip) + padding
        return torch.tensor(values, dtype=torch.float32)

    def decode(self, encoded_header):
        if encoded_header.numel() != self.size:
            raise ValueError(f"expected {self.size} floats, received {encoded_header.numel()}")
        values = encoded_header.tolist()
        return Header(
            sender=int(values[0]),
            step=int(values[1]),
            compute_time=values[2],
            wait_time=values[3],
            transfer_time=values[4],
            gossip=tuple(values[NUMBER_OF_FIXED_FIELDS:]),
        )
