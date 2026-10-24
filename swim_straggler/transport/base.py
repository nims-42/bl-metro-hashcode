"""What the ring all-reduce needs from a transport."""

from abc import ABC, abstractmethod

import torch


class Transport(ABC):
    rank: int
    world_size: int

    @abstractmethod
    def send_next(self, payload: torch.Tensor) -> None:
        """Send a 1-D float32 tensor to the next rank without blocking."""

    @abstractmethod
    def recv_prev(self, numel: int) -> torch.Tensor:
        """Block until `numel` floats arrive from the previous rank."""

    @abstractmethod
    def flush(self) -> None:
        """Block until all earlier sends have completed."""
