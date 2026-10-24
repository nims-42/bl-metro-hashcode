"""What the ring all-reduce needs from a transport."""

from abc import ABC, abstractmethod

import torch


class Transport(ABC):
    rank: int
    world_size: int

    @property
    def next_rank(self):
        return (self.rank + 1) % self.world_size

    @property
    def previous_rank(self):
        return (self.rank - 1) % self.world_size

    @abstractmethod
    def send_to_next(self, payload: torch.Tensor) -> None:
        """Send a one-dimensional float32 tensor to the next rank without waiting for delivery."""

    @abstractmethod
    def receive_from_previous(self, number_of_elements: int) -> torch.Tensor:
        """Wait until `number_of_elements` floats arrive from the previous rank, then return them."""

    @abstractmethod
    def flush(self) -> None:
        """Wait until every earlier send has been delivered."""

    def close(self) -> None:
        """Release resources. Does nothing unless a transport needs cleanup."""
