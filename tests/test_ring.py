import time

import pytest
import torch

from swim_straggler.allreduce.header import Header, HeaderCodec
from swim_straggler.allreduce.ring import chunk_bounds, ring_allreduce
from swim_straggler.transport.pipes import run_ranks_in_threads


def run_allreduce(world_size, vector_length, slow_rank=None, delay_seconds=0.0):
    """Every rank reduces its own random vector. Returns one tuple per rank:
    (original vector, reduced vector, wait seconds per round, headers received)."""

    def work_of_one_rank(transport):
        generator = torch.Generator().manual_seed(transport.rank)
        vector = torch.randn(vector_length, generator=generator)
        original_vector = vector.clone()
        if transport.rank == slow_rank:
            time.sleep(delay_seconds)  # this rank arrives late, like a slow worker
        received_headers = []
        wait_seconds_per_round = ring_allreduce(
            transport,
            vector,
            HeaderCodec(),
            make_header=lambda: Header(sender=transport.rank, step=7),
            on_header_received=lambda header, round_number: received_headers.append(header),
        )
        return original_vector, vector, wait_seconds_per_round, received_headers

    return run_ranks_in_threads(world_size, work_of_one_rank)


@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
@pytest.mark.parametrize("vector_length", [1, 7, 100, 1003])  # shorter than world_size, and not divisible by it
def test_ring_equals_plain_sum(world_size, vector_length):
    results = run_allreduce(world_size, vector_length)
    expected_sum = sum(original_vector for original_vector, _, _, _ in results)
    for _, reduced_vector, _, _ in results:
        torch.testing.assert_close(reduced_vector, expected_sum, rtol=1e-4, atol=1e-4)


def test_headers_come_from_the_predecessor():
    world_size = 4
    results = run_allreduce(world_size, 50)
    for rank, (_, _, wait_seconds_per_round, received_headers) in enumerate(results):
        assert len(received_headers) == len(wait_seconds_per_round) == 2 * (world_size - 1)
        for header in received_headers:
            assert header.sender == (rank - 1) % world_size
            assert header.step == 7


def test_slow_rank_is_the_only_one_that_does_not_wait():
    slow_rank, delay_seconds = 2, 0.3
    results = run_allreduce(4, 4000, slow_rank=slow_rank, delay_seconds=delay_seconds)
    total_waits = [sum(wait_seconds_per_round) for _, _, wait_seconds_per_round, _ in results]
    other_waits = [wait for rank, wait in enumerate(total_waits) if rank != slow_rank]
    # Loose threshold: the first call into PyTorch can eat part of the delay.
    assert min(other_waits) > 0.3 * delay_seconds, total_waits
    assert total_waits[slow_rank] < 0.25 * min(other_waits), total_waits


def test_chunk_bounds():
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert chunk_bounds(2, 4) == [(0, 1), (1, 2), (2, 2), (2, 2)]
