import time

import pytest
import torch

from swim_straggler.allreduce.header import Header, HeaderCodec
from swim_straggler.allreduce.ring import chunk_bounds, ring_allreduce
from swim_straggler.transport.pipes import run_ranks


def run_allreduce(n, numel, slow_rank=None, delay=0.0):
    """Every rank reduces a random vector; returns (input, result, waits, headers_seen) per rank."""

    def task(t):
        x = torch.randn(numel, generator=torch.Generator().manual_seed(t.rank))
        original = x.clone()
        if t.rank == slow_rank:
            time.sleep(delay)  # this rank arrives late, like a slow worker
        seen = []
        waits = ring_allreduce(t, x, HeaderCodec(), make_header=lambda: Header(t.rank, 7),
                               on_header=lambda h, i: seen.append(h))
        return original, x, waits, seen

    return run_ranks(n, task)


@pytest.mark.parametrize("n", [1, 2, 3, 8])
@pytest.mark.parametrize("numel", [1, 7, 100, 1003])  # smaller than n, and not divisible by n
def test_ring_equals_sum(n, numel):
    results = run_allreduce(n, numel)
    expected = sum(r[0] for r in results)
    for _, result, _, _ in results:
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_headers_come_from_predecessor():
    n = 4
    for rank, (_, _, waits, seen) in enumerate(run_allreduce(n, 50)):
        assert len(seen) == len(waits) == 2 * (n - 1)
        assert all(h.sender == (rank - 1) % n and h.step == 7 for h in seen)


def test_slow_rank_is_the_only_one_that_does_not_wait():
    slow, delay = 2, 0.3
    waits = [sum(r[2]) for r in run_allreduce(4, 4000, slow_rank=slow, delay=delay)]
    others = [w for i, w in enumerate(waits) if i != slow]
    assert min(others) > 0.3 * delay, waits  # loose: first-call torch warm-up can eat part of the delay
    assert waits[slow] < 0.25 * min(others), waits


def test_chunk_bounds():
    assert chunk_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert chunk_bounds(2, 4) == [(0, 1), (1, 2), (2, 2), (2, 2)]
