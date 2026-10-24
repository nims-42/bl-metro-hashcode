"""The same ring all-reduce, now over real torch.distributed (gloo) with one process per worker."""

import pytest
import torch
from torch import multiprocessing as torch_multiprocessing

from swim_straggler.allreduce.header import HeaderCodec
from swim_straggler.allreduce.ring import ring_allreduce
from swim_straggler.train.launcher import find_free_port
from swim_straggler.transport.gloo import GlooTransport


def check_ring_in_one_process(rank, world_size, vector_length, master_port):
    """Runs inside each spawned process; raises if the result differs from the plain sum."""
    torch.set_num_threads(1)
    transport = GlooTransport(rank, world_size, "127.0.0.1", master_port)
    try:
        all_vectors = [
            torch.randn(vector_length, generator=torch.Generator().manual_seed(other_rank))
            for other_rank in range(world_size)
        ]
        expected_sum = sum(all_vectors)
        my_vector = all_vectors[rank].clone()
        ring_allreduce(transport, my_vector, HeaderCodec())
        torch.testing.assert_close(my_vector, expected_sum, rtol=1e-4, atol=1e-4)
    finally:
        transport.close()


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("vector_length", [7, 1003])
def test_ring_equals_plain_sum_on_gloo(world_size, vector_length):
    torch_multiprocessing.spawn(
        check_ring_in_one_process,
        args=(world_size, vector_length, find_free_port()),
        nprocs=world_size,
        join=True,
    )
