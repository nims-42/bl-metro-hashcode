"""Hand-written ring all-reduce (sum) that times how long this rank waits for its predecessor.

The vector is split into `world_size` chunks. Rank r does two phases:

  Reduce-scatter, world_size - 1 rounds: in round s, send chunk (r - s), receive chunk (r - s - 1)
  and ADD it to the local copy. Afterwards each rank owns one fully summed chunk.

  All-gather, world_size - 1 rounds: in round s, send chunk (r + 1 - s), receive chunk (r - s)
  and OVERWRITE the local copy. Afterwards every rank holds the complete result.
"""

import time

import torch

from swim_straggler.allreduce.header import Header


def chunk_bounds(vector_length, number_of_chunks):
    """Split a vector into contiguous (start, end) ranges whose sizes differ by at most one."""
    base_size, number_of_larger_chunks = divmod(vector_length, number_of_chunks)
    bounds = []
    start = 0
    for chunk_index in range(number_of_chunks):
        size = base_size + (1 if chunk_index < number_of_larger_chunks else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def ring_allreduce(transport, vector, codec, make_header=None, on_header_received=None):
    """Replace `vector` (one-dimensional float32) by its sum over all ranks, in place.

    make_header():                         builds the header sent with every chunk.
    on_header_received(header, round):     called for every header received from the predecessor.

    Returns the seconds this rank spent waiting for data in each round, measured locally.
    """
    world_size = transport.world_size
    rank = transport.rank
    bounds = chunk_bounds(vector.numel(), world_size)
    wait_seconds_per_round = []

    for is_reduce_phase in (True, False):
        for round_in_phase in range(world_size - 1):
            if is_reduce_phase:
                chunk_to_send = (rank - round_in_phase) % world_size
                chunk_to_receive = (rank - round_in_phase - 1) % world_size
            else:
                chunk_to_send = (rank + 1 - round_in_phase) % world_size
                chunk_to_receive = (rank - round_in_phase) % world_size

            header = make_header() if make_header else Header(sender=rank, step=0)
            send_start, send_end = bounds[chunk_to_send]
            transport.send_to_next(torch.cat([codec.encode(header), vector[send_start:send_end]]))

            receive_start, receive_end = bounds[chunk_to_receive]
            wait_started_at = time.perf_counter()
            message = transport.receive_from_previous(codec.size + receive_end - receive_start)
            wait_seconds_per_round.append(time.perf_counter() - wait_started_at)  # <- the detector's signal

            if on_header_received:
                on_header_received(codec.decode(message[: codec.size]), len(wait_seconds_per_round) - 1)
            received_chunk = message[codec.size :]
            if is_reduce_phase:
                vector[receive_start:receive_end] += received_chunk
            else:
                vector[receive_start:receive_end] = received_chunk

    transport.flush()
    return wait_seconds_per_round
