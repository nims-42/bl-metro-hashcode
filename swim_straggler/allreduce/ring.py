"""Hand-written ring all-reduce (sum), timing how long this rank waits on its predecessor.

Rank r in a ring of n splits the vector into n chunks.
  Reduce-scatter (n-1 rounds): send chunk (r-s), receive chunk (r-s-1) and ADD it.
  All-gather     (n-1 rounds): send chunk (r+1-s), receive chunk (r-s) and OVERWRITE.
"""

import time

import torch

from swim_straggler.allreduce.header import Header


def chunk_bounds(numel, n):
    """Split numel into n contiguous (lo, hi) ranges whose sizes differ by at most one."""
    base, extra = divmod(numel, n)
    bounds, lo = [], 0
    for i in range(n):
        hi = lo + base + (1 if i < extra else 0)
        bounds.append((lo, hi))
        lo = hi
    return bounds


def ring_allreduce(transport, tensor, codec, make_header=None, on_header=None):
    """Sum `tensor` (1-D float32) across all ranks in place.

    make_header() builds the header sent with each chunk.
    on_header(header, round) is called for every header received.
    Returns the seconds spent blocked in recv, one entry per round (locally measured).
    """
    n, r = transport.world_size, transport.rank
    bounds = chunk_bounds(tensor.numel(), n)
    waits = []
    for reduce_phase in (True, False):
        for s in range(n - 1):
            send_i = (r - s) % n if reduce_phase else (r + 1 - s) % n
            recv_i = (r - s - 1) % n if reduce_phase else (r - s) % n

            header = make_header() if make_header else Header(sender=r, step=0)
            lo, hi = bounds[send_i]
            transport.send_next(torch.cat([codec.encode(header), tensor[lo:hi]]))

            lo, hi = bounds[recv_i]
            t0 = time.perf_counter()
            msg = transport.recv_prev(codec.size + hi - lo)
            waits.append(time.perf_counter() - t0)  # <- the signal the detector will use

            if on_header:
                on_header(codec.decode(msg[: codec.size]), len(waits) - 1)
            if reduce_phase:
                tensor[lo:hi] += msg[codec.size :]
            else:
                tensor[lo:hi] = msg[codec.size :]
    transport.flush()
    return waits
