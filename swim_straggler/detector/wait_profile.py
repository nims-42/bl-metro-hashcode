"""Reading the culprit from how long this worker waited in each reduce-scatter round.

In the ring, a worker only ever receives from its predecessor. In reduce-scatter round `s` the
data it receives has been combined by the `s` workers before that predecessor, so a delay
caused by worker `culprit` first reaches worker `culprit + 1 + s` in round `s`. Reading this
backwards: if the longest wait happened in round `s`, the culprit is `rank - 1 - s`.
"""


def estimate_culprit(rank, world_size, wait_seconds_per_round, wait_floor_seconds):
    """Return the rank this worker believes is slow, or None if its waits show nothing unusual.

    `wait_seconds_per_round` is what ring_allreduce returned. Only the first world_size - 1
    entries (the reduce-scatter rounds) are used.
    """
    reduce_scatter_waits = list(wait_seconds_per_round[: world_size - 1])
    if len(reduce_scatter_waits) < 2:
        return None  # with a single round every wait looks like the largest one

    longest_wait = max(reduce_scatter_waits)
    round_of_longest_wait = reduce_scatter_waits.index(longest_wait)
    other_waits = reduce_scatter_waits[:round_of_longest_wait] + reduce_scatter_waits[round_of_longest_wait + 1 :]
    typical_wait = sum(other_waits) / len(other_waits)

    if longest_wait - typical_wait < wait_floor_seconds:
        return None  # all rounds waited about equally long: nobody stands out
    return (rank - 1 - round_of_longest_wait) % world_size
