"""A pretend ring of detectors, for testing without torch, threads or processes.

Each step it works out, from the compute times you give it, how long every worker would wait in
every round of the ring all-reduce, passes the gossip along the ring round by round, and hands
the detectors the waits. Real timing noise is simulated by the random number generator you give it.
"""

from swim_straggler.detector.swim import StragglerDetector

TRANSFER_SECONDS = 0.0005  # time a message needs to cross the ring


def waits_in_ring(compute_seconds_per_rank):
    """Seconds each worker waits in each round (2 x (N - 1) rounds), if all start the step together."""
    world_size = len(compute_seconds_per_rank)
    ready_time = list(compute_seconds_per_rank)  # when each worker can send in the next round
    waits_per_rank = [[] for _ in range(world_size)]
    for _ in range(2 * (world_size - 1)):
        arrival_times = [ready_time[(rank - 1) % world_size] + TRANSFER_SECONDS for rank in range(world_size)]
        for rank in range(world_size):
            waits_per_rank[rank].append(max(0.0, arrival_times[rank] - ready_time[rank]))
        ready_time = [max(ready_time[rank], arrival_times[rank]) for rank in range(world_size)]
    return waits_per_rank


class SyntheticRing:
    def __init__(self, world_size, detector_config, seed=0):
        self.world_size = world_size
        self.detectors = [StragglerDetector(rank, world_size, detector_config, seed) for rank in range(world_size)]
        self.all_events = []  # (rank, event) pairs

    def run_step(self, step, compute_seconds_per_rank):
        for rank, detector in enumerate(self.detectors):
            detector.start_step(step, compute_seconds_per_rank[rank])
        for _ in range(2 * (self.world_size - 1)):
            messages = [detector.make_gossip() for detector in self.detectors]
            for rank, message in enumerate(messages):
                self.detectors[(rank + 1) % self.world_size].receive_gossip(message, sender=rank)
        waits_per_rank = waits_in_ring(compute_seconds_per_rank)
        for rank, detector in enumerate(self.detectors):
            for event in detector.finish_step(step, waits_per_rank[rank]):
                self.all_events.append((rank, event))

    def run(self, steps, compute_seconds_function):
        """compute_seconds_function(step, rank) -> seconds. Returns all events so far."""
        for step in range(steps):
            self.run_step(step, [compute_seconds_function(step, rank) for rank in range(self.world_size)])
        return self.all_events

    def events_with_action(self, action):
        return [(rank, event) for rank, event in self.all_events if event["action"] == action]
