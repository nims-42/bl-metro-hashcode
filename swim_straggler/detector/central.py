"""Baseline 1: a central monitor, in the style of Falcon (master-worker straggler detection).

Every worker reports its compute time every step; worker 0 alone compares them and decides.
There is no vote and no gossip: one place sees everything and decides alone. That is simple and
accurate, but worker 0 receives N reports every step, and if worker 0 fails nobody decides.

How the reports travel: each worker adds N numbers to the vector it all-reduces anyway, with its
own compute time in its own slot and zeros elsewhere. After the sum, the slots hold every
worker's compute time exactly. This is the same information a real report message to worker 0
would carry; only worker 0 looks at it.
"""

from swim_straggler.detector.base import Detector, StreakJudge, median_or_none, smooth
from swim_straggler.detector.table import ALIVE

MONITOR_RANK = 0


class CentralMonitor(Detector):
    needs_compute_reports = True

    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.is_monitor = rank == MONITOR_RANK
        self.smoothed_compute = [None] * world_size
        self.judge = StreakJudge(world_size, config.consecutive_steps, config.recovery_steps)

    def finish_step(self, step, wait_seconds_per_round, compute_seconds_of_every_worker=None):
        if not self.is_monitor:
            return []
        for worker, seconds in enumerate(compute_seconds_of_every_worker):
            self.smoothed_compute[worker] = smooth(self.smoothed_compute[worker], seconds, self.config.ewma_alpha)
        if step < self.config.warmup_steps:
            return []

        events = []
        for worker in range(self.world_size):
            reference = self.reference_compute(worker)
            looks_slow = reference is not None and self.smoothed_compute[worker] > self.config.factor * reference
            decision = self.judge.judge(worker, looks_slow, step)
            if decision:
                events.append({"action": decision, "subject": worker, "step": step, "decided_by": "central monitor"})
        return events

    def reference_compute(self, worker):
        """Median smoothed compute time of the other workers that are not already confirmed slow."""
        others = [self.smoothed_compute[other] for other in range(self.world_size)
                  if other != worker and self.judge.states[other] == ALIVE]
        return median_or_none(others)

    def straggler_ranks(self):
        return self.judge.straggler_ranks() if self.is_monitor else []
