"""The membership table: what one worker believes about every worker in the group."""

import statistics
from dataclasses import dataclass

ALIVE = "alive"  # no evidence of a problem
SUSPECT = "suspect"  # I suspect this worker and am asking other workers to confirm
STRAGGLER = "straggler"  # confirmed slow


@dataclass
class TableRow:
    rank: int
    step: int = -1  # the step this row describes; -1 means nothing has been heard yet
    compute_seconds: float = 0.0  # smoothed compute time of this worker
    # The worker's latest answer to "do you also see `probe_subject` computing slowly?"
    probe_subject: int = -1  # -1 means no answer
    probe_answer_step: int = -1  # the step at which the worker last gave this answer
    probe_says_slow: bool = False


class MembershipTable:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rows = [TableRow(rank) for rank in range(world_size)]
        self.states = [ALIVE] * world_size

    def update_own_compute(self, rank, step, compute_seconds, ewma_alpha):
        """Record this worker's own compute time, smoothed so that one noisy step does not matter."""
        row = self.rows[rank]
        if row.step < 0:
            row.compute_seconds = compute_seconds
        else:
            row.compute_seconds = ewma_alpha * compute_seconds + (1 - ewma_alpha) * row.compute_seconds
        row.step = step

    def merge_row(self, incoming_row):
        """Adopt a row received by gossip, but only if it is newer than what we already have."""
        known_row = self.rows[incoming_row.rank]
        if incoming_row.step > known_row.step:
            self.rows[incoming_row.rank] = incoming_row
            return True
        return False

    def median_compute_seconds(self, ranks_to_ignore=()):
        """Median smoothed compute time over the workers we have heard from, or None if fewer than two."""
        values = [
            row.compute_seconds
            for row in self.rows
            if row.step >= 0 and row.rank not in ranks_to_ignore
        ]
        if len(values) < 2:
            return None
        return statistics.median(values)

    def reference_median(self, subject):
        """The typical compute time of the other, still healthy-looking workers.

        The subject itself is left out, otherwise a slow worker would drag the median up and
        hide its own slowness. Workers already suspected or confirmed slow are left out too.
        """
        ranks_to_ignore = {subject} | {rank for rank in range(self.world_size) if self.states[rank] != ALIVE}
        return self.median_compute_seconds(ranks_to_ignore)

    def looks_slow(self, subject, factor):
        """True if the subject's compute time is more than `factor` times the typical compute time."""
        subject_row = self.rows[subject]
        reference = self.reference_median(subject)
        if subject_row.step < 0 or reference is None or reference <= 0:
            return False
        return subject_row.compute_seconds > factor * reference
