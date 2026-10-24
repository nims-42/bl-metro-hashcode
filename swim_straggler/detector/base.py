"""What every detector looks like to the training loop.

The training loop calls these hooks in this order every step:
    start_step(step, compute_seconds)                       after computing, before the all-reduce
    make_gossip() / receive_gossip(numbers, sender)          in every all-reduce round
    finish_step(step, wait_seconds_per_round, reports)       after the all-reduce
    after_update(step, step_seconds)                         when the step is over
Each detector uses only the hooks it needs; the others do nothing. The `finish_step` and
`after_update` hooks return the new events (a list of dictionaries).

Two kinds of events:
  * decisions (suspected, confirmed, rejected, cleared, learned_...): always returned;
  * trace events (marked "trace": True): the reasoning behind them, step by step. They are only
    returned when detector.trace is on, because they make the log much longer.
"""

import statistics

from swim_straggler.detector.table import ALIVE, STRAGGLER


class Detector:
    gossip_length = 0  # numbers this detector adds to every message header
    needs_compute_reports = False  # True: the all-reduce must also deliver every worker's compute time

    def start_step(self, step, compute_seconds):
        pass

    def make_gossip(self):
        return []

    def receive_gossip(self, numbers, sender=None):
        pass

    def gossip_contents(self):  # the simulator's way to pass gossip without numbers (see swim.py)
        return [], []

    def absorb_gossip(self, rows, notices, sender=None):
        pass

    def finish_step(self, step, wait_seconds_per_round, compute_seconds_of_every_worker=None):
        return []

    def after_update(self, step, step_seconds):
        return []

    def straggler_ranks(self):
        return []


def trace_event(action, subject, step, **details):
    """An event that explains the reasoning of a detector (only logged when detector.trace is on)."""
    return {"trace": True, "action": action, "subject": subject, "step": step, **details}


def smooth(previous_value, new_value, ewma_alpha):
    """Exponentially weighted moving average; the first value is taken as it is."""
    if previous_value is None:
        return new_value
    return ewma_alpha * new_value + (1 - ewma_alpha) * previous_value


class StreakJudge:
    """Turns a yes/no "looks slow" answer per step into confirmed / cleared decisions.

    Confirm after `consecutive_steps` slow steps in a row; clear a confirmed worker after
    `recovery_steps` normal steps in a row. Shared by both baselines so they behave alike.
    """

    def __init__(self, world_size, consecutive_steps, recovery_steps):
        self.consecutive_steps = consecutive_steps
        self.recovery_steps = recovery_steps
        self.states = [ALIVE] * world_size
        self.slow_streak = [0] * world_size
        self.normal_streak = [0] * world_size

    def judge(self, worker, looks_slow, step):
        """Returns "confirmed", "cleared" or None."""
        if looks_slow:
            self.slow_streak[worker] += 1
            self.normal_streak[worker] = 0
        else:
            self.slow_streak[worker] = 0
            self.normal_streak[worker] += 1
        if self.states[worker] == ALIVE and self.slow_streak[worker] >= self.consecutive_steps:
            self.states[worker] = STRAGGLER
            return "confirmed"
        if self.states[worker] == STRAGGLER and self.normal_streak[worker] >= self.recovery_steps:
            self.states[worker] = ALIVE
            return "cleared"
        return None

    def straggler_ranks(self):
        return [worker for worker, state in enumerate(self.states) if state == STRAGGLER]


def median_or_none(values):
    return statistics.median(values) if values else None
