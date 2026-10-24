"""Baseline 2: a step-time threshold on each worker, using nothing but its own step time.

This is what a simple "is my training slow?" alarm does: learn the normal step time during a
calibration window, then raise an alarm when the step time stays above `factor` times it.

In synchronous training every worker finishes every step at the same moment, so the step time
rises on ALL workers when ONE is slow. Each worker can only say "I am slow" (it knows nobody
else), so every worker accuses itself and the alarm cannot name the culprit. That is the point
this baseline is here to show.
"""

from swim_straggler.detector.base import Detector, StreakJudge, median_or_none, smooth, trace_event
from swim_straggler.detector.table import ALIVE

CALIBRATION_STEPS = 20  # steps after the warmup that define "normal"


class StepTimeDetector(Detector):
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.config = config
        self.smoothed_step_seconds = None
        self.calibration_values = []
        self.normal_step_seconds = None
        self.judge = StreakJudge(world_size, config.consecutive_steps, config.recovery_steps)
        self.trace = getattr(config, "trace", False)

    def after_update(self, step, step_seconds):
        self.smoothed_step_seconds = smooth(self.smoothed_step_seconds, step_seconds, self.config.ewma_alpha)
        if step < self.config.warmup_steps:
            return []
        if step < self.config.warmup_steps + CALIBRATION_STEPS:
            self.calibration_values.append(step_seconds)
            return []
        events = []
        if self.normal_step_seconds is None:
            self.normal_step_seconds = median_or_none(self.calibration_values)
            if self.trace:
                events.append(trace_event("learned_normal_step", self.rank, step,
                                          normal_step_ms=round(self.normal_step_seconds * 1000, 2)))

        looks_slow = self.smoothed_step_seconds > self.config.factor * self.normal_step_seconds
        streak_before = self.judge.slow_streak[self.rank]
        was_confirmed = self.judge.states[self.rank] != ALIVE
        decision = self.judge.judge(self.rank, looks_slow, step)
        if self.trace and not was_confirmed:
            if looks_slow:
                events.append(trace_event("own_step_slow", self.rank, step,
                                          step_ms=round(self.smoothed_step_seconds * 1000, 2),
                                          normal_step_ms=round(self.normal_step_seconds * 1000, 2), factor=self.config.factor,
                                          streak=self.judge.slow_streak[self.rank], needed=self.config.consecutive_steps))
            elif streak_before > 0:
                events.append(trace_event("own_step_normal_again", self.rank, step, after_steps=streak_before))
        if decision:
            events.append({"action": decision, "subject": self.rank, "step": step, "decided_by": "own step time"})
        return events

    def straggler_ranks(self):
        return self.judge.straggler_ranks()
