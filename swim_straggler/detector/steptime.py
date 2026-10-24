"""Baseline 2: a step-time threshold on each worker, using nothing but its own step time.

This is what a simple "is my training slow?" alarm does: learn the normal step time during a
calibration window, then raise an alarm when the step time stays above `factor` times it.

In synchronous training every worker finishes every step at the same moment, so the step time
rises on ALL workers when ONE is slow. Each worker can only say "I am slow" (it knows nobody
else), so every worker accuses itself and the alarm cannot name the culprit. That is the point
this baseline is here to show.
"""

from swim_straggler.detector.base import Detector, StreakJudge, median_or_none, smooth

CALIBRATION_STEPS = 20  # steps after the warmup that define "normal"


class StepTimeDetector(Detector):
    def __init__(self, rank, world_size, config):
        self.rank = rank
        self.config = config
        self.smoothed_step_seconds = None
        self.calibration_values = []
        self.normal_step_seconds = None
        self.judge = StreakJudge(world_size, config.consecutive_steps, config.recovery_steps)

    def after_update(self, step, step_seconds):
        self.smoothed_step_seconds = smooth(self.smoothed_step_seconds, step_seconds, self.config.ewma_alpha)
        if step < self.config.warmup_steps:
            return []
        if step < self.config.warmup_steps + CALIBRATION_STEPS:
            self.calibration_values.append(step_seconds)
            return []
        if self.normal_step_seconds is None:
            self.normal_step_seconds = median_or_none(self.calibration_values)

        looks_slow = self.smoothed_step_seconds > self.config.factor * self.normal_step_seconds
        decision = self.judge.judge(self.rank, looks_slow, step)
        if decision:
            return [{"action": decision, "subject": self.rank, "step": step, "decided_by": "own step time"}]
        return []

    def straggler_ranks(self):
        return self.judge.straggler_ranks()
