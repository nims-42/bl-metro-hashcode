"""Scenario tests: the whole detector on synthetic compute-time traces. No torch, no real timing."""

import random

from swim_straggler.config import DetectorConfig
from swim_straggler.detector.table import ALIVE, STRAGGLER
from synthetic_ring import SyntheticRing, waits_in_ring

NORMAL_COMPUTE_SECONDS = 0.020
WORLD_SIZE = 8


def make_trace(slow_ranks=(), multiplier=2.0, first_slow_step=50, last_slow_step=10**9, noise=0.05, seed=1,
               everyone_slow_from=None, period=None):
    """compute_seconds_function(step, rank) with small random noise and optional slow workers."""
    noise_generator = random.Random(seed)

    def compute_seconds(step, rank):
        seconds = NORMAL_COMPUTE_SECONDS * (1 + noise_generator.uniform(-noise, noise))
        is_slow_now = first_slow_step <= step <= last_slow_step
        if period is not None:
            is_slow_now = is_slow_now and ((step - first_slow_step) // period) % 2 == 0
        if is_slow_now and rank in slow_ranks:
            seconds *= multiplier
        if everyone_slow_from is not None and step >= everyone_slow_from:
            seconds *= multiplier
        return seconds

    return compute_seconds


def run(trace, steps, detector_config=None, world_size=WORLD_SIZE):
    ring = SyntheticRing(world_size, detector_config or DetectorConfig())
    ring.run(steps, trace)
    return ring


def test_the_wait_model_points_at_the_slow_worker():
    waits = waits_in_ring([0.02, 0.02, 0.06, 0.02])
    assert waits[2][0] < 0.001  # the slow worker itself does not wait
    assert waits[3][0] > 0.03  # its successor waits right away, in round 0
    assert waits[0][1] > 0.03  # the next one in round 1


def test_healthy_cluster_raises_no_alarm():
    ring = run(make_trace(), steps=300)
    assert ring.all_events == []


def test_uniform_slowdown_raises_no_alarm():
    """If every worker slows down together (for example a bigger batch), nobody is the straggler."""
    ring = run(make_trace(everyone_slow_from=50), steps=150)
    assert ring.all_events == []


def test_a_slow_worker_is_confirmed_quickly_and_everybody_learns_it():
    ring = run(make_trace(slow_ranks=[4], multiplier=2.0, first_slow_step=50), steps=100)
    confirmations = ring.events_with_action("confirmed")
    assert confirmations, "nobody confirmed the straggler"
    assert {event["subject"] for _, event in confirmations} == {4}
    assert 50 <= min(event["step"] for _, event in confirmations) <= 50 + 12  # a few steps after the fault starts
    for rank, detector in enumerate(ring.detectors):
        assert detector.straggler_ranks() == [4], f"worker {rank} believes {detector.straggler_ranks()}"
    assert [rank for rank, _ in ring.events_with_action("suspected") if rank == 4] == []  # the culprit accuses nobody


def test_no_healthy_worker_is_ever_flagged():
    ring = run(make_trace(slow_ranks=[4]), steps=100)
    for _, event in ring.all_events:
        assert event["subject"] == 4


def test_a_short_spike_is_ignored():
    ring = run(make_trace(slow_ranks=[2], multiplier=3.0, first_slow_step=40, last_slow_step=42), steps=120)
    assert ring.events_with_action("suspected") == []


def test_a_straggler_that_recovers_is_cleared_and_returns_to_alive():
    config = DetectorConfig(recovery_steps=15)
    ring = run(make_trace(slow_ranks=[3], first_slow_step=40, last_slow_step=79), steps=160, detector_config=config)
    assert ring.events_with_action("confirmed")
    cleared = ring.events_with_action("cleared")
    assert cleared and all(event["subject"] == 3 for _, event in cleared)
    assert min(event["step"] for _, event in cleared) >= 80 + 15 - 5  # not before it has really been normal for a while
    for detector in ring.detectors:
        assert detector.table.states == [ALIVE] * WORLD_SIZE


def test_an_intermittent_straggler_is_confirmed_and_does_not_flap():
    """It is slow for 10 steps, normal for 10 steps. The 20-step recovery period bridges the gaps."""
    ring = run(make_trace(slow_ranks=[5], first_slow_step=40, period=10), steps=200)
    assert {event["subject"] for _, event in ring.events_with_action("confirmed")} == {5}
    assert ring.events_with_action("cleared") == []
    assert all(detector.table.states[5] == STRAGGLER for detector in ring.detectors)


def test_two_stragglers_are_both_found():
    ring = run(make_trace(slow_ranks=[1, 5], multiplier=2.5), steps=120)
    for rank, detector in enumerate(ring.detectors):
        assert detector.straggler_ranks() == [1, 5], f"worker {rank} believes {detector.straggler_ranks()}"


def test_a_mild_slowdown_below_the_threshold_is_not_confirmed():
    """1.15x is below the factor of 1.3, so the peers' tables do not agree even if the waits notice it."""
    ring = run(make_trace(slow_ranks=[4], multiplier=1.15, noise=0.01), steps=150)
    assert ring.events_with_action("confirmed") == []


def test_larger_groups_work_too():
    ring = run(make_trace(slow_ranks=[17], multiplier=2.0), steps=100, world_size=32)
    for detector in ring.detectors:
        assert detector.straggler_ranks() == [17]
