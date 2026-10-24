import statistics
import time

import pytest

from swim_straggler.config import FaultConfig
from swim_straggler.faults.inject import FaultInjector, SlowLinkTransport
from swim_straggler.telemetry.events import read_events
from swim_straggler.train.launcher import run_training
from test_training import make_config


def active_steps(fault_config, rank, total_steps=60):
    injector = FaultInjector(fault_config, rank)
    return [step for step in range(total_steps) if injector.is_fault_active(step)]


def test_compute_fault_applies_only_to_listed_workers_from_the_start_step():
    fault = FaultConfig(kind="compute", workers=[2], start_step=10)
    assert active_steps(fault, rank=2) == list(range(10, 60))
    assert active_steps(fault, rank=1) == []


def test_duration_ends_the_fault():
    fault = FaultConfig(kind="compute", workers=[0], start_step=5, duration=3)
    assert active_steps(fault, rank=0) == [5, 6, 7]


def test_spike_is_short_and_intermittent_alternates():
    spike = FaultConfig(kind="spike", workers=[0], start_step=10, duration=2)
    assert active_steps(spike, rank=0) == [10, 11]
    intermittent = FaultConfig(kind="intermittent", workers=[0], start_step=10, period=3)
    assert active_steps(intermittent, rank=0, total_steps=26) == [10, 11, 12, 16, 17, 18, 22, 23, 24]


def test_two_stragglers_both_slow_down():
    fault = FaultConfig(kind="two", workers=[1, 3], start_step=0)
    assert active_steps(fault, rank=1, total_steps=3) == [0, 1, 2]
    assert active_steps(fault, rank=3, total_steps=3) == [0, 1, 2]
    assert active_steps(fault, rank=0, total_steps=3) == []


def test_extra_compute_time_scales_with_the_multiplier():
    injector = FaultInjector(FaultConfig(kind="compute", workers=[0], multiplier=3.0, start_step=0), rank=0)
    assert injector.extra_compute_seconds(step=0, measured_compute_seconds=0.010) == pytest.approx(0.020)
    healthy_injector = FaultInjector(FaultConfig(kind="compute", workers=[1], start_step=0), rank=0)
    assert healthy_injector.extra_compute_seconds(step=0, measured_compute_seconds=0.010) == 0.0


@pytest.mark.parametrize("style", ["sleep", "busy"])
def test_slow_down_really_waits(style):
    fault = FaultConfig(kind="compute", workers=[0], multiplier=2.0, start_step=0, slowdown_style=style)
    started_at = time.perf_counter()
    added_seconds = FaultInjector(fault, rank=0).slow_down_compute(step=0, measured_compute_seconds=0.05)
    elapsed_seconds = time.perf_counter() - started_at
    assert added_seconds == pytest.approx(0.05)
    assert 0.045 <= elapsed_seconds < 0.5


def test_link_fault_delays_sends_but_not_compute():
    fault = FaultConfig(kind="link", workers=[0], start_step=0, link_delay_seconds=0.05)
    injector = FaultInjector(fault, rank=0)
    assert injector.extra_compute_seconds(step=0, measured_compute_seconds=1.0) == 0.0
    assert injector.send_delay_seconds(step=0) == 0.05

    class RecordingTransport:
        rank, world_size = 0, 2

        def __init__(self):
            self.sent = []

        def send_to_next(self, payload):
            self.sent.append(payload)

    inner_transport = RecordingTransport()
    slow_link = SlowLinkTransport(inner_transport, injector)
    started_at = time.perf_counter()
    slow_link.send_to_next("message")
    assert time.perf_counter() - started_at >= 0.045
    assert inner_transport.sent == ["message"]


def average(events, rank, key, first_step, last_step):
    values = [e[key] for e in events if e["event"] == "step" and e["rank"] == rank and first_step <= e["step"] < last_step]
    return statistics.mean(values)


def test_a_compute_fault_shows_up_as_the_slow_worker_not_waiting(tmp_path):
    """End to end: the faulty worker computes longer and waits least; step times stay equal."""
    fault = FaultConfig(kind="compute", workers=[2], multiplier=5.0, start_step=15)
    result = run_training(make_config(tmp_path, steps=45, fault=fault))
    events = read_events(result.events_path)

    assert [(e["rank"], e["step"]) for e in events if e["event"] == "fault_started"] == [(2, 15)]
    during = (20, 45)
    slow_wait = average(events, 2, "wait_seconds", *during)
    other_waits = [average(events, rank, "wait_seconds", *during) for rank in (0, 1, 3)]
    assert slow_wait < 0.5 * min(other_waits), (slow_wait, other_waits)
    assert average(events, 2, "compute_seconds", *during) > 2 * average(events, 0, "compute_seconds", *during)
    step_times = [average(events, rank, "step_seconds", *during) for rank in range(4)]
    assert max(step_times) < 1.2 * min(step_times)  # step time alone cannot name the culprit
