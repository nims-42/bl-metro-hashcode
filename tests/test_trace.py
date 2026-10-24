"""The step-by-step trace: the detectors' trace events and trace.log built from them."""

import random
import re

import pytest

from swim_straggler.cli import main
from swim_straggler.config import DetectorConfig, FaultConfig
from swim_straggler.detector.central import CentralMonitor
from swim_straggler.detector.steptime import StepTimeDetector
from swim_straggler.telemetry.events import read_events
from swim_straggler.telemetry.trace import describe, parse_step_range, trace_lines, trace_text
from swim_straggler.train.launcher import run_training
from swim_straggler.train.worker import log_detector_events
from synthetic_ring import SyntheticRing
from test_training import make_config

WORLD_SIZE = 6
WORKER_LINE = re.compile(r"  w\d+: ")  # a line saying what one worker did
SLOW_WORKER = 3


def compute_seconds(step, rank, noise=random.Random(1)):
    seconds = 0.020 * (1 + noise.uniform(-0.05, 0.05))
    return seconds * 2.0 if rank == SLOW_WORKER and step >= 30 else seconds


def ring_events(trace):
    """Run a synthetic ring and return its events in the format of the event log."""
    ring = SyntheticRing(WORLD_SIZE, DetectorConfig(k_probes=3, trace=trace))
    noise = random.Random(1)
    ring.run(80, lambda step, rank: compute_seconds(step, rank, noise))
    events = [{"event": "run_start", "rank": 0, "world_size": WORLD_SIZE, "config": {
        "name": "synthetic", "cluster": {"backend": "synthetic"},
        "fault": {"kind": "compute", "workers": [SLOW_WORKER], "start_step": 30, "multiplier": 2.0},
        "detector": DetectorConfig(k_probes=3, trace=trace).__dict__}}]
    for rank, event in ring.all_events:
        event = dict(event)
        events.append({"event": "trace" if event.pop("trace", False) else "detector", "rank": rank, **event})
    return events


def test_tracing_does_not_change_any_decision():
    decisions_with_trace = [event for event in ring_events(trace=True) if event["event"] == "detector"]
    decisions_without = [event for event in ring_events(trace=False) if event["event"] == "detector"]
    assert decisions_with_trace == decisions_without
    assert not [event for event in ring_events(trace=False) if event["event"] == "trace"]


def test_the_trace_explains_how_the_culprit_was_found():
    events = ring_events(trace=True)
    actions = {event["action"] for event in events if event["event"] == "trace"}
    assert {"points_to", "answers", "got_answer", "distrusts_own_waits"} <= actions

    suspected = next(event for event in events if event.get("action") == "suspected")
    confirmed = next(event for event in events if event.get("action") == "confirmed")
    assert suspected["subject"] == confirmed["subject"] == SLOW_WORKER
    # before suspecting, the worker counted the culprit for consecutive_steps steps in a row
    counted = [event for event in events if event.get("action") == "points_to" and event["rank"] == suspected["rank"]
               and event["subject"] == SLOW_WORKER and event["step"] <= suspected["step"]]
    assert counted[-1]["streak"] == counted[-1]["needed"] == DetectorConfig().consecutive_steps
    # every vote the confirming worker counted shows up as an answer it heard
    heard = [event for event in events if event.get("action") == "got_answer" and event["rank"] == confirmed["rank"]
             and event["step"] <= confirmed["step"]]
    assert len([event for event in heard if event["says_slow"]]) >= confirmed["votes_for"]
    # the slow worker itself refuses to accuse, and learned workers say who told them
    assert any(event.get("action") == "distrusts_own_waits" and event["rank"] == SLOW_WORKER for event in events)
    learned = [event for event in events if event.get("action") == "learned_straggler"]
    assert learned and all(event["via"] == (event["rank"] - 1) % WORLD_SIZE for event in learned)  # gossip comes from the predecessor


def test_trace_log_is_grouped_by_step_and_can_be_filtered():
    events = ring_events(trace=True)
    text = trace_text(events)
    assert "detail: full" in text
    assert f"SUSPECTS w{SLOW_WORKER}" in text and f"CONFIRMS w{SLOW_WORKER} is a straggler" in text
    assert "beliefs now:" in text and text.rstrip().splitlines()[0].startswith("trace of 'synthetic'")

    one_worker = trace_lines(events, worker=1)
    detail_lines = [line for line in one_worker if WORKER_LINE.match(line)]
    assert detail_lines and all(line.startswith("  w1:") for line in detail_lines)

    about_culprit = trace_lines(events, about=SLOW_WORKER, first_step=30, last_step=60)
    step_numbers = [int(line.split()[1]) for line in about_culprit if line.startswith("step ")]
    assert step_numbers and 30 <= min(step_numbers) and max(step_numbers) <= 60
    assert all(f"w{SLOW_WORKER}" in line for line in about_culprit if WORKER_LINE.match(line))


def test_every_trace_event_has_a_sentence():
    for event in ring_events(trace=True):
        if event["event"] in ("detector", "trace"):
            sentence = describe(event, recovery_steps=20)
            assert sentence != f"{event['action']} w{event['subject']}", event  # not the fallback for unknown actions


def test_the_baselines_trace_their_counting():
    config = DetectorConfig(mode="central", trace=True, warmup_steps=2, consecutive_steps=3)
    monitor = CentralMonitor(0, 4, config)
    events = []
    for step in range(10):
        events += monitor.finish_step(step, [], [0.01, 0.01, 0.03 if step >= 4 else 0.01, 0.01])
    counting = [event for event in events if event.get("action") == "looks_slow"]
    assert [event["streak"] for event in counting] == [1, 2, 3] and all(event["trace"] for event in counting)
    assert [event["action"] for event in events if not event.get("trace")] == ["confirmed"]

    own = StepTimeDetector(1, 4, DetectorConfig(mode="steptime", trace=True, warmup_steps=0, consecutive_steps=2,
                                                 ewma_alpha=1.0))
    events = []
    for step in range(30):
        events += own.after_update(step, 0.05 if step >= 25 else 0.03)
    assert [event["action"] for event in events] == ["learned_normal_step", "own_step_slow", "own_step_slow", "confirmed"]


def test_decisions_and_reasoning_are_logged_as_different_events():
    logged = []

    class Logger:
        def log(self, name, **fields):
            logged.append((name, fields))

    log_detector_events(Logger(), [{"action": "confirmed", "subject": 1, "step": 5},
                                   {"trace": True, "action": "points_to", "subject": 1, "step": 5}])
    assert [name for name, _ in logged] == ["detector", "trace"]
    assert "trace" not in logged[1][1]


def test_container_faults_appear_when_issued_and_when_in_place():
    events = [
        {"event": "run_start", "rank": 0, "world_size": 2, "config": {
            "name": "docker", "cluster": {"backend": "docker"},
            "fault": {"kind": "container_cpu", "workers": [1], "start_step": 10, "multiplier": 3.0},
            "detector": DetectorConfig(mode="off").__dict__}},
        {"event": "step", "rank": 0, "step": 10, "loss": 1.0, "compute_seconds": 0.01, "wait_seconds": 0.0,
         "step_seconds": 0.02, "fault_active": False},
        {"event": "fault_started", "rank": 1, "source": "controller", "step": 10, "step_after_command": 14,
         "fault_kind": "container_cpu", "tool": "docker"},
        {"event": "fault_ended", "rank": 1, "source": "controller", "step": 30, "fault_kind": "container_cpu"},
    ]
    text = trace_text(events)
    assert "FAULT w1: container processor share cut to 1/3.0: docker command issued" in text
    assert "step 14\n  FAULT w1: fault command returned, the fault is in place" in text
    assert "FAULT w1: fault removed" in text


def test_step_ranges():
    assert parse_step_range("60-90") == (60, 90)
    assert parse_step_range("60-") == (60, None)
    assert parse_step_range("75") == (75, 75)
    assert parse_step_range(None) == (None, None)
    with pytest.raises(ValueError):
        parse_step_range("sixty")


def test_a_real_run_writes_trace_log_and_the_trace_command_filters_it(tmp_path, capsys):
    config = make_config(tmp_path, steps=40, fault=FaultConfig(kind="compute", workers=[1], multiplier=4.0, start_step=15))
    config.detector = DetectorConfig(mode="swim", k_probes=2, warmup_steps=5, trace=True)
    result = run_training(config)
    trace_log = (result.run_directory / "trace.log").read_text()
    assert "detail: full" in trace_log and "FAULT w1: compute slowed down x4.0 (active from this step)" in trace_log
    assert "step 39" in trace_log and "summary" in trace_log
    assert any(event["event"] == "trace" for event in read_events(result.events_path))

    capsys.readouterr()
    assert main(["trace", str(result.run_directory), "--worker", "0", "--steps", "15-25"]) == 0
    printed = capsys.readouterr().out
    detail_lines = [line for line in printed.splitlines() if WORKER_LINE.match(line)]
    assert detail_lines and all(line.startswith("  w0:") for line in detail_lines)
    assert main(["trace", str(tmp_path / "missing")]) == 1


def test_large_groups_get_short_lines():
    from swim_straggler.telemetry.trace import beliefs_text, compact_measurement_lines, ranges_text

    assert ranges_text([0, 1, 2, 5, 7, 8]) == "w0-w2, w5, w7-w8"
    beliefs = {rank: {9: "straggler"} for rank in range(20)}
    beliefs[4] = {9: "suspect"}
    assert beliefs_text(beliefs) == "w0-w3, w5-w19 think w9 straggler; w4 thinks w9 suspect"
    per_worker = {rank: {"compute_seconds": 0.012 if rank != 9 else 0.030, "wait_seconds": 0.010 if rank != 9 else 0.001}
                  for rank in range(20)}
    assert compact_measurement_lines(per_worker) == ["  compute ms : median 12.0, slowest w9 30.0",
                                                     "  wait ms    : median 10.0, least w9 1.0"]
