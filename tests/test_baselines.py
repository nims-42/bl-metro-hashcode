"""The two baselines: unit tests on made-up numbers, then each one inside real (pipes) training."""

import pytest

from swim_straggler.config import DetectorConfig, FaultConfig
from swim_straggler.detector.central import CentralMonitor
from swim_straggler.detector.steptime import CALIBRATION_STEPS, StepTimeDetector
from swim_straggler.telemetry.events import read_events, summarize_detection
from swim_straggler.train.launcher import run_training
from test_training import make_config

CONFIG = DetectorConfig(mode="central", warmup_steps=5, consecutive_steps=3, recovery_steps=4, ewma_alpha=1.0)


def feed_central(monitor, compute_by_step):
    events = []
    for step, compute_of_every_worker in enumerate(compute_by_step):
        events += monitor.finish_step(step, [], compute_of_every_worker)
    return events


# ---- central monitor ------------------------------------------------------------------------

def test_central_monitor_confirms_the_slow_worker_after_consecutive_steps():
    monitor = CentralMonitor(0, 4, CONFIG)
    trace = [[0.01, 0.01, 0.01, 0.01]] * 10 + [[0.01, 0.01, 0.03, 0.01]] * 5
    events = feed_central(monitor, trace)
    assert [(event["action"], event["subject"], event["step"]) for event in events] == [("confirmed", 2, 12)]
    assert monitor.straggler_ranks() == [2]


def test_central_monitor_clears_a_recovered_worker():
    monitor = CentralMonitor(0, 4, CONFIG)
    trace = [[0.01] * 4] * 10 + [[0.01, 0.01, 0.03, 0.01]] * 5 + [[0.01] * 4] * 5
    actions = [(event["action"], event["subject"]) for event in feed_central(monitor, trace)]
    assert actions == [("confirmed", 2), ("cleared", 2)]


def test_central_monitor_ignores_noise_and_short_spikes():
    monitor = CentralMonitor(0, 4, CONFIG)
    trace = [[0.010, 0.011, 0.012, 0.009]] * 10 + [[0.01, 0.05, 0.01, 0.01]] * 2 + [[0.01] * 4] * 10
    assert feed_central(monitor, trace) == []


def test_only_worker_zero_decides():
    other = CentralMonitor(1, 4, CONFIG)
    assert feed_central(other, [[0.01, 0.01, 0.05, 0.01]] * 20) == []
    assert other.straggler_ranks() == []


def test_two_slow_workers_are_both_confirmed_because_confirmed_ones_leave_the_median():
    monitor = CentralMonitor(0, 5, CONFIG)
    trace = [[0.01] * 5] * 6 + [[0.01, 0.03, 0.01, 0.03, 0.01]] * 6
    feed_central(monitor, trace)
    assert monitor.straggler_ranks() == [1, 3]


# ---- step-time threshold --------------------------------------------------------------------

def feed_step_times(detector, step_seconds_by_step):
    events = []
    for step, seconds in enumerate(step_seconds_by_step):
        events += detector.after_update(step, seconds)
    return events


def test_step_time_detector_accuses_itself_when_the_step_time_rises():
    detector = StepTimeDetector(3, 4, CONFIG)
    normal = [0.03] * (CONFIG.warmup_steps + CALIBRATION_STEPS + 5)
    events = feed_step_times(detector, normal + [0.06] * 5)
    assert [(event["action"], event["subject"]) for event in events] == [("confirmed", 3)]  # always itself


def test_step_time_detector_stays_quiet_on_a_steady_run():
    detector = StepTimeDetector(1, 4, CONFIG)
    assert feed_step_times(detector, [0.030, 0.031, 0.029] * 30) == []


def test_step_time_detector_learns_normal_only_from_the_calibration_window():
    detector = StepTimeDetector(0, 4, CONFIG)
    feed_step_times(detector, [0.5] * CONFIG.warmup_steps + [0.03] * CALIBRATION_STEPS)  # slow start is ignored
    assert detector.normal_step_seconds is None  # decided at the first step after calibration
    detector.after_update(CONFIG.warmup_steps + CALIBRATION_STEPS, 0.03)
    assert detector.normal_step_seconds == pytest.approx(0.03)


# ---- inside real training -------------------------------------------------------------------

def run_mode(tmp_path, mode, fault=None, steps=70):
    config = make_config(tmp_path, steps=steps, fault=fault)
    config.detector = DetectorConfig(mode=mode, warmup_steps=10, factor=1.5, consecutive_steps=8, k_probes=2)
    return read_events(run_training(config).events_path)


SLOW_WORKER_TWO = FaultConfig(kind="compute", workers=[2], multiplier=4.0, start_step=25)


def test_central_mode_in_training_names_the_culprit_and_only_worker_zero_speaks(tmp_path):
    events = run_mode(tmp_path, "central", SLOW_WORKER_TWO)
    summary = summarize_detection(events)
    assert summary["per_fault"][2]["first_confirmed_step"] is not None, summary
    assert summary["false_confirmations"] == []
    assert {event["rank"] for event in events if event["event"] == "detector"} == {0}


def test_step_time_mode_in_training_cannot_name_the_culprit(tmp_path):
    """Every worker's step time rises, so every worker accuses itself: 3 of 4 accusations are wrong."""
    events = run_mode(tmp_path, "steptime", SLOW_WORKER_TWO, steps=90)
    confirmations = [(event["rank"], event["subject"]) for event in events
                     if event["event"] == "detector" and event["action"] == "confirmed"]
    assert sorted(confirmations) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert len(summarize_detection(events)["false_confirmations"]) == 3


def test_central_reports_do_not_change_what_is_learned(tmp_path):
    def checksums(events):
        return sorted(event["parameter_checksum"] for event in events if event["event"] == "run_end")

    with_reports = checksums(run_mode(tmp_path / "central", "central", steps=30))
    without = checksums(run_mode(tmp_path / "off", "off", steps=30))
    for one, other in zip(with_reports, without):
        assert one == pytest.approx(other, abs=1e-4)


def test_demo_compare_flag_prints_one_line_per_detector(tmp_path, capsys):
    import yaml

    from swim_straggler.cli import main

    config = make_config(tmp_path, steps=20)
    config.detector = DetectorConfig(mode="swim", k_probes=2)
    config_path = tmp_path / "compare.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    assert main(["demo", "--compare", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    for mode in ("swim", "central", "steptime"):
        assert any(line.startswith(mode) for line in output.splitlines())


def test_demo_without_the_flag_does_not_run_the_baselines(tmp_path, capsys):
    import yaml

    from swim_straggler.cli import main

    config = make_config(tmp_path, steps=15)
    config.detector = DetectorConfig(mode="swim", k_probes=2)
    config_path = tmp_path / "plain.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    assert main(["demo", "--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "central" not in output and "steptime" not in output
