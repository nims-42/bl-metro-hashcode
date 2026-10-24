"""The detector inside real training: pipes (threads) for speed, one gloo test for realism.

Timing-based, so margins are generous and the faults are large. The detector must never confirm
a healthy worker, and it must never change what the training learns.
"""

import pytest

from swim_straggler.config import ConfigError, DetectorConfig, FaultConfig
from swim_straggler.telemetry.events import read_events, summarize_detection
from swim_straggler.train.launcher import run_training
from test_training import make_config

# Less sensitive than the defaults on purpose: four workers share very few processors in the test machine,
# and the timing noise makes the default settings confirm a healthy worker now and then (measured in milestone 5).
DETECTOR = dict(mode="swim", k_probes=2, warmup_steps=10, factor=1.5, consecutive_steps=8)


def config_with_detector(tmp_path, detector, **keywords):
    config = make_config(tmp_path, **keywords)
    config.detector = detector
    return config


def checksums(events):
    return sorted(event["parameter_checksum"] for event in events if event["event"] == "run_end")


def test_a_slowed_worker_is_confirmed_and_known_to_all(tmp_path):
    fault = FaultConfig(kind="compute", workers=[2], multiplier=4.0, start_step=25)
    config = config_with_detector(tmp_path, DetectorConfig(**DETECTOR), steps=70, fault=fault)
    events = read_events(run_training(config).events_path)
    summary = summarize_detection(events)

    report = summary["per_fault"][2]
    assert report["first_confirmed_step"] is not None, summary
    assert 25 <= report["first_confirmed_step"] <= 25 + 30
    assert report["workers_that_never_learned"] == []
    assert summary["false_confirmations"] == []
    run_end = [event for event in events if event["event"] == "run_end"]
    assert all(event["believed_stragglers"] == [2] for event in run_end)


def test_a_healthy_run_confirms_nobody(tmp_path):
    config = config_with_detector(tmp_path, DetectorConfig(**DETECTOR), steps=80)
    events = read_events(run_training(config).events_path)
    assert summarize_detection(events)["false_confirmations"] == []
    assert all(event["believed_stragglers"] == [] for event in events if event["event"] == "run_end")


def test_detection_does_not_change_what_is_learned(tmp_path):
    """Same seed, same data: the weights must be identical with the detector on or off."""
    with_detector = read_events(run_training(config_with_detector(tmp_path / "on", DetectorConfig(**DETECTOR))).events_path)
    without_detector = read_events(run_training(config_with_detector(tmp_path / "off", DetectorConfig(mode="off"))).events_path)
    for on_value, off_value in zip(checksums(with_detector), checksums(without_detector)):
        assert on_value == pytest.approx(off_value, abs=1e-4)


def test_step_events_carry_the_wait_of_every_round(tmp_path):
    events = read_events(run_training(config_with_detector(tmp_path, DetectorConfig(**DETECTOR), steps=12)).events_path)
    step_event = next(event for event in events if event["event"] == "step")
    assert len(step_event["wait_seconds_per_round"]) == 2 * (4 - 1)
    assert step_event["wait_seconds"] == pytest.approx(sum(step_event["wait_seconds_per_round"]))


def test_detector_off_writes_no_detector_events(tmp_path):
    events = read_events(run_training(config_with_detector(tmp_path, DetectorConfig(mode="off"), steps=12)).events_path)
    assert [event for event in events if event["event"] == "detector"] == []


def test_baseline_modes_are_not_available_yet(tmp_path):
    with pytest.raises(ConfigError, match="milestone 6"):
        run_training(config_with_detector(tmp_path, DetectorConfig(mode="central"), steps=5))


@pytest.mark.slow
def test_gloo_processes_detect_a_slowed_worker(tmp_path):
    fault = FaultConfig(kind="compute", workers=[1], multiplier=4.0, start_step=25)
    config = config_with_detector(tmp_path, DetectorConfig(**DETECTOR), backend="gloo", world_size=4, steps=70, fault=fault)
    summary = summarize_detection(read_events(run_training(config).events_path))
    assert summary["per_fault"][1]["first_confirmed_step"] is not None, summary
    assert summary["per_fault"][1]["workers_that_never_learned"] == []
    assert summary["false_confirmations"] == []
