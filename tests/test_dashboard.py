"""The dashboard: its data functions on made-up events, then the real app, headless, on a real run."""

import json
import os
from pathlib import Path

import pytest

from swim_straggler.config import DetectorConfig, FaultConfig
from swim_straggler.dashboard.data import (
    belief_rows, chart_markers, detector_event_rows, list_runs, load_run_events, round_wait_rows, run_overview,
    step_rows,
)
from swim_straggler.train.launcher import run_training
from test_training import make_config

APP_PATH = Path(__file__).resolve().parents[1] / "swim_straggler" / "dashboard" / "app.py"


def detector_event(rank, step, action, subject):
    return {"event": "detector", "rank": rank, "step": step, "action": action, "subject": subject}


def test_beliefs_follow_the_detector_events():
    events = [
        detector_event(0, 10, "suspected", 2),
        detector_event(0, 12, "confirmed", 2),
        detector_event(1, 13, "learned_straggler", 2),
        detector_event(0, 30, "cleared", 2),
    ]
    def belief(step, believer, subject):
        rows = belief_rows(events, step, 3)
        return next(row["belief"] for row in rows if row["worker (believer)"] == believer and row["about worker"] == subject)

    assert belief(9, 0, 2) == "alive"
    assert belief(10, 0, 2) == "suspect"
    assert belief(12, 0, 2) == "straggler" and belief(12, 1, 2) == "alive"
    assert belief(13, 1, 2) == "straggler"
    assert belief(30, 0, 2) == "alive" and belief(30, 1, 2) == "straggler"  # worker 1 never heard of the clearing


def test_round_waits_and_markers():
    events = [
        {"event": "step", "rank": 0, "step": 4, "wait_seconds_per_round": [0.001, 0.010], "compute_seconds": 0.01,
         "wait_seconds": 0.011, "step_seconds": 0.03, "loss": 1.0},
        {"event": "fault_started", "rank": 1, "step": 3},
        detector_event(0, 9, "confirmed", 1),
        detector_event(2, 11, "confirmed", 1),
    ]
    assert round_wait_rows(events, 4) == [{"worker": 0, "round": 0, "wait (ms)": 1.0}, {"worker": 0, "round": 1, "wait (ms)": 10.0}]
    assert {(marker["step"], marker["label"]) for marker in chart_markers(events)} == {
        (3, "fault on worker 1"), (9, "worker 1 confirmed slow")}  # only the first confirmation is marked
    assert step_rows(events)[0]["compute (ms)"] == pytest.approx(10.0)


def test_a_run_that_is_still_going_is_read_from_the_worker_files(tmp_path):
    run_directory = tmp_path / "live_run"
    run_directory.mkdir()
    with open(run_directory / "events_rank_0.jsonl", "w", encoding="utf-8") as events_file:
        events_file.write(json.dumps({"event": "step", "rank": 0, "step": 0}) + "\n")
        events_file.write('{"event": "step", "rank": 0, "st')  # being written right now
    with open(run_directory / "events_rank_1.jsonl", "w", encoding="utf-8") as events_file:
        events_file.write(json.dumps({"event": "step", "rank": 1, "step": 0}) + "\n")
    events, is_finished = load_run_events(run_directory)
    assert not is_finished and [event["rank"] for event in events] == [0, 1]
    assert list_runs(tmp_path) == [("live_run", run_directory)]


def test_a_real_run_gives_every_table(tmp_path):
    config = make_config(tmp_path, steps=60, fault=FaultConfig(kind="compute", workers=[2], multiplier=4.0, start_step=20))
    config.detector = DetectorConfig(mode="swim", k_probes=2, warmup_steps=10, factor=1.5, consecutive_steps=8)
    result = run_training(config)
    events, is_finished = load_run_events(result.run_directory)
    assert is_finished
    assert run_overview(events)["fault"] == "compute" and run_overview(events)["workers"] == 4
    assert len(step_rows(events)) == 4 * 60
    assert len(round_wait_rows(events, 30)) == 4 * 2 * 3
    assert any(row["action"] == "confirmed" and row["about worker"] == 2 for row in detector_event_rows(events))
    final_beliefs = belief_rows(events, 59, 4)
    assert all(row["belief"] == "straggler" for row in final_beliefs if row["about worker"] == 2)


def test_the_app_draws_a_real_run_without_errors(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    config = make_config(tmp_path, steps=40, fault=FaultConfig(kind="compute", workers=[1], multiplier=4.0, start_step=15))
    config.detector = DetectorConfig(mode="swim", k_probes=2, warmup_steps=5)
    run_training(config)

    monkeypatch.setenv("SWIM_RUNS_DIRECTORY", config.run_directory)
    app = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    headers = [header.value for header in app.header]
    assert headers == ["What the detectors found", "The signal", "One step up close", "Detector events"]


def test_the_app_explains_an_empty_runs_folder(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("SWIM_RUNS_DIRECTORY", str(tmp_path / "nothing_here"))
    app = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception
    assert "No runs found" in app.info[0].value


def run_with_detector(tmp_path, mode):
    config = make_config(tmp_path, steps=40, fault=FaultConfig(kind="compute", workers=[1], multiplier=4.0, start_step=15))
    config.name = f"cmp_{mode}"
    config.detector = DetectorConfig(mode=mode, k_probes=2, warmup_steps=5)
    run_training(config)


def test_the_comparison_is_hidden_until_the_button_is_clicked(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    for mode in ("swim", "central", "steptime"):
        run_with_detector(tmp_path, mode)
    monkeypatch.setenv("SWIM_RUNS_DIRECTORY", str(tmp_path / "runs"))

    app = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert "Comparison with baseline detectors" not in [header.value for header in app.header]
    assert [button.label for button in app.button] == ["Compare with baseline detectors"]

    app.button[0].click().run()
    assert not app.exception, app.exception
    assert "Comparison with baseline detectors" in [header.value for header in app.header]
    table = app.dataframe[-1].value
    assert list(table["mode"]) == ["swim", "central", "steptime"]
    assert [button.label for button in app.button] == ["Hide comparison with baseline detectors"]

    app.button[0].click().run()  # hides it again
    assert "Comparison with baseline detectors" not in [header.value for header in app.header]


def test_the_comparison_says_what_is_missing(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    run_with_detector(tmp_path, "swim")
    monkeypatch.setenv("SWIM_RUNS_DIRECTORY", str(tmp_path / "runs"))
    app = testing.AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run()
    app.button[0].click().run()
    assert not app.exception, app.exception
    assert any("central, steptime" in info.value and "demo --compare" in info.value for info in app.info)
