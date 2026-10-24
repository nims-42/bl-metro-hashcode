"""Sweeps, metrics, summaries, plots and results.md. Training is replaced by a stand-in where possible."""

import json
from pathlib import Path

import pytest
import yaml

from swim_straggler.config import ConfigError
from swim_straggler.evaluation.metrics import run_metrics, step_time_overhead, summarize_group, summarize_sweep
from swim_straggler.evaluation.report import write_results
from swim_straggler.evaluation.sweep import load_sweep_spec, plan_runs, run_planned

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_spec(tmp_path, **changes):
    spec = {
        "name": "tiny",
        "base_config": str(PROJECT_ROOT / "experiments" / "configs" / "smoke.yaml"),
        "overrides": {"cluster": {"world_size": 4, "backend": "pipes"}, "train": {"steps": 30}},
        "repeats": 2,
        "faults": [{"label": "healthy", "kind": "none"},
                   {"label": "x4", "kind": "compute", "workers": [2], "multiplier": 4.0, "start_step": 10}],
        "detectors": [{"label": "off", "mode": "off", "only_faults": ["healthy"]},
                      {"label": "swim", "mode": "swim", "warmup_steps": 3}],
        **changes,
    }
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


# ---- planning --------------------------------------------------------------------------------

def test_plan_expands_faults_detectors_and_repeats(tmp_path):
    planned = plan_runs(load_sweep_spec(write_spec(tmp_path)))
    assert sorted(run.key for run in planned) == sorted([
        ("off", "healthy", 0), ("off", "healthy", 1),  # `off` runs only on the healthy fault
        ("swim", "healthy", 0), ("swim", "healthy", 1), ("swim", "x4", 0), ("swim", "x4", 1)])


def test_every_detector_sees_the_same_seeds(tmp_path):
    planned = plan_runs(load_sweep_spec(write_spec(tmp_path)))
    seeds = {(run.detector, run.repeat): run.config.train.seed for run in planned if run.fault == "healthy"}
    assert seeds[("off", 0)] == seeds[("swim", 0)] and seeds[("off", 1)] == seeds[("swim", 1)]
    assert seeds[("swim", 0)] != seeds[("swim", 1)]


def test_a_fault_does_not_inherit_the_base_configs_fault(tmp_path):
    planned = plan_runs(load_sweep_spec(write_spec(tmp_path)))
    healthy = next(run for run in planned if run.fault == "healthy")
    assert healthy.config.fault.kind == "none" and healthy.config.fault.workers == []


def test_unquoted_off_in_a_sweep_file_still_means_off(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(write_spec(tmp_path).read_text().replace("'off'", "off"))
    planned = plan_runs(load_sweep_spec(path))
    assert {run.detector for run in planned} == {"off", "swim"}
    assert {run.config.detector.mode for run in planned if run.detector == "off"} == {"off"}


def test_misspelt_fields_are_rejected(tmp_path):
    spec = load_sweep_spec(write_spec(tmp_path, faults=[{"label": "bad", "kind": "compute", "wrokers": [1]}]))
    with pytest.raises(ConfigError, match="wrokers"):
        plan_runs(spec)


def test_shipped_sweeps_are_valid():
    for name in ("main", "docker"):
        path = PROJECT_ROOT / "experiments" / "sweeps" / f"{name}.yaml"
        assert len(plan_runs(load_sweep_spec(path), path.parent)) > 0


# ---- metrics and summaries -------------------------------------------------------------------

def row(detector, fault, found, delay, false_confirmations=0, step_ms=30.0, faulty=1):
    return {"detector": detector, "fault": fault, "culprit_found": found if faulty else None, "faulty_workers": faulty,
            "delay_steps": delay, "delay_after_applied_steps": delay, "everyone_knew_delay_steps": delay,
            "compute_ratio": 2.0 if faulty else None, "false_confirmations": false_confirmations, "false_suspicions": 0,
            "mean_step_ms": step_ms}


def test_group_summary_counts_only_found_runs_for_the_delay():
    summary = summarize_group([row("swim", "x2", True, 6), row("swim", "x2", True, 10), row("swim", "x2", False, None, 1)])
    assert (summary["found"], summary["runs"]) == (2, 3)
    assert summary["median_delay_steps"] == 8 and (summary["min_delay_steps"], summary["max_delay_steps"]) == (6, 10)
    assert summary["runs_with_false_confirmation"] == 1


def test_healthy_groups_have_no_found_count_and_overhead_is_against_off():
    rows = [row("off", "healthy", None, None, step_ms=30.0, faulty=0), row("swim", "healthy", None, None, step_ms=33.0, faulty=0)]
    summaries = summarize_sweep(rows)
    assert summaries[0]["found"] is None
    overhead = {item["detector"]: item["overhead_percent"] for item in step_time_overhead(summaries)}
    assert overhead["off"] == pytest.approx(0) and overhead["swim"] == pytest.approx(10)


# ---- a whole small sweep, with a stand-in for training -----------------------------------------

def stand_in_training(config):
    """Writes a believable event log instead of training: slow worker 2 is confirmed 5 steps after its fault."""
    from swim_straggler.train.launcher import TrainingResult

    run_directory = Path(config.run_directory) / config.name
    run_directory.mkdir(parents=True, exist_ok=True)
    events = [{"event": "run_start", "rank": 0, "config": config.to_dict(), "dataset_source": "synthetic", "world_size": 4}]
    is_faulty = config.fault.kind != "none"
    for step in range(config.train.steps):
        for rank in range(4):
            slow = is_faulty and rank in config.fault.workers and step >= config.fault.start_step
            events.append({"event": "step", "rank": rank, "step": step, "loss": 1.0, "compute_seconds": 0.02 if slow else 0.01,
                           "wait_seconds": 0.001, "step_seconds": 0.03, "wait_seconds_per_round": [0.0] * 6})
    if is_faulty:
        events.append({"event": "fault_started", "rank": 2, "step": config.fault.start_step, "fault_kind": "compute"})
        if config.detector.mode != "off":
            for rank in range(4):
                events.append({"event": "detector", "rank": rank, "step": config.fault.start_step + 5,
                               "action": "confirmed" if rank != 2 else "learned_straggler", "subject": 2})
    path = run_directory / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return TrainingResult(run_directory=run_directory, events_path=path, dataset_source="synthetic")


def test_a_sweep_can_be_stopped_and_continued(tmp_path):
    planned = plan_runs(load_sweep_spec(write_spec(tmp_path)))
    for run in planned:
        run.config.run_directory = str(tmp_path / "runs")
    results_path = tmp_path / "runs.jsonl"
    started = []
    run_planned(planned[:2], results_path, train=lambda config: started.append(config.name) or stand_in_training(config),
                progress=lambda text: None)
    rows = run_planned(planned, results_path, train=lambda config: started.append(config.name) or stand_in_training(config),
                       progress=lambda text: None)
    assert len(rows) == len(planned) and len(started) == len(planned)  # nothing was run twice
    faulty_rows = [item for item in rows if item["fault"] == "x4"]
    assert all(item["culprit_found"] and item["delay_steps"] == 5 for item in faulty_rows)


def test_results_and_plots(tmp_path):
    results_directory = tmp_path / "results" / "tiny"
    results_directory.mkdir(parents=True)
    planned = plan_runs(load_sweep_spec(write_spec(tmp_path)))
    for run in planned:
        run.config.run_directory = str(tmp_path / "runs")
    run_planned(planned, results_directory / "runs.jsonl", train=stand_in_training, progress=lambda text: None)
    (results_directory / "machine.json").write_text(json.dumps(
        {"processors": 2, "platform": "test", "python": "3", "torch": "x", "date": "2026-10-25"}))

    results_path = write_results(results_directory)
    text = results_path.read_text()
    assert "| swim | x4 | 2 of 2 | 5.0 (5–5) |" in text
    for image in ("detection_delay.png", "false_confirmations.png", "step_time_overhead.png"):
        assert (results_directory / image).stat().st_size > 1000


def test_run_metrics_on_a_real_short_run(tmp_path):
    from swim_straggler.config import DetectorConfig, FaultConfig
    from swim_straggler.telemetry.events import read_events
    from swim_straggler.train.launcher import run_training
    from test_training import make_config

    config = make_config(tmp_path, steps=60, fault=FaultConfig(kind="compute", workers=[2], multiplier=4.0, start_step=20))
    config.detector = DetectorConfig(mode="swim", k_probes=2, warmup_steps=10, factor=1.5, consecutive_steps=8)
    metrics = run_metrics(read_events(run_training(config).events_path))
    assert metrics["faulty_workers"] == 1 and metrics["culprit_found"] is True
    assert metrics["compute_ratio"] > 2 and metrics["steps"] == 60
