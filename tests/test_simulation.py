"""The simulator: the real detectors on made-up timings."""

import json
import random

import pytest

from swim_straggler.cli import main
from swim_straggler.config import (
    ClusterConfig, Config, ConfigError, DetectorConfig, FaultConfig, SimulationConfig, TrainConfig,
)
from swim_straggler.detector.swim import StragglerDetector
from swim_straggler.evaluation.report import write_results
from swim_straggler.evaluation.scaling import detection_load
from swim_straggler.simulation.simulator import fit_timing_model
from swim_straggler.telemetry.events import read_events, summarize_detection
from swim_straggler.train.launcher import run_training


def simulated_config(tmp_path, workers=8, steps=120, fault=None, mode="swim", **simulation):
    return Config(
        name="sim", run_directory=str(tmp_path / "runs"),
        cluster=ClusterConfig(world_size=workers, backend="simulated"),
        train=TrainConfig(steps=steps, seed=3),
        fault=fault or FaultConfig(kind="compute", workers=[5], multiplier=3.0, start_step=40),
        detector=DetectorConfig(mode=mode, k_probes=min(3, workers - 2)),
        simulation=SimulationConfig(**simulation),
    )


def test_a_simulated_run_finds_the_slow_worker_and_logs_like_a_real_run(tmp_path):
    result = run_training(simulated_config(tmp_path))
    events = read_events(result.events_path)
    assert events[0]["event"] == "run_start" and events[0]["simulated"] is True
    assert len([event for event in events if event["event"] == "step"]) == 8 * 120
    report = summarize_detection(events)["per_fault"][5]
    assert report["first_confirmed_step"] is not None and report["delay_steps"] < 30
    assert report["workers_that_never_learned"] == []
    trace_log = (result.run_directory / "trace.log").read_text()
    assert trace_log.startswith("SIMULATION:")


def without_wall_clock(events):
    return [{key: value for key, value in event.items() if key != "simulator_wall_seconds"} for event in events]


def test_the_simulation_is_reproducible_from_its_seed(tmp_path):
    first = read_events(run_training(simulated_config(tmp_path / "a", workers=6, steps=80)).events_path)
    second = read_events(run_training(simulated_config(tmp_path / "b", workers=6, steps=80)).events_path)
    assert without_wall_clock(first[1:]) == without_wall_clock(second[1:])  # the first event names the run folder


def test_passing_gossip_as_objects_gives_exactly_the_decisions_of_passing_numbers():
    """The simulator skips encoding gossip into numbers; that must not change anything."""
    world_size, config = 6, DetectorConfig(k_probes=3, trace=True)
    by_numbers = [StragglerDetector(rank, world_size, config) for rank in range(world_size)]
    by_objects = [StragglerDetector(rank, world_size, config) for rank in range(world_size)]
    noise = random.Random(4)
    events = {"numbers": [], "objects": []}
    for step in range(90):
        compute = [0.02 * (1 + noise.uniform(-0.05, 0.05)) * (2.5 if rank == 2 and step >= 30 else 1)
                   for rank in range(world_size)]
        waits = [[noise.uniform(0, 0.001) for _ in range(2 * (world_size - 1))] for _ in range(world_size)]
        if step >= 30:
            waits[3][0] += 0.02  # the slow worker's successor waits for it in round 0
        for rank in range(world_size):
            by_numbers[rank].start_step(step, compute[rank])
            by_objects[rank].start_step(step, compute[rank])
        for _ in range(2 * (world_size - 1)):
            numbers = [detector.make_gossip() for detector in by_numbers]
            contents = [detector.gossip_contents() for detector in by_objects]
            for sender in range(world_size):
                by_numbers[(sender + 1) % world_size].receive_gossip(numbers[sender], sender=sender)
                by_objects[(sender + 1) % world_size].absorb_gossip(*contents[sender], sender=sender)
        for rank in range(world_size):
            events["numbers"] += by_numbers[rank].finish_step(step, waits[rank])
            events["objects"] += by_objects[rank].finish_step(step, waits[rank])
    assert events["numbers"] == events["objects"]
    assert any(event["action"] == "confirmed" for event in events["objects"])


def test_without_noise_the_waits_follow_the_ring_exactly(tmp_path):
    config = simulated_config(tmp_path, workers=6, steps=30, compute_noise=0.0, latency_noise=0.0,
                              fault=FaultConfig(kind="compute", workers=[1], multiplier=2.0, start_step=10))
    events = read_events(run_training(config).events_path)
    # The first slow step, after steps in which everybody was exactly equal: in reduce-scatter round s,
    # the worker s + 1 places after the slow one waits longest. (In later steps the workers start at
    # slightly different times, because the delay reached them through the ring one hop after another.)
    step_10 = {event["rank"]: event for event in events if event["event"] == "step" and event["step"] == 10}
    for rounds_after in range(5):
        waiting_worker = (1 + 1 + rounds_after) % 6
        waits = step_10[waiting_worker]["wait_seconds_per_round"][:5]
        assert waits.index(max(waits)) == rounds_after
    assert step_10[1]["compute_seconds"] == pytest.approx(2 * 0.0117)


def test_only_faults_the_simulator_can_simulate_are_accepted(tmp_path):
    config = simulated_config(tmp_path, fault=FaultConfig(kind="container_cpu", workers=[1]))
    with pytest.raises(ConfigError):
        config.validate()
    simulated_config(tmp_path, fault=FaultConfig(kind="link", workers=[1], link_delay_seconds=0.01)).validate()


def test_the_timing_model_can_be_fitted_to_a_measured_run():
    rng = random.Random(0)
    events = [{"event": "run_start", "rank": 0, "world_size": 4}]
    for step in range(60):
        for rank in range(4):
            events.append({"event": "step", "rank": rank, "step": step, "compute_seconds": 0.012 * rng.lognormvariate(0, 0.2),
                           "step_seconds": 0.030, "allreduce_seconds": 0.0102, "wait_seconds": 0.0072, "fault_active": False,
                           "wait_seconds_per_round": [0.001] * 3 + [0.0006] * 3})
    fitted = fit_timing_model(events)
    assert fitted["compute_ms"] == pytest.approx(12, rel=0.1)
    assert fitted["compute_noise"] == pytest.approx(0.2, rel=0.25)
    assert fitted["message_latency_ms"] == pytest.approx(0.6)
    assert fitted["allreduce_data_ms"] == pytest.approx(4 * 0.5)  # 3 ms busy over 6 rounds, times 4 workers
    with pytest.raises(ValueError):
        fit_timing_model(events[:10])


def test_detection_load_comes_from_the_design():
    load = detection_load(64)
    assert load["swim_per_message"] == 8 * 6 + 4 * 3  # default gossip_rows and verdict_slots
    assert load["swim_per_step"] == load["swim_per_message"] * 2 * 63
    assert load["central_per_step"] == 2 * 63


def test_a_sweep_over_group_sizes_writes_the_scaling_report(tmp_path):
    results = tmp_path / "results" / "sim"
    results.mkdir(parents=True)
    rows = []
    for workers in (4, 16):
        for repeat in range(2):
            for detector in ("swim", "central"):
                base = {"detector": detector, "workers": workers, "repeat": repeat, "backend": "simulated",
                        "simulated": True, "false_suspicions": 0, "mean_step_ms": 30.0, "compute_ratio": 2.0,
                        "delay_after_applied_steps": None}
                rows.append({**base, "fault": "healthy", "faulty_workers": 0, "culprit_found": None,
                             "delay_steps": None, "everyone_knew_delay_steps": None, "false_confirmations": repeat})
                rows.append({**base, "fault": "x2", "faulty_workers": 1, "culprit_found": True, "delay_steps": 5 + workers // 4,
                             "everyone_knew_delay_steps": 7, "false_confirmations": 0})
    (results / "runs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (results / "machine.json").write_text(json.dumps({"processors": 2, "platform": "test", "python": "3", "torch": "x",
                                                     "date": "2026-10-25"}))
    text = write_results(results).read_text()
    assert "SIMULATION" in text and "| 16 | 2/2, 9 (9-9) | 2/2, 9 (9-9) |" in text
    for image in ("scaling_delay.png", "scaling_false_confirmations.png", "scaling_load.png"):
        assert (results / image).stat().st_size > 1000


def test_the_simulate_command(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "simulated.yaml"
    config_path.write_text(f"name: s\nrun_directory: {tmp_path / 'runs'}\ncluster: {{world_size: 6, backend: simulated}}\n"
                           "train: {steps: 60}\nfault: {kind: compute, workers: [2], multiplier: 3.0, start_step: 20}\n")
    assert main(["simulate", "--config", str(config_path), "--workers", "10", "--steps", "70"]) == 0
    printed = capsys.readouterr().out
    assert "10 workers, backend=simulated" in printed and "SIMULATION" in printed


def test_the_dashboard_shows_a_simulated_run_and_says_it_is_simulated(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    from pathlib import Path

    config = simulated_config(tmp_path, workers=24, steps=80)
    run_training(config)
    monkeypatch.setenv("SWIM_RUNS_DIRECTORY", config.run_directory)
    app_path = Path(__file__).resolve().parents[1] / "swim_straggler" / "dashboard" / "app.py"
    app = testing.AppTest.from_file(str(app_path), default_timeout=60)
    app.run()
    assert not app.exception, app.exception
    assert app.warning and app.warning[0].value.startswith("SIMULATION")
