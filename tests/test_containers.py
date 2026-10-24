"""Everything about the Docker backend that can be checked without Docker: files, commands and logic."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from swim_straggler.config import Config, ConfigError, FaultConfig, load_config
from swim_straggler.containers.commands import ContainerError, build_fault_action
from swim_straggler.containers.compose import build_compose, config_for_containers, container_name, project_name
from swim_straggler.containers.controller import FaultController, latest_step
from swim_straggler.containers.runner import run_in_containers
from swim_straggler.telemetry.events import read_events, summarize_detection
from swim_straggler.train.launcher import find_free_port
from test_training import make_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLOW_C2 = "docker update --cpu-period 2000 --cpu-quota 1000 c2"
NORMAL_C2 = "docker update --cpu-period 100000 --cpu-quota 100000 c2"


def docker_config(tmp_path, kind="container_cpu", tool="docker", **fault_fields):
    config = make_config(tmp_path, backend="docker", world_size=4, steps=20)
    config.fault = FaultConfig(kind=kind, workers=[2], start_step=5, fault_tool=tool, **fault_fields)
    config.cluster.container_cpus = 1.0
    config.validate()
    return config


# ---- configuration -------------------------------------------------------------------------

def test_container_faults_need_the_docker_backend(tmp_path):
    config = make_config(tmp_path, backend="pipes")
    config.fault = FaultConfig(kind="container_cpu", workers=[1])
    with pytest.raises(ConfigError, match="docker"):
        config.validate()


def test_docker_configs_in_the_repository_are_valid():
    for name in ("docker_cpu", "docker_stress", "docker_network", "docker_pause"):
        assert load_config(PROJECT_ROOT / "experiments" / "configs" / f"{name}.yaml").cluster.backend == "docker"


def test_a_worker_never_injects_container_faults_itself():
    from swim_straggler.faults.inject import FaultInjector

    fault = FaultConfig(kind="container_cpu", workers=[2], start_step=0)
    assert not FaultInjector(fault, rank=2).is_fault_active(10)


# ---- compose file ---------------------------------------------------------------------------

def test_compose_has_one_container_per_worker(tmp_path):
    config = docker_config(tmp_path)
    compose = build_compose(config, tmp_path / "run1", tmp_path / "data", "synthetic")
    services = compose["services"]
    assert sorted(services) == ["worker0", "worker1", "worker2", "worker3"]
    worker_two = services["worker2"]
    assert worker_two["command"][:2] == ["worker", "--config"]
    assert worker_two["command"][worker_two["command"].index("--rank") + 1] == "2"
    assert (worker_two["cpu_period"], worker_two["cpu_quota"]) == (100000, 100000)
    assert "cpus" not in worker_two  # docker refuses to change period/quota later on a container made with `cpus`
    assert "NET_ADMIN" in worker_two["cap_add"]
    assert worker_two["environment"]["OMP_NUM_THREADS"] == "1"
    assert {volume["target"] for volume in worker_two["volumes"]} == {"/run", "/app/data", "/app/swim_straggler"}
    assert worker_two["container_name"] == container_name(project_name(config, tmp_path / "run1"), 2)


def test_containers_run_the_current_code_not_the_code_baked_into_the_image(tmp_path):
    config = docker_config(tmp_path)
    worker = build_compose(config, tmp_path / "run1", tmp_path / "data", "synthetic")["services"]["worker1"]
    mounts = {volume["target"]: volume for volume in worker["volumes"]}
    assert mounts["/app/swim_straggler"]["source"] == str(Path(PROJECT_ROOT / "swim_straggler").resolve())
    assert mounts["/app/swim_straggler"]["read_only"] is True
    assert worker["environment"]["PYTHONPATH"] == "/app"

    config.cluster.mount_source_code = False
    worker = build_compose(config, tmp_path / "run1", tmp_path / "data", "synthetic")["services"]["worker1"]
    assert "PYTHONPATH" not in worker["environment"] and len(worker["volumes"]) == 2


def test_containers_find_worker_zero_by_its_service_name(tmp_path):
    container_config = config_for_containers(docker_config(tmp_path))
    assert container_config.cluster.master_address == "worker0"
    assert container_config.cluster.master_port == 29500
    assert container_config.train.data_directory == "data"
    container_config.validate()


def test_the_written_container_config_can_be_loaded_again(tmp_path):
    from swim_straggler.containers.compose import write_container_config

    config = docker_config(tmp_path)
    config.detector.mode = "off"  # YAML must not turn a bare `off` into False
    write_container_config(config, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml").detector.mode == "off"


# ---- fault commands -------------------------------------------------------------------------

def commands_of(action):
    return [" ".join(command) for command in action.start], [" ".join(command) for command in action.stop]


def test_cpu_fault_halves_the_processor_limit_and_restores_it():
    fault = FaultConfig(kind="container_cpu", workers=[2], multiplier=2.0)
    cluster = Config().cluster
    cluster.container_cpus = 1.0
    start, stop = commands_of(build_fault_action(fault, cluster, "box"))
    assert start == ["docker update --cpu-period 2000 --cpu-quota 1000 box"]  # half a processor, in 1 ms slices
    assert stop == ["docker update --cpu-period 100000 --cpu-quota 100000 box"]


def test_network_delay_uses_tc_netem_or_pumba():
    cluster = Config().cluster
    fault = FaultConfig(kind="container_network_delay", workers=[2], network_delay_ms=30)
    start, stop = commands_of(build_fault_action(fault, cluster, "box"))
    assert start == ["docker exec box tc qdisc add dev eth0 root netem delay 30ms"]
    assert stop == ["docker exec box tc qdisc del dev eth0 root"]

    fault.fault_tool = "pumba"
    start, stop = commands_of(build_fault_action(fault, cluster, "box"))
    assert "pumba" in start[0] and "netem" in start[0] and "delay --time 30 box" in start[0]
    assert stop == ["docker stop box_pumba"]


def test_throttle_slices_scale_with_the_multiplier_and_the_container_limit():
    from swim_straggler.containers.commands import throttled_cpu_limit

    cluster = Config().cluster
    cluster.container_cpus = 1.0
    assert throttled_cpu_limit(cluster, 2.0) == (2000, 1000)
    assert throttled_cpu_limit(cluster, 4.0) == (4000, 1000)
    cluster.container_cpus = 0.5
    period, quota = throttled_cpu_limit(cluster, 2.0)
    assert quota / period == pytest.approx(0.25) and quota >= 1000  # the Linux minimum quota


def test_stress_and_pause_commands():
    cluster = Config().cluster
    start, stop = commands_of(build_fault_action(FaultConfig(kind="container_stress", workers=[1]), cluster, "box"))
    assert start[0].startswith("docker exec -d box stress-ng --cpu 2") and stop == ["docker exec box pkill stress-ng"]
    start, stop = commands_of(build_fault_action(FaultConfig(kind="container_pause", workers=[1]), cluster, "box"))
    assert start == ["docker pause box"] and stop == ["docker unpause box"]
    start, stop = commands_of(build_fault_action(FaultConfig(kind="container_stress", workers=[1], fault_tool="pumba"), cluster, "box"))
    assert " stress " in start[0] and stop == ["docker stop box_pumba"]


def test_cpu_fault_ignores_the_pumba_setting_because_pumba_cannot_throttle():
    cluster = Config().cluster
    start, _ = commands_of(build_fault_action(FaultConfig(kind="container_cpu", workers=[1], fault_tool="pumba"), cluster, "box"))
    assert start[0].startswith("docker update")


# ---- fault controller -----------------------------------------------------------------------

class RecordingRunner:
    def __init__(self):
        self.commands = []

    def run(self, command, check=True):
        self.commands.append(" ".join(command))
        return ""


def write_steps(path, first, last):
    with open(path, "a", encoding="utf-8") as events_file:
        for step in range(first, last + 1):
            events_file.write(json.dumps({"event": "step", "rank": 0, "step": step}) + "\n")


def test_latest_step_ignores_a_half_written_last_line(tmp_path):
    path = tmp_path / "events_rank_0.jsonl"
    assert latest_step(path) == -1
    write_steps(path, 0, 7)
    with open(path, "a", encoding="utf-8") as events_file:
        events_file.write('{"event": "step", "ra')
    assert latest_step(path) == 7


def run_controller_while(controller, steps_to_write):
    thread = threading.Thread(target=controller.run)
    thread.start()
    for first, last in steps_to_write:
        write_steps(controller.worker_events_path, first, last)
        time.sleep(0.15)
    controller.stop_requested.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_controller_applies_the_fault_at_the_start_step_and_removes_it_after_the_duration(tmp_path):
    config = docker_config(tmp_path, duration=4)
    runner = RecordingRunner()
    controller = FaultController(config, ["c0", "c1", "c2", "c3"], tmp_path, runner, poll_seconds=0.01)
    run_controller_while(controller, [(0, 3), (4, 6), (7, 12)])
    assert runner.commands == [SLOW_C2, NORMAL_C2]
    events = read_events(controller.controller_events_path)
    assert [(event["event"], event["rank"]) for event in events] == [("fault_started", 2), ("fault_ended", 2)]
    assert events[0]["step"] >= 5 and events[0]["source"] == "controller"
    assert events[1]["step"] >= 9


def test_controller_does_nothing_before_the_start_step(tmp_path):
    runner = RecordingRunner()
    controller = FaultController(docker_config(tmp_path), ["c0", "c1", "c2", "c3"], tmp_path, runner, poll_seconds=0.01)
    run_controller_while(controller, [(0, 3)])
    assert runner.commands == [] and not controller.controller_events_path.exists()


def test_controller_cleans_up_a_fault_that_has_no_end(tmp_path):
    runner = RecordingRunner()
    controller = FaultController(docker_config(tmp_path), ["c0", "c1", "c2", "c3"], tmp_path, runner, poll_seconds=0.01)
    run_controller_while(controller, [(0, 8)])
    assert runner.commands == [SLOW_C2, NORMAL_C2]


def test_pause_ends_after_seconds_not_after_steps(tmp_path):
    config = docker_config(tmp_path, kind="container_pause", pause_seconds=0.2)
    runner = RecordingRunner()
    controller = FaultController(config, ["c0", "c1", "c2", "c3"], tmp_path, runner, poll_seconds=0.01)
    run_controller_while(controller, [(0, 6), (6, 6), (6, 6)])  # steps stop advancing, as in a real stall
    assert runner.commands == ["docker pause c2", "docker unpause c2"]


# ---- the whole run, with a pretend Docker ---------------------------------------------------

class PretendDocker:
    """Answers docker commands like a working Docker and writes the files the workers would write."""

    def __init__(self, fail_worker=None):
        self.commands = []
        self.finished = False
        self.run_directory = None
        self.fail_worker = fail_worker
        self.world_size = 4

    def run(self, command, check=True):
        self.commands.append(" ".join(command))
        if command[:3] == ["docker", "compose", "-f"] and "up" in command:
            compose = yaml.safe_load(Path(command[3]).read_text())
            self.world_size = len(compose["services"])
            self.run_directory = Path(compose["services"]["worker0"]["volumes"][0]["source"])
            threading.Thread(target=self.pretend_training).start()
        if command[:2] == ["docker", "inspect"]:
            if self.fail_worker is not None and command[-1].endswith(f"worker{self.fail_worker}") and self.finished:
                return "exited 1"
            return "exited 0" if self.finished else "running 0"
        if command[:2] == ["docker", "logs"]:
            return "line one\nTraceback: boom" if self.fail_worker is not None else "all fine"
        return ""

    def pretend_training(self):
        for step in range(20):
            for rank in range(self.world_size):
                record = {"event": "step", "rank": rank, "step": step, "loss": 1.0, "step_seconds": 0.03, "compute_seconds": 0.01}
                with open(self.run_directory / f"events_rank_{rank}.jsonl", "a", encoding="utf-8") as events_file:
                    events_file.write(json.dumps(record) + "\n")
            time.sleep(0.03)
        for rank in range(self.world_size):
            with open(self.run_directory / f"events_rank_{rank}.jsonl", "a", encoding="utf-8") as events_file:
                events_file.write(json.dumps({"event": "run_end", "rank": rank, "parameter_checksum": 1.0}) + "\n")
        self.finished = True


def test_a_whole_container_run_with_a_pretend_docker(tmp_path):
    config = docker_config(tmp_path, duration=5)
    config.fault.start_step = 3
    docker = PretendDocker()
    result = run_in_containers(config, tmp_path / "run", command_runner=docker)

    events = read_events(result.events_path)
    assert len([event for event in events if event["event"] == "step"]) == 4 * 20
    assert summarize_detection(events)["faulty_workers"] == [2]  # ground truth came from the controller
    assert any(command.startswith("docker update --cpu-period 2000") for command in docker.commands)
    assert any("compose" in command and " down" in command for command in docker.commands)  # always cleaned up
    assert (tmp_path / "run" / "container_logs").is_dir()
    assert (tmp_path / "run" / "compose.yaml").exists() and (tmp_path / "run" / "config.yaml").exists()


def test_a_failing_container_is_reported_with_its_log(tmp_path):
    config = docker_config(tmp_path)
    with pytest.raises(ContainerError, match="Traceback: boom"):
        run_in_containers(config, tmp_path / "run", command_runner=PretendDocker(fail_worker=1))


def test_missing_docker_gives_a_helpful_message(tmp_path):
    class NoDocker:
        def run(self, command, check=True):
            raise ContainerError("cannot run 'docker'")

    with pytest.raises(ContainerError, match="Docker Desktop"):
        run_in_containers(docker_config(tmp_path), tmp_path / "run", command_runner=NoDocker())


def test_missing_image_tells_the_user_how_to_build_it(tmp_path):
    class NoImage:
        def run(self, command, check=True):
            if command[:3] == ["docker", "image", "inspect"]:
                raise ContainerError("no such image")
            return ""

    with pytest.raises(ContainerError, match="build-image"):
        run_in_containers(docker_config(tmp_path), tmp_path / "run", command_runner=NoImage())


# ---- the worker command, the part a container runs, without Docker --------------------------

@pytest.mark.slow
def test_three_worker_commands_train_together_like_three_containers(tmp_path):
    """Each `swim-straggler worker` below is exactly what a container executes; here they are plain processes."""
    config = make_config(tmp_path, backend="gloo", world_size=3, steps=15)
    config.cluster.master_port = find_free_port()
    config.fault = FaultConfig(kind="none")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    processes = [
        subprocess.Popen([sys.executable, "-m", "swim_straggler.cli", "worker", "--config", str(config_path),
                          "--rank", str(rank), "--dataset-source", "synthetic", "--run-directory", str(run_directory)],
                         cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
        for rank in range(3)
    ]
    assert [process.wait(timeout=120) for process in processes] == [0, 0, 0]
    for rank in range(3):
        events = read_events(run_directory / f"events_rank_{rank}.jsonl")
        assert len([event for event in events if event["event"] == "step"]) == 15
        assert events[-1]["event"] == "run_end"


# ---- checking that a fault really had an effect ---------------------------------------------

def make_events(compute_before, compute_during, fault_kind="container_cpu", issued=20, applied=26):
    events = []
    for step in range(60):
        for rank in range(2):
            seconds = compute_during if (rank == 1 and step >= applied) else compute_before
            events.append({"event": "step", "rank": rank, "step": step, "compute_seconds": seconds})
    events.append({"event": "fault_started", "rank": 1, "step": issued, "step_after_command": applied,
                   "fault_kind": fault_kind, "source": "controller"})
    return events


def test_the_measured_effect_of_a_fault_is_reported():
    report = summarize_detection(make_events(0.010, 0.020))["per_fault"][1]
    assert report["compute_ratio"] == pytest.approx(2.0)
    assert (report["fault_issued_step"], report["fault_applied_step"]) == (20, 26)


def test_a_fault_with_no_effect_is_visible_in_the_summary(capsys):
    from swim_straggler.cli import print_detection_summary

    print_detection_summary(make_events(0.010, 0.0101))
    assert "nothing to detect" in capsys.readouterr().out


def test_delays_are_given_from_issue_and_from_application():
    events = make_events(0.010, 0.020)
    events.append({"event": "detector", "rank": 0, "step": 30, "action": "confirmed", "subject": 1})
    report = summarize_detection(events)["per_fault"][1]
    assert (report["delay_steps"], report["delay_steps_after_applied"]) == (10, 4)


def test_accusing_the_culprit_before_its_fault_started_is_a_false_alarm_not_a_detection():
    events = make_events(0.010, 0.020)
    events.append({"event": "detector", "rank": 0, "step": 12, "action": "confirmed", "subject": 1})
    summary = summarize_detection(events)
    assert summary["per_fault"][1]["first_confirmed_step"] is None
    assert summary["false_confirmations"] == [(0, 1, 12)]
