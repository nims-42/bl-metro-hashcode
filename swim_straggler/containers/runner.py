"""Run a whole training in containers: prepare files, start the containers, inject the fault, collect the logs."""

import threading
import time
from pathlib import Path

from swim_straggler.containers.commands import CommandRunner, ContainerError
from swim_straggler.containers.compose import (
    build_compose, container_name, project_name, uses_container_faults, write_compose_file, write_container_config,
)
from swim_straggler.containers.controller import FaultController
from swim_straggler.telemetry.events import merge_event_files
from swim_straggler.train.data import prepare_dataset

POLL_SECONDS = 1.0


def check_docker_is_ready(runner, image):
    try:
        runner.run(["docker", "version", "--format", "{{.Server.Version}}"])
    except ContainerError as error:
        raise ContainerError(f"Docker is not available. Is Docker Desktop running?\n{error}") from error
    try:
        runner.run(["docker", "image", "inspect", image])
    except ContainerError as error:
        raise ContainerError(f"image '{image}' not found. Build it first with: swim-straggler build-image") from error


def container_state(runner, name):
    """(status, exit code) of a container, for example ("running", 0) or ("exited", 1)."""
    output = runner.run(["docker", "inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", name]).split()
    return output[0], int(output[1])


def wait_for_containers(runner, names, timeout_seconds, sleep=time.sleep):
    """Wait until all containers have exited. Returns {name: exit code}. Stops early if one fails."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        states = {name: container_state(runner, name) for name in names}
        failed = {name: code for name, (status, code) in states.items() if status in ("exited", "dead") and code != 0}
        if failed:
            return failed
        if all(status in ("exited", "dead") for status, _ in states.values()):
            return {name: code for name, (_, code) in states.items()}
        sleep(POLL_SECONDS)
    raise ContainerError(f"containers did not finish within {timeout_seconds} seconds")


def save_container_logs(runner, names, run_directory):
    log_directory = Path(run_directory) / "container_logs"
    log_directory.mkdir(exist_ok=True)
    tails = {}
    for name in names:
        try:
            logs = runner.run(["docker", "logs", name], check=False)
        except ContainerError:
            logs = ""
        (log_directory / f"{name}.log").write_text(logs, encoding="utf-8")
        tails[name] = "\n".join(logs.splitlines()[-15:])
    return tails


def run_in_containers(config, run_directory, command_runner=None):
    """Train with one container per worker. Returns a TrainingResult like the other backends."""
    from swim_straggler.train.launcher import TrainingResult  # imported here: launcher imports this module

    runner = command_runner or CommandRunner()
    check_docker_is_ready(runner, config.cluster.container_image)

    dataset_source = prepare_dataset(config.train)
    run_directory = Path(run_directory).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    data_directory = Path(config.train.data_directory).resolve()
    data_directory.mkdir(parents=True, exist_ok=True)

    project = project_name(config, run_directory)
    names = [container_name(project, rank) for rank in range(config.cluster.world_size)]
    write_container_config(config, run_directory / "config.yaml")
    compose_path = run_directory / "compose.yaml"
    write_compose_file(build_compose(config, run_directory, data_directory, dataset_source), compose_path)
    compose_command = ["docker", "compose", "-f", str(compose_path)]

    controller = None
    controller_thread = None
    if uses_container_faults(config):
        controller = FaultController(config, names, run_directory, runner)
        controller_thread = threading.Thread(target=controller.run, daemon=True)

    exit_codes = {}
    try:
        runner.run(compose_command + ["up", "-d"])
        if controller_thread:
            controller_thread.start()
        timeout_seconds = 600 + 5 * config.train.steps
        exit_codes = wait_for_containers(runner, names, timeout_seconds)
    finally:
        if controller:
            controller.stop_requested.set()
            controller_thread.join(timeout=60)
        tails = save_container_logs(runner, names, run_directory)
        runner.run(compose_command + ["down", "--timeout", "5"], check=False)

    if controller and controller.error:
        raise ContainerError(f"the fault controller failed: {controller.error}")
    failed = {name: code for name, code in exit_codes.items() if code != 0}
    if failed:
        details = "\n".join(f"--- {name} (exit code {code}) ---\n{tails[name]}" for name, code in failed.items())
        raise ContainerError(f"{len(failed)} container(s) failed:\n{details}")

    rank_files = [run_directory / f"events_rank_{rank}.jsonl" for rank in range(config.cluster.world_size)]
    controller_file = run_directory / "events_controller.jsonl"
    events_path = run_directory / "events.jsonl"
    merge_event_files(rank_files + ([controller_file] if controller_file.exists() else []), events_path)
    return TrainingResult(run_directory=run_directory, events_path=events_path, dataset_source=dataset_source)
