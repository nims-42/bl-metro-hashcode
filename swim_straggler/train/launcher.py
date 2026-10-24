"""Starts all workers, waits for them, and merges their event files."""

import socket
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import multiprocessing as torch_multiprocessing

from swim_straggler.telemetry.events import merge_event_files
from swim_straggler.telemetry.trace import write_trace
from swim_straggler.train.data import prepare_dataset
from swim_straggler.train.worker import create_detector, train_one_rank
from swim_straggler.transport.gloo import GlooTransport
from swim_straggler.transport.pipes import run_ranks_in_threads


@dataclass
class TrainingResult:
    run_directory: Path
    events_path: Path
    dataset_source: str


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe_socket:
        probe_socket.bind(("127.0.0.1", 0))
        return probe_socket.getsockname()[1]


def gloo_worker_entry(rank, config, dataset_source, run_directory_text, master_port):
    """Entry point of one worker process (torch.multiprocessing passes the rank first)."""
    torch.set_num_threads(1)  # more threads per worker would create fake stragglers
    transport = GlooTransport(rank, config.cluster.world_size, config.cluster.master_address, master_port)
    try:
        train_one_rank(transport, config, dataset_source, Path(run_directory_text))
    finally:
        transport.close()


def run_training(config, run_directory=None):
    """Train with all workers and return where the merged event log was written (trace.log is written next to it)."""
    create_detector(config, rank=0, world_size=config.cluster.world_size)  # fail early on an unusable detector mode
    if run_directory is None:
        time_text = time.strftime("%Y%m%d_%H%M%S")
        run_directory = Path(config.run_directory) / f"{config.name}_{time_text}"
    run_directory = Path(run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)

    if config.cluster.backend == "simulated":  # no training at all; see simulation/simulator.py
        from swim_straggler.simulation.simulator import simulate_run

        events_path = simulate_run(config, run_directory)
        write_trace(run_directory)
        return TrainingResult(run_directory=run_directory, events_path=events_path, dataset_source="simulated")

    dataset_source = prepare_dataset(config.train)

    if config.cluster.backend == "docker":
        from swim_straggler.containers.runner import run_in_containers

        result = run_in_containers(config, run_directory)
        write_trace(result.run_directory)
        return result

    world_size = config.cluster.world_size
    if config.cluster.backend == "gloo":
        master_port = config.cluster.master_port or find_free_port()
        torch_multiprocessing.spawn(
            gloo_worker_entry,
            args=(config, dataset_source, str(run_directory), master_port),
            nprocs=world_size,
            join=True,
        )
    else:
        torch.set_num_threads(1)

        def run_one_rank(transport):
            train_one_rank(transport, config, dataset_source, run_directory)

        run_ranks_in_threads(world_size, run_one_rank, timeout_seconds=600.0)

    rank_files = [run_directory / f"events_rank_{rank}.jsonl" for rank in range(world_size)]
    events_path = run_directory / "events.jsonl"
    merge_event_files(rank_files, events_path)
    write_trace(run_directory)
    return TrainingResult(run_directory=run_directory, events_path=events_path, dataset_source=dataset_source)
