"""Command line entry point: train, demo, build-image (Docker) and worker (one worker, used inside containers)."""

import argparse
import os
import subprocess

from swim_straggler.config import ConfigError, load_config
from swim_straggler.containers.commands import ContainerError
from swim_straggler.telemetry.events import read_events, summarize_detection, summarize_training
from swim_straggler.train.launcher import run_training

DEFAULT_DEMO_CONFIG = "experiments/configs/demo.yaml"


def warn_if_too_many_workers(config):
    available_processors = os.cpu_count() or 1
    world_size = config.cluster.world_size
    if config.cluster.backend == "docker":
        needed = world_size * config.cluster.container_cpus
        if needed > available_processors:
            print(f"WARNING: {world_size} containers x {config.cluster.container_cpus} processors = {needed} processors "
                  f"but this computer has {available_processors}. Workers will compete and create fake stragglers.")
        print("note: Docker Desktop may give its virtual machine fewer processors than this computer has "
              "(Settings > Resources).")
    elif world_size > available_processors:
        print(
            f"WARNING: {world_size} workers but only {available_processors} processors. "
            "Workers will compete for processor time and create fake stragglers."
        )


def print_training_summary(result, events):
    summary = summarize_training(events)
    print(f"data source          : {result.dataset_source}")
    print(f"steps                : {summary['steps']}")
    print(f"loss (first 5 steps) : {summary['loss_at_start']:.4f}")
    print(f"loss (last 5 steps)  : {summary['loss_at_end']:.4f}")
    print(f"mean step time       : {summary['mean_step_seconds'] * 1000:.1f} ms")
    print(f"event log            : {result.events_path}")


def print_detection_summary(events):
    summary = summarize_detection(events)
    print()
    print("detection (steps are training steps; the injected fault is the ground truth)")
    if not summary["faulty_workers"]:
        print("  no fault was injected")
    for culprit, report in summary["per_fault"].items():
        print(f"  worker {culprit}: fault issued at step {report['fault_issued_step']}, "
              f"in place when the command returned at step {report['fault_applied_step']}")
        ratio = report["compute_ratio"]
        if ratio is not None:
            print(f"    measured effect on its compute time : x{ratio:.2f}")
            if ratio < 1.15 and summary["fault_kinds"].get(culprit) not in ("link", "container_network_delay", "container_pause"):
                print("    WARNING: the fault barely changed this worker's compute time, so there was nothing to detect.")
        print(f"    first suspected by any worker : step {report['first_suspected_step']}")
        print(f"    first confirmed by any worker : step {report['first_confirmed_step']}"
              f"  (delay {report['delay_steps']} steps after the command was issued, "
              f"{report['delay_steps_after_applied']} after it returned)")
        print(f"    every worker knew by          : step {report['all_workers_knew_at_step']}")
        if report["workers_that_never_learned"]:
            print(f"    never learned                 : workers {report['workers_that_never_learned']}")
    print(f"  healthy workers wrongly suspected : {summary['false_suspicions']}  (the vote rejected {summary['rejected_suspicions']} suspicions)")
    print(f"  healthy workers wrongly confirmed : {len(summary['false_confirmations'])}")
    for rank, subject, step in summary["false_confirmations"][:5]:
        print(f"    worker {rank} confirmed worker {subject} at step {step}")
    print(f"  stragglers cleared again          : {summary['cleared_events']}")


def run_and_report(config):
    warn_if_too_many_workers(config)
    print(f"training '{config.name}': {config.cluster.world_size} workers, backend={config.cluster.backend}, "
          f"fault={config.fault.kind}, detector={config.detector.mode}")
    try:
        result = run_training(config)
    except (ConfigError, ContainerError) as error:
        print(f"error: {error}")
        return 1
    events = read_events(result.events_path)
    print_training_summary(result, events)
    if config.detector.mode != "off":
        print_detection_summary(events)
    return 0


def command_train(arguments):
    try:
        config = load_config(arguments.config)
    except (ConfigError, OSError) as error:
        print(f"error: {error}")
        return 1
    return run_and_report(config)


def command_demo(arguments):
    try:
        config = load_config(arguments.config)
        if arguments.workers is not None:
            config.cluster.world_size = arguments.workers
            config.fault.workers = [arguments.workers // 2]  # a worker in the middle of the ring
            config.detector.k_probes = min(config.detector.k_probes, arguments.workers - 2)
            config.validate()
    except (ConfigError, OSError) as error:
        print(f"error: {error}")
        return 1
    return run_and_report(config)


def command_worker(arguments):
    """Run ONE worker. This is what every container executes; it also works on a plain computer."""
    from swim_straggler.train.launcher import gloo_worker_entry

    try:
        config = load_config(arguments.config)
    except (ConfigError, OSError) as error:
        print(f"error: {error}")
        return 1
    if arguments.master_address:
        config.cluster.master_address = arguments.master_address
    master_port = arguments.master_port or config.cluster.master_port
    if master_port == 0:
        print("error: the master port is 0; set cluster.master_port or pass --master-port")
        return 1
    gloo_worker_entry(arguments.rank, config, arguments.dataset_source, arguments.run_directory, master_port)
    return 0


def command_build_image(arguments):
    """Build the Docker image from the Dockerfile in the current folder."""
    config = load_config(arguments.config) if arguments.config else None
    image = config.cluster.container_image if config else "swim-straggler:latest"
    try:
        return subprocess.run(["docker", "build", "-t", image, "."]).returncode
    except FileNotFoundError:
        print("error: 'docker' was not found. Is Docker installed and on the PATH?")
        return 1


def main(argument_list=None):
    parser = argparse.ArgumentParser(prog="swim-straggler")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_parser = subcommands.add_parser("train", help="run distributed training")
    train_parser.add_argument("--config", required=True, help="path to a YAML config file")
    train_parser.set_defaults(handler=command_train)

    demo_parser = subcommands.add_parser("demo", help="train with an injected slow worker and show what the detector found")
    demo_parser.add_argument("--config", default=DEFAULT_DEMO_CONFIG, help="config file (default: the demo config)")
    demo_parser.add_argument("--workers", type=int, help="number of workers; the slow one is placed in the middle")
    demo_parser.set_defaults(handler=command_demo)

    worker_parser = subcommands.add_parser("worker", help="run one worker (used inside containers)")
    worker_parser.add_argument("--config", required=True)
    worker_parser.add_argument("--rank", type=int, required=True)
    worker_parser.add_argument("--dataset-source", required=True, choices=["mnist", "synthetic"])
    worker_parser.add_argument("--run-directory", required=True)
    worker_parser.add_argument("--master-address")
    worker_parser.add_argument("--master-port", type=int)
    worker_parser.set_defaults(handler=command_worker)

    build_parser = subcommands.add_parser("build-image", help="build the Docker image (run in the project folder)")
    build_parser.add_argument("--config", help="optional config file, only to read cluster.container_image")
    build_parser.set_defaults(handler=command_build_image)

    arguments = parser.parse_args(argument_list)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
