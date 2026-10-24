"""Command line entry point: train, demo (--compare for the baselines), sweep, report, dashboard, build-image (Docker) and worker (one worker, used inside containers)."""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

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
        if summary["detector_mode"] == "central":
            print("    every worker knew by          : not applicable (only the central monitor, worker 0, decides)")
        else:
            print(f"    every worker knew by          : step {report['all_workers_knew_at_step']}")
            if report["workers_that_never_learned"]:
                print(f"    never learned                 : workers {report['workers_that_never_learned']}")
    if summary["detector_mode"] == "steptime":
        print("  note: with step times only, each worker can judge nobody but itself; in synchronous training")
        print("        all step times rise together, so healthy workers end up accusing themselves.")
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
    if arguments.compare:
        return run_comparison(config, arguments.repeats)
    return run_and_report(config)


COMPARED_MODES = ("swim", "central", "steptime")


def print_table(rows):
    columns = list(rows[0])
    widths = [max(len(column), *(len(str(row[column])) for row in rows)) for column in columns]
    print("  ".join(column.ljust(width) for column, width in zip(columns, widths)))
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(columns, widths)))


def comparison_plan(base_config, repeats):
    """The runs of `demo --compare`: every detector, same fault, same seeds."""
    from swim_straggler.evaluation.sweep import PlannedRun

    planned = []
    for repeat in range(repeats):
        for mode in COMPARED_MODES:
            config = copy.deepcopy(base_config)
            config.detector.mode = mode
            config.train.seed = base_config.train.seed + repeat
            config.name = f"{base_config.name}_{mode}"
            planned.append(PlannedRun(mode, "from config", repeat, config))
    return planned


def comparison_table_rows(summaries, show_applied_delay):
    rows = []
    for summary in summaries:
        delay = "-"
        if summary["median_delay_steps"] is not None:
            delay = f"{summary['median_delay_steps']:.0f}"
            if summary["runs"] > 1:
                delay += f" ({summary['min_delay_steps']:.0f}-{summary['max_delay_steps']:.0f})"
        row = {
            "mode": summary["detector"],
            "culprit found": "-" if summary["found"] is None else f"{summary['found']} of {summary['runs']}",
            "delay (steps)": delay,
        }
        if show_applied_delay:
            applied = summary["median_delay_after_applied_steps"]
            row["delay after fault command returned"] = "-" if applied is None else f"{applied:.0f}"
        row["runs with healthy wrongly confirmed"] = f"{summary['runs_with_false_confirmation']} of {summary['runs']}"
        row["healthy wrongly suspected"] = summary["false_suspicions"]
        row["mean step (ms)"] = f"{summary['mean_step_ms']:.1f}"
        rows.append(row)
    return rows


def run_comparison(base_config, repeats=1):
    """Run the same config with every detector (same fault, same seeds) and print one line per detector."""
    from swim_straggler.evaluation.metrics import summarize_sweep
    from swim_straggler.evaluation.sweep import run_planned

    warn_if_too_many_workers(base_config)
    try:
        rows = run_planned(comparison_plan(base_config, repeats), progress=lambda text: print("running " + text))
    except (ConfigError, ContainerError) as error:
        print(f"error: {error}")
        return 1
    print()
    print(f"fault: {base_config.fault.kind} on workers {base_config.fault.workers} from step {base_config.fault.start_step}, "
          f"{base_config.cluster.world_size} workers, {base_config.train.steps} steps, {repeats} run(s) per detector"
          + ("" if repeats > 1 else " (one run: timing varies between runs, use --repeats for more)"))
    is_container_fault = base_config.fault.kind.startswith("container_")
    print_table(comparison_table_rows(summarize_sweep(rows), show_applied_delay=is_container_fault))
    if is_container_fault:
        print("(container faults need time to apply: the second delay counts from when the fault command returned)")
    return 0


def command_sweep(arguments):
    from swim_straggler.evaluation.report import write_results
    from swim_straggler.evaluation.sweep import run_sweep

    try:
        results_directory, rows = run_sweep(arguments.spec, arguments.results)
    except (ConfigError, ContainerError, OSError) as error:
        print(f"error: {error}")
        return 1
    results_path = write_results(results_directory)
    print(f"{len(rows)} runs finished. Results: {results_path}")
    return 0


def command_report(arguments):
    from swim_straggler.evaluation.report import write_results

    if not (Path(arguments.results) / "runs.jsonl").exists():
        print(f"error: no runs.jsonl in {arguments.results}; run a sweep first")
        return 1
    results_path = write_results(arguments.results)
    print(f"wrote {results_path} and the plots next to it")
    return 0


def command_dashboard(arguments):
    """Start the Streamlit dashboard in the browser."""
    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    try:
        import streamlit  # noqa: F401  (only to give a clear message when it is missing)
    except ImportError:
        print('error: the dashboard needs Streamlit. Install it with:  pip install -e ".[dashboard]"')
        return 1
    return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--", "--runs", arguments.runs]).returncode


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
    demo_parser.add_argument("--compare", action="store_true",
                             help="also run the baseline detectors (central monitor, step time) and print a comparison table")
    demo_parser.add_argument("--repeats", type=int, default=1, help="with --compare: runs per detector (default 1)")
    demo_parser.set_defaults(handler=command_demo)

    sweep_parser = subcommands.add_parser("sweep", help="repeat runs over detectors and faults, then write results and plots")
    sweep_parser.add_argument("--spec", default="experiments/sweeps/main.yaml", help="sweep file (default: the main sweep)")
    sweep_parser.add_argument("--results", default="results", help="folder for results (default: results)")
    sweep_parser.set_defaults(handler=command_sweep)

    report_parser = subcommands.add_parser("report", help="rebuild results.md and the plots from a sweep's runs.jsonl")
    report_parser.add_argument("--results", required=True, help="a sweep's results folder, for example results/main")
    report_parser.set_defaults(handler=command_report)

    dashboard_parser = subcommands.add_parser("dashboard", help="open the dashboard in the browser")
    dashboard_parser.add_argument("--runs", default="runs", help="folder that holds the runs (default: runs)")
    dashboard_parser.set_defaults(handler=command_dashboard)

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
