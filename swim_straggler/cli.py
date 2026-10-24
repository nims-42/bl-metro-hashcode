"""Command line entry point: train, demo (--compare for the baselines), simulate, trace, sweep, report, dashboard, build-image (Docker) and worker (one worker, used inside containers)."""

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
    if config.cluster.backend == "simulated":
        return  # nothing runs in parallel; the simulator is one ordinary program
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
    if summary["loss_at_start"] is None:
        print("SIMULATION           : nothing was trained; the timings come from the simulator's model")
        print(f"mean step time       : {summary['mean_step_seconds'] * 1000:.1f} ms (simulated)")
    else:
        print(f"loss (first 5 steps) : {summary['loss_at_start']:.4f}")
        print(f"loss (last 5 steps)  : {summary['loss_at_end']:.4f}")
        print(f"mean step time       : {summary['mean_step_seconds'] * 1000:.1f} ms")
    print(f"event log            : {result.events_path}")
    print(f"step-by-step trace   : {result.run_directory / 'trace.log'}")


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
            never_learned = report["workers_that_never_learned"]
            if never_learned:
                shown = never_learned if len(never_learned) <= 12 else f"{len(never_learned)} workers"
                print(f"    never learned                 : workers {shown}")
    if summary["detector_mode"] == "steptime":
        print("  note: with step times only, each worker can judge nobody but itself; in synchronous training")
        print("        all step times rise together, so healthy workers end up accusing themselves.")
    print(f"  healthy workers wrongly suspected : {summary['false_suspicions']}  (the vote rejected {summary['rejected_suspicions']} suspicions)")
    print(f"  healthy workers wrongly confirmed : {len(summary['false_confirmations'])}")
    for rank, subject, step in summary["false_confirmations"][:5]:
        print(f"    worker {rank} confirmed worker {subject} at step {step}")
    print(f"  stragglers cleared again          : {summary['cleared_events']}")
    for rank, step in summary["network_diagnoses"]:
        print(f"  EXPERIMENTAL diagnosis            : worker {rank} at step {step}: the delay is not compute (network?)")


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
    if arguments.trace:
        config.detector.trace = True
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
    if arguments.trace:
        config.detector.trace = True
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


DEFAULT_SIMULATION_CONFIG = "experiments/configs/simulated.yaml"


def command_simulate(arguments):
    """Simulate a run: the real detectors on made-up timings (cluster.backend: simulated)."""
    from swim_straggler.simulation.simulator import fit_timing_model

    try:
        config = load_config(arguments.config)
        config.cluster.backend = "simulated"
        if arguments.workers is not None:
            config.cluster.world_size = arguments.workers
            config.fault.workers = [worker for worker in config.fault.workers if worker < arguments.workers] or [arguments.workers // 2]
            config.detector.k_probes = min(config.detector.k_probes, arguments.workers - 2)
        if arguments.steps is not None:
            config.train.steps = arguments.steps
        if arguments.trace:
            config.detector.trace = True
        if arguments.fit_from:
            run_path = Path(arguments.fit_from)
            fitted = fit_timing_model(read_events(run_path if run_path.is_file() else run_path / "events.jsonl"))
            print(f"timing model fitted to {arguments.fit_from}: {fitted}")
            print("(to keep it, copy these values into the simulation section of your config)")
            for name, value in fitted.items():
                setattr(config.simulation, name, value)
        config.validate()
    except (ConfigError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 1
    return run_and_report(config)


def command_trace(arguments):
    """Print the step-by-step trace of a finished run, optionally only part of it."""
    from swim_straggler.telemetry.trace import parse_step_range, trace_text

    run_path = Path(arguments.run)
    events_path = run_path if run_path.is_file() else run_path / "events.jsonl"
    if not events_path.exists():
        print(f"error: no events.jsonl in {run_path}")
        return 1
    try:
        first_step, last_step = parse_step_range(arguments.steps)
    except ValueError:
        print(f"error: --steps must look like 60-90, 60- or 60, not '{arguments.steps}'")
        return 1
    text = trace_text(read_events(events_path), worker=arguments.worker, about=arguments.about,
                      first_step=first_step, last_step=last_step)
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
        print(f"wrote {arguments.output}")
    else:
        print(text, end="")
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
    train_parser.add_argument("--trace", action="store_true", help="also log the reasoning behind every decision (see trace.log)")
    train_parser.set_defaults(handler=command_train)

    demo_parser = subcommands.add_parser("demo", help="train with an injected slow worker and show what the detector found")
    demo_parser.add_argument("--config", default=DEFAULT_DEMO_CONFIG, help="config file (default: the demo config)")
    demo_parser.add_argument("--workers", type=int, help="number of workers; the slow one is placed in the middle")
    demo_parser.add_argument("--compare", action="store_true",
                             help="also run the baseline detectors (central monitor, step time) and print a comparison table")
    demo_parser.add_argument("--repeats", type=int, default=1, help="with --compare: runs per detector (default 1)")
    demo_parser.add_argument("--trace", action="store_true", help="also log the reasoning behind every decision (see trace.log)")
    demo_parser.set_defaults(handler=command_demo)

    simulate_parser = subcommands.add_parser(
        "simulate", help="SIMULATION: run the real detectors on made-up timings, for groups too large to run")
    simulate_parser.add_argument("--config", default=DEFAULT_SIMULATION_CONFIG, help="config file (default: the simulation config)")
    simulate_parser.add_argument("--workers", type=int, help="number of simulated workers")
    simulate_parser.add_argument("--steps", type=int, help="number of steps")
    simulate_parser.add_argument("--trace", action="store_true", help="also log the reasoning behind every decision")
    simulate_parser.add_argument("--fit-from", help="fit the timing model to a measured run folder first")
    simulate_parser.set_defaults(handler=command_simulate)

    trace_parser = subcommands.add_parser("trace", help="show what happened in a run, step by step (all of it or a part)")
    trace_parser.add_argument("run", help="a run folder (or its events.jsonl)")
    trace_parser.add_argument("--worker", type=int, help="only what this worker did")
    trace_parser.add_argument("--about", type=int, help="only what concerns this worker (suspicions, answers, verdicts, faults)")
    trace_parser.add_argument("--steps", help="only these steps, for example 60-90, 60- or 75")
    trace_parser.add_argument("--output", help="write to this file instead of printing")
    trace_parser.set_defaults(handler=command_trace)

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
