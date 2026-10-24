"""Command line entry point. Only `train` exists so far."""

import argparse
import os

from swim_straggler.config import ConfigError, load_config
from swim_straggler.telemetry.events import read_events, summarize_training
from swim_straggler.train.launcher import run_training


def warn_if_too_many_workers(world_size):
    available_processors = os.cpu_count() or 1
    if world_size > available_processors:
        print(
            f"WARNING: {world_size} workers but only {available_processors} processors. "
            "Workers will compete for processor time and create fake stragglers."
        )


def command_train(arguments):
    try:
        config = load_config(arguments.config)
    except (ConfigError, OSError) as error:
        print(f"error: {error}")
        return 1

    warn_if_too_many_workers(config.cluster.world_size)
    print(f"training '{config.name}': {config.cluster.world_size} workers, backend={config.cluster.backend}, "
          f"fault={config.fault.kind}")
    result = run_training(config)

    summary = summarize_training(read_events(result.events_path))
    print(f"data source          : {result.dataset_source}")
    print(f"steps                : {summary['steps']}")
    print(f"loss (first 5 steps) : {summary['loss_at_start']:.4f}")
    print(f"loss (last 5 steps)  : {summary['loss_at_end']:.4f}")
    print(f"mean step time       : {summary['mean_step_seconds'] * 1000:.1f} ms")
    print(f"event log            : {result.events_path}")
    return 0


def main(argument_list=None):
    parser = argparse.ArgumentParser(prog="swim-straggler")
    subcommands = parser.add_subparsers(dest="command", required=True)
    train_parser = subcommands.add_parser("train", help="run distributed training")
    train_parser.add_argument("--config", required=True, help="path to a YAML config file")
    train_parser.set_defaults(handler=command_train)

    arguments = parser.parse_args(argument_list)
    return arguments.handler(arguments)
