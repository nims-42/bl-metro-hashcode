"""Command line entry point. Only `train` exists so far, and it just checks the config."""

import argparse
import os

from swim_straggler.config import ConfigError, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="swim-straggler")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="run distributed training").add_argument("--config", required=True)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}")
        return 1
    cpus = os.cpu_count() or 1
    if cfg.cluster.world_size > cpus:
        print(f"WARNING: {cfg.cluster.world_size} workers but only {cpus} CPUs: workers will fight for CPU "
              "and create fake stragglers.")
    print(f"config OK: {cfg.name}, {cfg.cluster.world_size} workers")
    print("training is not implemented yet (Milestone 2)")
    return 2
