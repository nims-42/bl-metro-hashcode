"""Sweeps: the same experiment repeated over detectors, faults and random seeds.

A sweep file (YAML) names a base config, optional overrides, a list of faults, a list of
detectors and a number of repeats, and optionally a list of group sizes (worker_counts; used by
the simulator sweeps). Every (fault, detector, group size, repeat) combination is one run.
Repeat r uses seed base_seed + r for every detector, so detectors are compared on the same data.

Each finished run appends one line to `runs.jsonl` in the results folder, so a sweep that is
stopped can be continued: runs already in the file are skipped.
"""

import copy
import json
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from swim_straggler.config import ConfigError, load_config
from swim_straggler.evaluation.metrics import run_metrics
from swim_straggler.telemetry.events import read_events


@dataclass
class SweepSpec:
    name: str
    base_config: str
    overrides: dict = field(default_factory=dict)
    faults: list = field(default_factory=list)  # each: {label: ..., and FaultConfig fields}
    detectors: list = field(default_factory=list)  # each: {label: ..., and DetectorConfig fields, optional only_faults}
    repeats: int = 3
    worker_counts: list = field(default_factory=list)  # group sizes to try; empty: the base config's world_size
    compare_with: str = ""  # results folder of a measured sweep to compare the simulation with (report only)
    keep_run_folders: bool = True  # False: delete each run's folder once its numbers are in runs.jsonl
    #                                (for the simulator: its runs are exactly reproducible from the seed, and large)


@dataclass
class PlannedRun:
    detector: str
    fault: str
    repeat: int
    config: object

    @property
    def key(self):
        return (self.detector, self.fault, self.config.cluster.world_size, self.repeat)


def load_sweep_spec(file_path):
    with open(file_path, encoding="utf-8") as spec_file:
        raw = yaml.safe_load(spec_file) or {}
    try:
        spec = SweepSpec(**raw)
    except TypeError as error:
        raise ConfigError(f"sweep file: {error}") from error
    if not spec.faults or not spec.detectors or spec.repeats < 1:
        raise ConfigError("a sweep needs at least one fault, one detector and repeats >= 1")
    return spec


def set_section_fields(section, values, section_name):
    for field_name, value in values.items():
        if not hasattr(section, field_name):
            raise ConfigError(f"sweep: {section_name} has no field '{field_name}'")
        if field_name == "mode" and value is False:
            value = "off"  # YAML reads an unquoted `off` as False
        setattr(section, field_name, value)


def plan_runs(spec, spec_directory=Path(".")):
    """Every run of the sweep, as configs ready to train. Invalid combinations raise ConfigError."""
    base_path = Path(spec.base_config)
    if not base_path.exists():
        base_path = Path(spec_directory) / spec.base_config
    base = load_config(base_path)
    for section_name, values in spec.overrides.items():
        if section_name in ("name", "run_directory"):
            setattr(base, section_name, values)
        else:
            set_section_fields(getattr(base, section_name), values, section_name)

    planned = []
    worker_counts = spec.worker_counts or [base.cluster.world_size]
    for entry in spec.faults + spec.detectors:
        entry["label"] = "off" if entry.get("label") is False else str(entry.get("label"))
    for fault in spec.faults:
        fault_fields = {key: value for key, value in fault.items() if key != "label"}
        for detector in spec.detectors:
            only_faults = detector.get("only_faults")
            if only_faults is not None and fault["label"] not in only_faults:
                continue
            detector_fields = {key: value for key, value in detector.items() if key not in ("label", "only_faults")}
            for workers, repeat in [(workers, repeat) for workers in worker_counts for repeat in range(spec.repeats)]:
                config = copy.deepcopy(base)
                config.cluster.world_size = workers
                config.fault = type(base.fault)()  # every fault starts from the defaults, not from the base config's fault
                set_section_fields(config.fault, fault_fields, "fault")
                set_section_fields(config.detector, detector_fields, "detector")
                if config.detector.mode == "swim":
                    config.detector.k_probes = min(config.detector.k_probes, config.cluster.world_size - 2)
                config.train.seed = base.train.seed + repeat
                size_part = f"_{workers}w" if spec.worker_counts else ""
                config.name = f"{spec.name}_{detector['label']}_{fault['label']}{size_part}_{repeat}".replace(" ", "_")
                config.validate()
                planned.append(PlannedRun(detector["label"], fault["label"], repeat, config))
    return planned


def read_finished_rows(results_path):
    if not Path(results_path).exists():
        return []
    with open(results_path, encoding="utf-8") as results_file:
        return [json.loads(line) for line in results_file if line.strip()]


def machine_description():
    """Where the numbers were measured. Timing results only mean something together with this."""
    import torch

    return {
        "processors": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%d"),
    }


def run_planned(planned, results_path=None, train=None, progress=print, keep_run_folders=True):
    """Run every planned run that is not in the results file yet. Returns all rows (old and new)."""
    import shutil

    from swim_straggler.train.launcher import run_training

    train = train or run_training
    rows = read_finished_rows(results_path) if results_path else []
    done = {(row["detector"], row["fault"], row["workers"], row["repeat"]) for row in rows}
    remaining = [run for run in planned if run.key not in done]
    for position, run in enumerate(remaining, start=1):
        progress(f"[{position}/{len(remaining)}] detector={run.detector} fault={run.fault} "
                 f"workers={run.config.cluster.world_size} repeat={run.repeat}")
        result = train(run.config)
        row = {"detector": run.detector, "fault": run.fault, "repeat": run.repeat, "seed": run.config.train.seed,
               "workers": run.config.cluster.world_size, "backend": run.config.cluster.backend,
               "run_directory": str(result.run_directory), **run_metrics(read_events(result.events_path))}
        if not keep_run_folders:
            shutil.rmtree(result.run_directory, ignore_errors=True)
            row["run_directory"] = None  # deleted; the same config and seed reproduce it exactly
        rows.append(row)
        if results_path:
            with open(results_path, "a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(row) + "\n")
    return rows


def run_sweep(spec_path, results_root="results", progress=print):
    """Run (or continue) a sweep. Returns (results folder, rows)."""
    spec = load_sweep_spec(spec_path)
    planned = plan_runs(spec, Path(spec_path).parent)
    results_directory = Path(results_root) / spec.name
    results_directory.mkdir(parents=True, exist_ok=True)
    machine_path = results_directory / "machine.json"
    if not machine_path.exists():
        machine_path.write_text(json.dumps(machine_description(), indent=2), encoding="utf-8")
    (results_directory / "sweep.yaml").write_text(Path(spec_path).read_text(encoding="utf-8"), encoding="utf-8")
    progress(f"sweep '{spec.name}': {len(planned)} runs planned, results in {results_directory}")
    rows = run_planned(planned, results_directory / "runs.jsonl", progress=progress,
                       keep_run_folders=spec.keep_run_folders)
    return results_directory, rows
