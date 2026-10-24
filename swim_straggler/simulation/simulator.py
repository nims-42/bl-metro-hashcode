"""A discrete-event simulator of synchronous ring all-reduce training, for groups too large to run.

SIMULATION, NOT MEASUREMENT. Nothing is trained and nothing is timed: the compute and message
times come from a random model (the `simulation` section of the config). What is real is the
detector: every worker runs the unchanged detector code (SWIM, central monitor or step time), it
receives exactly the waits, compute times and gossip it would receive in a real run, and its
decisions are logged in the same event format as a real run. So all the summaries, sweeps, the
dashboard and trace.log work on simulated runs too.

One step of worker i in the model:
  1. it starts when its previous all-reduce finished, and computes for
         compute_ms x (random factor, spread compute_noise) x (fault multiplier, if slow);
  2. the ring all-reduce has 2 x (N - 1) rounds. In every round each worker sends to its successor
     as soon as it is ready; the message arrives after
         message_latency_ms x (random factor, spread latency_noise) + allreduce_data_ms / N
     (plus the link delay of a link fault). The receiver waits if the message is not there yet;
  3. every round, every worker's gossip goes to its successor, in the same order as in the real
     ring (send what you know, then absorb what arrived).
The random factors are log-normal: never negative, occasionally much larger than typical, like
real timings. Consecutive steps are independent here; real noise often comes in bursts.
"""

import json
import math
import time
from pathlib import Path

import numpy

from swim_straggler.detector.factory import create_detector
from swim_straggler.faults.schedule import COMPUTE_FAULT_KINDS, fault_is_active


class SimulatedLog:
    """Collects events in memory and writes them in the order merge_event_files would produce."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.first = []
        self.by_step = {}
        self.last = []

    def add(self, at_step, rank, event_name, fields):
        record = {"event": event_name, "rank": rank, **fields}
        self.by_step.setdefault(at_step, [[] for _ in range(self.world_size)])[rank].append(record)

    def write(self, file_path):
        with open(file_path, "w", encoding="utf-8") as events_file:
            for record in self.first:
                events_file.write(json.dumps(record) + "\n")
            for step in sorted(self.by_step):
                for rank_records in self.by_step[step]:  # within a step: by rank, each worker in its own order
                    for record in rank_records:
                        events_file.write(json.dumps(record) + "\n")
            for record in self.last:
                events_file.write(json.dumps(record) + "\n")


def lognormal_factors(generator, spread, count):
    """Random factors with median 1. spread 0 gives exactly 1."""
    if spread <= 0:
        return numpy.ones(count)
    return generator.lognormal(mean=0.0, sigma=spread, size=count)


def simulate_run(config, run_directory):
    """Simulate config.train.steps steps of config.cluster.world_size workers. Returns the events.jsonl path."""
    world_size = config.cluster.world_size
    model = config.simulation
    fault = config.fault
    generator = numpy.random.default_rng(config.train.seed)
    detectors = [create_detector(config, rank, world_size) for rank in range(world_size)]
    uses_gossip = detectors[0].gossip_length > 0
    rounds = 2 * (world_size - 1)
    log_round_waits = world_size <= model.round_waits_up_to_workers
    log = SimulatedLog(world_size)
    log.first.append({"event": "run_start", "rank": 0, "config": config.to_dict(), "dataset_source": "simulated",
                      "world_size": world_size, "simulated": True})

    compute_seconds_typical = model.compute_ms / 1000
    latency_seconds = model.message_latency_ms / 1000
    data_seconds_per_round = model.allreduce_data_ms / 1000 / world_size
    finish_time = numpy.zeros(world_size)
    fault_start_logged = set()
    wall_clock_start = time.perf_counter()

    for step in range(config.train.steps):
        active = numpy.array([fault_is_active(fault, rank, step) for rank in range(world_size)])
        compute = compute_seconds_typical * lognormal_factors(generator, model.compute_noise, world_size)
        healthy_compute = compute.copy()
        if fault.kind in COMPUTE_FAULT_KINDS:
            compute[active] *= fault.multiplier
        send_delay = numpy.where(active, fault.link_delay_seconds, 0.0) if fault.kind == "link" else numpy.zeros(world_size)
        for rank in numpy.flatnonzero(active):
            if rank not in fault_start_logged:
                fault_start_logged.add(rank)
                log.add(step, int(rank), "fault_started", {"step": step, "fault_kind": fault.kind})

        for rank, detector in enumerate(detectors):
            detector.start_step(step, float(compute[rank]))

        # the ring all-reduce, round by round
        start_time = finish_time
        ready_time = start_time + compute
        waits = numpy.empty((rounds, world_size))
        for round_number in range(rounds):
            message_cost = latency_seconds * lognormal_factors(generator, model.latency_noise, world_size)
            sent_arrives = ready_time + send_delay + message_cost + data_seconds_per_round
            arrival_time = numpy.roll(sent_arrives, 1)  # worker i receives what worker i - 1 sent
            waits[round_number] = numpy.maximum(0.0, arrival_time - ready_time)
            ready_time = numpy.maximum(ready_time, arrival_time)
            if uses_gossip:
                messages = [detector.gossip_contents() for detector in detectors]
                for sender, (rows, notices) in enumerate(messages):
                    detectors[(sender + 1) % world_size].absorb_gossip(rows, notices, sender=sender)
        finish_time = ready_time
        step_seconds = finish_time - start_time

        compute_list = compute.tolist()
        for rank, detector in enumerate(detectors):
            wait_list = waits[:, rank].tolist()
            for event in detector.finish_step(step, wait_list, compute_list if detector.needs_compute_reports else None):
                log.add(step, rank, "trace" if event.pop("trace", False) else "detector", event)
            step_event = {
                "step": step, "loss": None, "compute_seconds": compute_list[rank], "wait_seconds": sum(wait_list),
                "step_seconds": float(step_seconds[rank]), "fault_active": bool(active[rank]),
            }
            if log_round_waits:  # the details make the log much larger, so only for small groups
                step_event["wait_seconds_per_round"] = wait_list
                step_event["allreduce_seconds"] = float(finish_time[rank] - start_time[rank] - compute[rank])
                step_event["injected_compute_seconds"] = float(compute[rank] - healthy_compute[rank])
            log.add(step, rank, "step", step_event)
            for event in detector.after_update(step, float(step_seconds[rank])):
                log.add(step, rank, "trace" if event.pop("trace", False) else "detector", event)

    wall_seconds = time.perf_counter() - wall_clock_start
    for rank, detector in enumerate(detectors):
        log.last.append({"event": "run_end", "rank": rank, "parameter_checksum": None,
                         "believed_stragglers": detector.straggler_ranks(),
                         "simulated_seconds": float(finish_time[rank]), "simulator_wall_seconds": wall_seconds})
    events_path = Path(run_directory) / "events.jsonl"
    log.write(events_path)
    return events_path


def fit_timing_model(events, warmup_steps=10):
    """Simulation parameters fitted to the healthy steps of a measured run (steps without an active fault).

    compute_ms and compute_noise: median and spread (standard deviation of the logarithm) of the compute times.
    message_latency_ms: median wait in the all-gather rounds, where a balanced ring waits about one message latency.
    allreduce_data_ms: N x the median time per round that the all-reduce was busy rather than waiting.
    Time outside compute and all-reduce (the optimizer update, logging) is left out: it is the same for every
    worker and does not change who waits for whom.
    These are rough: the measured run's noise includes everything that happened on that machine.
    """
    import statistics

    run_start = next(event for event in events if event["event"] == "run_start")
    world_size = run_start["world_size"]
    steps = [event for event in events if event["event"] == "step" and event["step"] >= warmup_steps
             and not event.get("fault_active") and event.get("compute_seconds", 0) > 0]
    if len(steps) < 20 or world_size < 2:
        raise ValueError("the run needs at least 20 healthy steps after the warmup to fit the timing model")
    compute_ms = [event["compute_seconds"] * 1000 for event in steps]
    logarithms = [math.log(value) for value in compute_ms]
    all_gather_waits = [wait * 1000 for event in steps for wait in event.get("wait_seconds_per_round", [])[world_size - 1:]]
    latency_ms = statistics.median(all_gather_waits) if all_gather_waits else 0.5
    rounds = 2 * (world_size - 1)
    busy_per_round_ms = statistics.median(
        max(0.0, event["allreduce_seconds"] - event["wait_seconds"]) * 1000 / rounds for event in steps
        if "allreduce_seconds" in event) if all("allreduce_seconds" in event for event in steps) else 0.0
    return {
        "compute_ms": round(statistics.median(compute_ms), 2),
        "compute_noise": round(statistics.pstdev(logarithms), 3),
        "message_latency_ms": round(latency_ms, 2),
        "latency_noise": 0.5,  # not fitted: the waits mix latency noise with compute noise
        "allreduce_data_ms": round(busy_per_round_ms * world_size, 2),
    }
