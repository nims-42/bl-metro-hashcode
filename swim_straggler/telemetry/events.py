"""Event logging: one JSON object per line.

Each worker writes its own file (so processes never write to the same file), and the launcher
merges them into one `events.jsonl` when the run ends. Times in events are durations measured
on the worker's own clock, never timestamps that would be compared across processes.
"""

import json
import statistics


class EventLogger:
    def __init__(self, file_path, rank):
        self.rank = rank
        self._file = open(file_path, "w", encoding="utf-8")

    def log(self, event_name, **fields):
        record = {"event": event_name, "rank": self.rank}
        record.update(fields)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()


def read_events(file_path):
    with open(file_path, encoding="utf-8") as events_file:
        return [json.loads(line) for line in events_file if line.strip()]


def sort_key(record):
    """Run start first, run end last, everything else by step and then by rank."""
    if record["event"] == "run_start":
        return (0, 0, record["rank"])
    if record["event"] == "run_end":
        return (2, 0, record["rank"])
    return (1, record.get("step", 0), record["rank"])


def merge_event_files(rank_file_paths, merged_file_path):
    records = []
    for rank_file_path in rank_file_paths:
        records.extend(read_events(rank_file_path))
    records.sort(key=sort_key)
    with open(merged_file_path, "w", encoding="utf-8") as merged_file:
        for record in records:
            merged_file.write(json.dumps(record) + "\n")


def summarize_training(events, steps_to_average=5):
    """Loss at the start and end of training, and the mean step time, from the step events."""
    step_events = [event for event in events if event["event"] == "step"]
    rank_zero_events = sorted((event for event in step_events if event["rank"] == 0), key=lambda event: event["step"])
    losses = [event["loss"] for event in rank_zero_events if event.get("loss") is not None]  # none in simulated runs
    first_losses = losses[:steps_to_average]
    last_losses = losses[-steps_to_average:]
    step_seconds = [event["step_seconds"] for event in step_events]
    return {
        "steps": len(rank_zero_events),
        "loss_at_start": sum(first_losses) / len(first_losses) if losses else None,
        "loss_at_end": sum(last_losses) / len(last_losses) if losses else None,
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
    }


def observed_compute_ratio(events, rank, issued_step, applied_step, ended_step, settle_steps=3, warmup_steps=10):
    """How much longer the worker really computed during the fault than before it (median over steps).

    This checks that an injected fault had an effect at all: a throttle that never binds leaves the
    ratio near 1.0, and then there is nothing for a detector to find. None if there are too few steps.
    """
    steps = {event["step"]: event["compute_seconds"] for event in events if event["event"] == "step" and event["rank"] == rank}
    before = [seconds for step, seconds in steps.items() if warmup_steps <= step < issued_step]
    during = [seconds for step, seconds in steps.items()
              if step >= applied_step + settle_steps and (ended_step is None or step < ended_step)]
    if len(before) < 5 or len(during) < 5:
        return None
    return statistics.median(during) / statistics.median(before)


def summarize_detection(events):
    """What the detectors concluded, compared with the faults that were really injected.

    The ground truth is the `fault_started` events. Steps are step numbers, not times.
    Fields are None when there was nothing to report (for example no confirmation).
    """
    detector_events = [event for event in events if event["event"] == "detector"]
    faulty_ranks = {event["rank"] for event in events if event["event"] == "fault_started"}
    fault_start_steps = [event["step"] for event in events if event["event"] == "fault_started"]
    fault_start_step = min(fault_start_steps) if fault_start_steps else None
    all_ranks = sorted({event["rank"] for event in events if event["event"] == "step"})

    issued_step_of = {}  # faulty worker -> step its fault was started (issued)
    for event in events:
        if event["event"] == "fault_started":
            issued_step_of.setdefault(event["rank"], event["step"])

    def is_wrong(event):
        """An accusation of a worker that was healthy at that moment (never faulty, or not faulty yet)."""
        return event["subject"] not in issued_step_of or event["step"] < issued_step_of[event["subject"]]

    def first_step(actions, subject):
        steps = [event["step"] for event in detector_events
                 if event["action"] in actions and event["subject"] == subject and not is_wrong(event)]
        return min(steps) if steps else None

    run_start = [event for event in events if event["event"] == "run_start"]
    detector_mode = run_start[0]["config"]["detector"]["mode"] if run_start else None
    summary = {
        "detector_mode": detector_mode,
        "faulty_workers": sorted(faulty_ranks),
        "fault_kinds": {event["rank"]: event.get("fault_kind") for event in events if event["event"] == "fault_started"},
        "fault_start_step": fault_start_step,
        "per_fault": {},
        # A false suspicion is a healthy worker that someone suspected; the vote normally rejects it.
        # A false confirmation is a healthy worker that someone declared slow. That one is the real error.
        # Accusing the later culprit BEFORE its fault started also counts as wrong.
        "false_suspicions": len({(event["rank"], event["subject"], event["step"]) for event in detector_events
                                 if event["action"] == "suspected" and is_wrong(event)}),
        "false_confirmations": sorted({(event["rank"], event["subject"], event["step"]) for event in detector_events
                                       if event["action"] == "confirmed" and is_wrong(event)}),
        "rejected_suspicions": len([event for event in detector_events if event["action"] == "rejected"]),
        "cleared_events": len([event for event in detector_events if event["action"] == "cleared"]),
        # EXPERIMENTAL: workers that said "the delay is not compute" (detector.diagnose_network), with the first step
        "network_diagnoses": sorted({event["rank"]: event["step"] for event in reversed(detector_events)
                                     if event["action"] == "delay_not_compute"}.items()),
    }
    for culprit in sorted(faulty_ranks):
        first_confirmed = first_step(("confirmed",), culprit)
        started = [event for event in events if event["event"] == "fault_started" and event["rank"] == culprit][0]
        ended = [event["step"] for event in events if event["event"] == "fault_ended" and event["rank"] == culprit]
        # Container faults take time to apply: `step` is when the command was issued, `step_after_command`
        # when it returned. Worker-side faults take effect immediately, so both are the same.
        issued_step = started["step"]
        applied_step = started.get("step_after_command", issued_step)
        compute_ratio = observed_compute_ratio(events, culprit, issued_step, applied_step, ended[0] if ended else None)
        step_each_worker_knew = {}
        for rank in all_ranks:
            known_at = [event["step"] for event in detector_events
                        if event["rank"] == rank and event["subject"] == culprit and not is_wrong(event)
                        and event["action"] in ("confirmed", "learned_straggler")]
            if known_at:
                step_each_worker_knew[rank] = min(known_at)
        everybody_knew = len(step_each_worker_knew) == len(all_ranks)
        summary["per_fault"][culprit] = {
            "first_suspected_step": first_step(("suspected",), culprit),
            "first_confirmed_step": first_confirmed,
            "delay_steps": None if first_confirmed is None else first_confirmed - issued_step,
            "delay_steps_after_applied": None if first_confirmed is None else first_confirmed - applied_step,
            "fault_issued_step": issued_step,
            "fault_applied_step": applied_step,
            "compute_ratio": compute_ratio,
            "all_workers_knew_at_step": max(step_each_worker_knew.values()) if everybody_knew else None,
            "workers_that_never_learned": [rank for rank in all_ranks if rank not in step_each_worker_knew],
        }
    return summary


def comparison_row(mode, events):
    """One line of the detector comparison table, from the events of one run."""
    summary = summarize_detection(events)
    training = summarize_training(events)
    culprit_reports = list(summary["per_fault"].values())
    confirmed = [report for report in culprit_reports if report["first_confirmed_step"] is not None]
    delays = [report["delay_steps"] for report in culprit_reports if report["delay_steps"] is not None]
    applied_delays = [report["delay_steps_after_applied"] for report in culprit_reports
                      if report["delay_steps_after_applied"] is not None]
    return {
        "mode": mode,
        "culprits found": f"{len(confirmed)} of {len(culprit_reports)}",
        "delay (steps)": ", ".join(str(delay) for delay in delays) or "-",
        "delay after fault applied (steps)": ", ".join(str(delay) for delay in applied_delays) or "-",
        "healthy wrongly confirmed": len(summary["false_confirmations"]),
        "healthy wrongly suspected": summary["false_suspicions"],
        "mean step (ms)": f"{training['mean_step_seconds'] * 1000:.1f}",
    }
