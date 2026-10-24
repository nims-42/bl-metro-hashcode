"""The numbers measured from one run, and their summary over many runs. Plain Python, no plotting."""

import statistics

from swim_straggler.telemetry.events import summarize_detection, summarize_training


def run_metrics(events):
    """One row of numbers for one finished run."""
    detection = summarize_detection(events)
    training = summarize_training(events)
    reports = list(detection["per_fault"].values())

    def mean_or_none(values):
        values = [value for value in values if value is not None]
        return statistics.mean(values) if values else None

    knew_delays = [None if report["all_workers_knew_at_step"] is None
                   else report["all_workers_knew_at_step"] - report["fault_issued_step"] for report in reports]
    return {
        "faulty_workers": len(reports),
        # True only if EVERY faulty worker was confirmed after its fault started; None for healthy runs.
        "culprit_found": all(report["first_confirmed_step"] is not None for report in reports) if reports else None,
        "delay_steps": mean_or_none(report["delay_steps"] for report in reports),
        "delay_after_applied_steps": mean_or_none(report["delay_steps_after_applied"] for report in reports),
        "everyone_knew_delay_steps": mean_or_none(knew_delays),
        "compute_ratio": mean_or_none(report["compute_ratio"] for report in reports),
        "false_confirmations": len(detection["false_confirmations"]),
        "false_suspicions": detection["false_suspicions"],
        "rejected_suspicions": detection["rejected_suspicions"],
        "mean_step_ms": training["mean_step_seconds"] * 1000,
        "loss_at_end": training["loss_at_end"],
        "steps": training["steps"],
    }


def median_or_none(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarize_group(rows):
    """Summary of the repeated runs of one (detector, fault) pair."""
    is_healthy = all(row["faulty_workers"] == 0 for row in rows)
    found = [row for row in rows if row["culprit_found"]]
    delays = [row["delay_steps"] for row in found]
    applied_delays = [row["delay_after_applied_steps"] for row in found]
    return {
        "runs": len(rows),
        "found": None if is_healthy else len(found),
        "median_delay_steps": median_or_none(delays),
        "min_delay_steps": min(delays) if delays else None,
        "max_delay_steps": max(delays) if delays else None,
        "median_delay_after_applied_steps": median_or_none(applied_delays),
        "median_everyone_knew_delay_steps": median_or_none(row["everyone_knew_delay_steps"] for row in found),
        "runs_with_false_confirmation": sum(1 for row in rows if row["false_confirmations"] > 0),
        "false_confirmations": sum(row["false_confirmations"] for row in rows),
        "false_suspicions": sum(row["false_suspicions"] for row in rows),
        "median_compute_ratio": median_or_none(row["compute_ratio"] for row in rows),
        "mean_step_ms": statistics.mean(row["mean_step_ms"] for row in rows),
    }


def summarize_sweep(rows):
    """One summary per (detector, fault) pair, in the order they first appear."""
    groups = {}
    for row in rows:
        groups.setdefault((row["detector"], row["fault"]), []).append(row)
    return [{"detector": detector, "fault": fault, **summarize_group(group_rows)}
            for (detector, fault), group_rows in groups.items()]


def step_time_overhead(summaries, baseline_detector="off", fault="healthy"):
    """Mean step time of each detector on healthy runs, relative to the detector switched off."""
    healthy = {summary["detector"]: summary for summary in summaries if summary["fault"] == fault}
    if baseline_detector not in healthy:
        return []
    baseline = healthy[baseline_detector]["mean_step_ms"]
    return [{"detector": detector, "mean_step_ms": summary["mean_step_ms"],
             "overhead_percent": 100 * (summary["mean_step_ms"] / baseline - 1)}
            for detector, summary in healthy.items()]
