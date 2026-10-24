"""Everything the dashboard shows, computed from a run's event log. No Streamlit here, so it can be tested.

A run that is still going has no merged `events.jsonl` yet; then the per-worker files are read
directly, and a half-written last line (a worker is writing it right now) is skipped.
"""

import json
from pathlib import Path

from swim_straggler.detector.table import ALIVE, STRAGGLER, SUSPECT
from swim_straggler.telemetry.events import comparison_row, sort_key

STATE_AFTER_ACTION = {
    "suspected": SUSPECT,
    "confirmed": STRAGGLER,
    "learned_straggler": STRAGGLER,
    "rejected": ALIVE,
    "cleared": ALIVE,
    "learned_cleared": ALIVE,
}


def read_events_tolerantly(file_path):
    events = []
    with open(file_path, encoding="utf-8") as events_file:
        for line in events_file:
            try:
                events.append(json.loads(line))
            except ValueError:
                continue  # a line that is still being written
    return events


def list_runs(runs_directory):
    """Run folders (those holding any event file), newest first, as (name, path) pairs."""
    runs_directory = Path(runs_directory)
    if not runs_directory.is_dir():
        return []
    runs = [folder for folder in runs_directory.iterdir()
            if folder.is_dir() and (any(folder.glob("events_rank_*.jsonl")) or (folder / "events.jsonl").exists())]
    runs.sort(key=lambda folder: folder.stat().st_mtime, reverse=True)
    return [(folder.name, folder) for folder in runs]


def load_run_events(run_directory):
    """All events of a run, finished or still running. Returns (events, is_finished)."""
    run_directory = Path(run_directory)
    merged_path = run_directory / "events.jsonl"
    if merged_path.exists():
        return read_events_tolerantly(merged_path), True
    events = []
    for file_path in sorted(run_directory.glob("events_rank_*.jsonl")) + sorted(run_directory.glob("events_controller.jsonl")):
        events += read_events_tolerantly(file_path)
    events.sort(key=sort_key)
    return events, False


def run_overview(events):
    """The settings that matter for reading the charts."""
    run_start = next((event for event in events if event["event"] == "run_start"), None)
    if run_start is None:
        return {}
    config = run_start["config"]
    return {
        "name": config["name"],
        "workers": run_start["world_size"],
        "backend": config["cluster"]["backend"],
        "data": run_start["dataset_source"],
        "steps": config["train"]["steps"],
        "fault": config["fault"]["kind"],
        "faulty workers": config["fault"]["workers"],
        "fault from step": config["fault"]["start_step"],
        "detector": config["detector"]["mode"],
    }


def world_size_of(events):
    overview = run_overview(events)
    if overview:
        return overview["workers"]
    return 1 + max((event["rank"] for event in events if event["event"] == "step"), default=-1)


def step_rows(events):
    """One row per worker per step, times in milliseconds (easier to read than seconds)."""
    rows = []
    for event in events:
        if event["event"] != "step":
            continue
        rows.append({
            "step": event["step"],
            "worker": event["rank"],
            "compute (ms)": event["compute_seconds"] * 1000,
            "wait (ms)": event["wait_seconds"] * 1000,
            "step (ms)": event["step_seconds"] * 1000,
            "loss": event["loss"],
            "fault active": event.get("fault_active", False),
        })
    return rows


def latest_step(events):
    return max((event["step"] for event in events if event["event"] == "step"), default=-1)


def round_wait_rows(events, step):
    """How long each worker waited in each ring round of one step (the detector's raw signal)."""
    rows = []
    for event in events:
        if event["event"] == "step" and event["step"] == step:
            for round_number, seconds in enumerate(event.get("wait_seconds_per_round", [])):
                rows.append({"worker": event["rank"], "round": round_number, "wait (ms)": seconds * 1000})
    return rows


def detector_event_rows(events):
    """The detectors' decisions as a table, in step order."""
    rows = []
    for event in events:
        if event["event"] != "detector":
            continue
        details = {key: value for key, value in event.items() if key not in ("event", "rank", "step", "action", "subject")}
        rows.append({"step": event["step"], "worker": event["rank"], "action": event["action"],
                     "about worker": event["subject"],
                     "details": ", ".join(f"{key}={value}" for key, value in details.items())})
    return rows


def belief_rows(events, step, world_size):
    """What each worker believed about each worker at the end of `step`: alive, suspect or straggler."""
    beliefs = [[ALIVE] * world_size for _ in range(world_size)]
    for event in events:
        if event["event"] == "detector" and event["step"] <= step and event["action"] in STATE_AFTER_ACTION:
            beliefs[event["rank"]][event["subject"]] = STATE_AFTER_ACTION[event["action"]]
    return [{"worker (believer)": believer, "about worker": subject, "belief": beliefs[believer][subject]}
            for believer in range(world_size) for subject in range(world_size)]


def chart_markers(events):
    """Vertical lines for the charts: when faults started, and when each was first confirmed."""
    markers = []
    for event in events:
        if event["event"] == "fault_started":
            markers.append({"step": event["step"], "label": f"fault on worker {event['rank']}"})
        if event["event"] == "fault_ended":
            markers.append({"step": event["step"], "label": f"fault on worker {event['rank']} ended"})
    first_confirmed = {}
    for event in events:
        if event["event"] == "detector" and event["action"] == "confirmed":
            first_confirmed.setdefault(event["subject"], event["step"])
    for subject, step in first_confirmed.items():
        markers.append({"step": step, "label": f"worker {subject} confirmed slow"})
    return markers


COMPARED_MODES = ("swim", "central", "steptime")
MOST_RUNS_TO_SEARCH = 40


def same_experiment_key(events):
    """What makes two runs comparable: same cluster, length and fault (the detector may differ)."""
    run_start = next((event for event in events if event["event"] == "run_start"), None)
    if run_start is None:
        return None
    config = run_start["config"]
    fault = config["fault"]
    return (run_start["world_size"], config["cluster"]["backend"], config["train"]["steps"], config["train"]["seed"],
            fault["kind"], tuple(fault["workers"]), fault["start_step"], fault["multiplier"], fault["duration"])


def detector_mode_of(events):
    run_start = next((event for event in events if event["event"] == "run_start"), None)
    return run_start["config"]["detector"]["mode"] if run_start else None


def comparison_with_other_detectors(runs_directory, chosen_events):
    """The newest finished run of each detector on the same experiment as the chosen run.

    Returns (rows, missing_modes). It only looks at runs that already exist; it never starts training.
    """
    wanted_key = same_experiment_key(chosen_events)
    newest_by_mode = {}
    for name, folder in list_runs(runs_directory)[:MOST_RUNS_TO_SEARCH]:
        events, is_finished = load_run_events(folder)
        mode = detector_mode_of(events)
        if not is_finished or mode not in COMPARED_MODES or mode in newest_by_mode:
            continue
        if same_experiment_key(events) == wanted_key:
            newest_by_mode[mode] = (name, events)  # runs are newest first, so the first one found is the newest
    rows = [{"run": newest_by_mode[mode][0], **comparison_row(mode, newest_by_mode[mode][1])}
            for mode in COMPARED_MODES if mode in newest_by_mode]
    missing = [mode for mode in COMPARED_MODES if mode not in newest_by_mode]
    return rows, missing
