"""Event logging: one JSON object per line.

Each worker writes its own file (so processes never write to the same file), and the launcher
merges them into one `events.jsonl` when the run ends. Times in events are durations measured
on the worker's own clock, never timestamps that would be compared across processes.
"""

import json


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
    losses = [event["loss"] for event in rank_zero_events]
    first_losses = losses[:steps_to_average]
    last_losses = losses[-steps_to_average:]
    step_seconds = [event["step_seconds"] for event in step_events]
    return {
        "steps": len(rank_zero_events),
        "loss_at_start": sum(first_losses) / len(first_losses),
        "loss_at_end": sum(last_losses) / len(last_losses),
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
    }
