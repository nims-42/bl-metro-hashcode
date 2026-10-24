"""The fault controller: applies a container fault when training reaches a chosen step.

It watches the event file of worker 0 (which the container writes into a folder shared with
this computer) to learn which step training is at, applies the fault to the listed workers
with real tools, and writes the ground truth to `events_controller.jsonl`.

Accuracy of the ground truth: the fault takes effect somewhere between the step seen before the
command and the step seen after it returned. We record the earlier one as `step`, so measured
detection delays are never too optimistic, and the later one as `step_after_command`.
"""

import json
import threading
import time
from pathlib import Path

from swim_straggler.containers.commands import build_fault_action


def latest_step(events_path):
    """The highest step worker 0 has finished, or -1. A half-written last line is ignored."""
    try:
        text = Path(events_path).read_text(encoding="utf-8")
    except OSError:
        return -1
    for line in reversed(text.splitlines()):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "step":
            return record["step"]
    return -1


class FaultController:
    def __init__(self, config, container_names, run_directory, command_runner, poll_seconds=0.02):
        self.config = config
        self.container_names = container_names  # container name by worker rank
        self.worker_events_path = Path(run_directory) / "events_rank_0.jsonl"
        self.controller_events_path = Path(run_directory) / "events_controller.jsonl"
        self.runner = command_runner
        self.poll_seconds = poll_seconds
        self.fault_is_applied = False
        self.fault_has_ended = False
        self.applied_at = None
        self.stop_requested = threading.Event()
        self.error = None

    def write_event(self, event_name, worker, **fields):
        record = {"event": event_name, "rank": worker, "source": "controller", **fields}
        with open(self.controller_events_path, "a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(record) + "\n")

    def actions(self):
        return {worker: build_fault_action(self.config.fault, self.config.cluster, self.container_names[worker])
                for worker in self.config.fault.workers}

    def apply_fault(self, step_before):
        for worker, action in self.actions().items():
            for command in action.start:
                self.runner.run(command)
            self.write_event("fault_started", worker, step=step_before,
                             step_after_command=latest_step(self.worker_events_path),
                             fault_kind=self.config.fault.kind, tool=self.config.fault.fault_tool)
        self.fault_is_applied = True
        self.applied_at = time.monotonic()

    def remove_fault(self, step_before):
        for worker, action in self.actions().items():
            for command in action.stop:
                self.runner.run(command, check=False)  # removing must not stop the clean-up of the others
            self.write_event("fault_ended", worker, step=step_before, fault_kind=self.config.fault.kind)
        self.fault_has_ended = True

    def fault_is_over(self, step):
        fault = self.config.fault
        if fault.kind == "container_pause":
            # A frozen worker stops every step, so the end cannot be counted in steps.
            return time.monotonic() - self.applied_at >= fault.pause_seconds
        return fault.duration > 0 and step >= fault.start_step + fault.duration

    def run(self):
        """Watch training until `stop_requested` is set. Always removes an applied fault before returning."""
        fault = self.config.fault
        try:
            while not self.stop_requested.is_set():
                step = latest_step(self.worker_events_path)
                if not self.fault_is_applied and step >= fault.start_step:
                    self.apply_fault(step)
                elif self.fault_is_applied and not self.fault_has_ended and self.fault_is_over(step):
                    self.remove_fault(step)
                self.stop_requested.wait(self.poll_seconds)
        except Exception as error:  # reported by the runner, which owns the thread
            self.error = error
        finally:
            if self.fault_is_applied and not self.fault_has_ended:
                self.remove_fault(latest_step(self.worker_events_path))
