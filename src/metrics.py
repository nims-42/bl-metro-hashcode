"""Per-node counters, timings and events, exported as JSON lines (CONTEXT.md 3.2 L7, 9).

One object per training step goes to <results_dir>/<run_id>/<node_id>.jsonl and
is flushed immediately, so a crash loses at most one step. Detection code may
write here but never reads the exported data back.

Durations come from time.monotonic() on this node; `ts` is wall clock and is
only used to line runs up with ground truth afterwards (6.3a).
"""

import json
import os
import threading
import time

COUNTERS = ("msgs_sent", "msgs_recv", "ping_req_sent", "ping_req_recv")


class Metrics:
    def __init__(self, node_id, n, run_id, results_dir, wall=time.time, enabled=True):
        self.node_id = node_id
        self.n = n
        self.run_id = run_id
        self._wall = wall
        self._lock = threading.Lock()
        self._counters = dict.fromkeys(COUNTERS, 0)
        self._events = []
        self._file = None
        if enabled:
            run_dir = os.path.join(results_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            self.path = os.path.join(run_dir, f"{node_id}.jsonl")
            self._file = open(self.path, "w", encoding="utf-8")

    def count(self, name, amount=1):
        with self._lock:
            self._counters[name] += amount

    def counters(self):
        with self._lock:
            return dict(self._counters)

    def event(self, type_, about, cls=None, from_=None, **extra):
        record = {"type": type_, "about": about, "class": cls, "from": from_,
                  "ts": self._wall(), **extra}
        with self._lock:
            self._events.append(record)
        return record

    def write_step(self, step, compute_t, wait_t, pred_lag, step_t, lhm, member_states, **extra):
        """Write one step record and reset the per-step event list.
        Counters are cumulative over the run."""
        with self._lock:
            record = {
                "node_id": self.node_id, "n": self.n, "step": step, "ts": self._wall(),
                "compute_t": compute_t, "wait_t": wait_t, "pred_lag": pred_lag,
                "step_t": step_t, "lhm": lhm, **self._counters,
                "member_states": member_states, "events": self._events, **extra,
            }
            self._events = []
            if self._file:
                self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
                self._file.flush()
        return record

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
