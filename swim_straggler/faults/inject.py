"""Fault injection: make chosen workers slow in a controlled, known way.

Because we inject the faults ourselves, every run has ground truth: we know which worker
was slow and from which step, so detection accuracy and latency can be measured honestly.

Compute faults ("compute", "two", "intermittent", "spike") add extra time to a worker's
compute phase, scaled by how long the real computation took. A multiplier of 2.0 therefore
doubles the worker's compute time, whatever the machine's speed.

Link faults delay every message the worker sends, while its compute stays normal.
"""

import time

from swim_straggler.faults.schedule import COMPUTE_FAULT_KINDS, fault_is_active
from swim_straggler.transport.base import Transport


class FaultInjector:
    def __init__(self, fault_config, rank):
        self._fault = fault_config
        self._rank = rank

    def is_fault_active(self, step):
        """True if this worker is supposed to be slow during the given step."""
        return fault_is_active(self._fault, self._rank, step)

    def extra_compute_seconds(self, step, measured_compute_seconds):
        if self._fault.kind in COMPUTE_FAULT_KINDS and self.is_fault_active(step):
            return (self._fault.multiplier - 1.0) * measured_compute_seconds
        return 0.0

    def send_delay_seconds(self, step):
        if self._fault.kind == "link" and self.is_fault_active(step):
            return self._fault.link_delay_seconds
        return 0.0

    def slow_down_compute(self, step, measured_compute_seconds):
        """Wait for the extra time (if any). Returns how many seconds were added."""
        extra_seconds = self.extra_compute_seconds(step, measured_compute_seconds)
        if extra_seconds <= 0.0:
            return 0.0
        if self._fault.slowdown_style == "sleep":
            time.sleep(extra_seconds)  # the processor stays free for other workers
        else:
            finish_time = time.perf_counter() + extra_seconds
            while time.perf_counter() < finish_time:
                pass  # keep the processor busy, like a throttled chip would
        return extra_seconds


class SlowLinkTransport(Transport):
    """Wraps a transport and delays every send while a link fault is active."""

    def __init__(self, inner_transport, fault_injector):
        self.rank = inner_transport.rank
        self.world_size = inner_transport.world_size
        self.current_step = 0  # the training loop updates this before each all-reduce
        self._inner_transport = inner_transport
        self._fault_injector = fault_injector

    def send_to_next(self, payload):
        delay_seconds = self._fault_injector.send_delay_seconds(self.current_step)
        if delay_seconds > 0.0:
            time.sleep(delay_seconds)
        self._inner_transport.send_to_next(payload)

    def receive_from_previous(self, number_of_elements):
        return self._inner_transport.receive_from_previous(number_of_elements)

    def flush(self):
        self._inner_transport.flush()

    def close(self):
        self._inner_transport.close()
