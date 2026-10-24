"""When a fault is active. Plain Python, so the simulator can use it without torch."""

from swim_straggler.config import CONTAINER_FAULT_KINDS

COMPUTE_FAULT_KINDS = ("compute", "two", "intermittent", "spike")


def fault_is_active(fault, rank, step):
    """True if worker `rank` is supposed to be slow during `step` (container faults: never, see the controller)."""
    if fault.kind == "none" or fault.kind in CONTAINER_FAULT_KINDS or rank not in fault.workers:
        return False  # container faults are applied from outside by the fault controller
    if step < fault.start_step:
        return False
    if fault.duration > 0 and step >= fault.start_step + fault.duration:
        return False
    if fault.kind == "intermittent":
        steps_since_start = step - fault.start_step
        return (steps_since_start // fault.period) % 2 == 0  # on, off, on, off, ...
    return True
