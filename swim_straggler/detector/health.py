"""The local health check: should this worker trust its own waiting times?

Lifeguard (the SWIM extension) observed that a member that is itself slow produces false
accusations about everybody else. Our simplified version: a worker whose own compute time is
well above the typical one does not accuse anybody.
"""


def is_locally_healthy(own_compute_seconds, typical_compute_seconds, health_factor):
    if typical_compute_seconds is None or typical_compute_seconds <= 0:
        return True  # not enough information yet to say otherwise
    return own_compute_seconds <= health_factor * typical_compute_seconds
