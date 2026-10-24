"""Local health awareness and suspicion logic (CONTEXT.md 3.3, 6.2, 6.3, 6.5).

Lifeguard pieces mapped onto straggler detection:

* LHM: a saturating counter 0..S that stretches this node's probe timeout,
  probe interval and the number of late steps needed before it accuses.
* Self-check: a node that is itself the ring outlier does not accuse anyone.
* LHA-Suspicion: each suspicion starts at Max and shrinks towards Min as
  independent confirmations arrive.
* Buddy System: gossip about a suspected node is sent to it first.
* Classification (FALCON): slow compute_t -> "compute", otherwise "network".

The "step stats" passed around here are dicts of per-node values for one
step: {"compute_t": {node: s}, "wait_t": {node: s}, "pred_lag": {node: s}}.
"""

import math
import statistics
import time
from dataclasses import dataclass, field

from src import membership

COMPUTE = "compute"
NETWORK = "network"


def exceeds(value, median, factor, floor):
    """FALCON's rule: above `factor` x median, and by at least `floor`
    seconds. The floor is the 10% jitter filter expressed in seconds
    (jitter_threshold x typical step time); smaller changes are jitter."""
    return value > factor * median and (value - median) >= floor


class LocalHealth:
    """Lifeguard's Local Health Multiplier: saturating counter in [0, S]."""

    def __init__(self, max_value=8):
        self.max_value = max_value
        self.value = 0

    def _bump(self, delta):
        self.value = min(self.max_value, max(0, self.value + delta))
        return self.value

    def probe_succeeded(self):
        return self._bump(-1)

    def probe_failed(self):
        return self._bump(+1)

    def refuted_self(self):
        return self._bump(+1)

    def missed_nack(self):
        return self._bump(+1)

    def suspicion_dropped(self):
        return self._bump(+1)

    @property
    def multiplier(self):
        return self.value + 1


@dataclass
class Suspicion:
    about: str
    started: float
    cls: str | None = None
    origin: str | None = None
    confirmers: set = field(default_factory=set)


class Lifeguard:
    def __init__(self, n, lhm_max=8, base_probe_timeout=0.5, probe_interval=1.0,
                 alpha=5.0, beta=6.0, k_confirm=3, lateness_factor=1.1,
                 jitter_threshold=0.10, clock=time.monotonic):
        self.n = n
        self.health = LocalHealth(lhm_max)
        self.base_probe_timeout = base_probe_timeout
        self.probe_interval = probe_interval
        self.alpha = alpha
        self.beta = beta
        self.k_confirm = k_confirm
        self.factor = lateness_factor
        self.jitter_threshold = jitter_threshold
        self._clock = clock
        self._suspicions = {}

    # --- LHM-scaled timeouts (6.2) -------------------------------------
    @property
    def lhm(self):
        return self.health.value

    def probe_timeout(self):
        return self.base_probe_timeout * self.health.multiplier

    def probe_interval_scaled(self):
        return self.probe_interval * self.health.multiplier

    def required_consec(self, n_consec):
        return n_consec * self.health.multiplier

    # --- LHA-Suspicion (6.3) -------------------------------------------
    def suspicion_min(self):
        return self.alpha * math.log10(self.n) * self.probe_interval

    def suspicion_max(self):
        return self.beta * self.suspicion_min()

    def suspicion_timeout(self, confirmations):
        lo, hi = self.suspicion_min(), self.suspicion_max()
        frac = math.log(confirmations + 1) / math.log(self.k_confirm + 1)
        return max(lo, hi - (hi - lo) * frac)

    def start_suspicion(self, about, cls=None, origin=None, confirmers=()):
        """Start the suspicion timer for `about` unless one is running.
        `origin` raised the suspicion; `confirmers` already back it up."""
        if about not in self._suspicions:
            self._suspicions[about] = Suspicion(about, self._clock(), cls, origin,
                                                set(confirmers) - {origin})
        return self._suspicions[about]

    def add_confirmation(self, about, from_):
        """Count an independent confirmation (not the origin, not a repeat).
        Returns True if it was new."""
        s = self._suspicions.get(about)
        if s is None or from_ == s.origin or from_ in s.confirmers:
            return False
        s.confirmers.add(from_)
        return True

    def suspicion(self, about):
        return self._suspicions.get(about)

    def deadline(self, about):
        s = self._suspicions[about]
        return s.started + self.suspicion_timeout(len(s.confirmers))

    def expired(self):
        now = self._clock()
        return [a for a in list(self._suspicions) if now >= self.deadline(a)]

    def clear(self, about):
        return self._suspicions.pop(about, None)

    # --- Step-stat checks shared by the self-check, peers and classifier ----
    def jitter_floor(self, step_time):
        return self.jitter_threshold * step_time

    def slow_compute(self, stats, node, floor):
        values = stats.get("compute_t", {})
        if node not in values or len(values) < 2:
            return False
        return exceeds(values[node], statistics.median(values.values()), self.factor, floor)

    def waits_least(self, stats, node, floor):
        """The straggler signature: lowest wait_t, well below the median."""
        values = stats.get("wait_t", {})
        if node not in values or len(values) < 2:
            return False
        mine = values[node]
        if mine > min(values.values()):
            return False
        return exceeds(statistics.median(values.values()), mine, self.factor, floor)

    def lag_from(self, stats, succ, floor):
        """Did `succ` report lag originating at its predecessor this step?"""
        values = stats.get("pred_lag", {})
        if succ not in values or len(values) < 2:
            return False
        return exceeds(values[succ], statistics.median(values.values()), self.factor, floor)

    def implicated(self, stats, node, succ, floor):
        """Does the ring view point at `node` (with successor `succ`) as a
        straggler this step? Slow compute, the waits-least signature, or its
        successor reporting lag that originates at it."""
        return (self.slow_compute(stats, node, floor) or self.waits_least(stats, node, floor)
                or self.lag_from(stats, succ, floor))

    def is_self_outlier(self, stats, self_id, succ, floor):
        """Am I the slow one? If so, my view of others is not trusted (6.2)."""
        return self.implicated(stats, self_id, succ, floor)

    def classify(self, stats_window, suspect, floor):
        """FALCON split (6.5): mostly-slow compute_t -> compute, else network."""
        votes = [self.slow_compute(s, suspect, floor) for s in stats_window
                 if suspect in s.get("compute_t", {})]
        if votes and sum(votes) * 2 > len(votes):
            return COMPUTE
        return NETWORK

    # --- Buddy System (6.4) ----------------------------------------------
    @staticmethod
    def order_gossip(updates, recipient):
        """Stable reorder that puts SUSPECTs about `recipient` first."""
        first = [u for u in updates if u["type"] == "SUSPECT" and u["about"] == recipient]
        rest = [u for u in updates if not (u["type"] == "SUSPECT" and u["about"] == recipient)]
        return first + rest

    # --- Refutation (3.3 step 5) -------------------------------------------
    @staticmethod
    def should_refute(self_outlier, my_state):
        """A suspected node refutes only when its own self-check says it is
        not the outlier; a node that knows it is slow accepts the verdict."""
        if my_state in (membership.FAILED, membership.LEFT):
            return False
        return not self_outlier
