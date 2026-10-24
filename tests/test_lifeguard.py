import math

import pytest

from src import messages as msg
from src.lifeguard import COMPUTE, NETWORK, LocalHealth, Lifeguard, exceeds
from src.membership import ALIVE, FAILED, LEFT


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


def make(n=5, clock=None, **kw):
    return Lifeguard(n, clock=clock or FakeClock(), **kw)


def stats(compute, wait):
    names = [msg.node_name(i) for i in range(len(compute))]
    return {"compute_t": dict(zip(names, compute)), "wait_t": dict(zip(names, wait))}


# --- LHM ------------------------------------------------------------------

def test_lhm_saturates_between_zero_and_s():
    h = LocalHealth(max_value=3)
    h.probe_succeeded()
    assert h.value == 0
    for _ in range(10):
        h.probe_failed()
    assert h.value == 3


@pytest.mark.parametrize("event,delta", [
    ("probe_succeeded", -1), ("probe_failed", +1), ("refuted_self", +1),
    ("missed_nack", +1), ("suspicion_dropped", +1),
])
def test_each_lhm_event_moves_counter_as_specified(event, delta):
    h = LocalHealth(max_value=8)
    h.value = 4
    getattr(h, event)()
    assert h.value == 4 + delta


def test_timeouts_scale_with_lhm_plus_one():
    lg = make(base_probe_timeout=0.5, probe_interval=1.0)
    assert lg.probe_timeout() == 0.5 and lg.required_consec(3) == 3
    lg.health.value = 4
    assert lg.probe_timeout() == pytest.approx(2.5)
    assert lg.probe_interval_scaled() == pytest.approx(5.0)
    assert lg.required_consec(3) == 15


# --- LHA-Suspicion ---------------------------------------------------------

def test_suspicion_timeout_is_max_at_zero_and_min_at_k():
    lg = make(alpha=5, beta=6, k_confirm=3, probe_interval=1.0)
    assert lg.suspicion_timeout(0) == pytest.approx(lg.suspicion_max())
    assert lg.suspicion_timeout(3) == pytest.approx(lg.suspicion_min())
    assert lg.suspicion_timeout(10) == pytest.approx(lg.suspicion_min())
    assert lg.suspicion_timeout(1) < lg.suspicion_timeout(0)
    assert lg.suspicion_timeout(2) < lg.suspicion_timeout(1)


@pytest.mark.parametrize("n", [3, 5, 8, 10, 64])
def test_suspicion_min_grows_with_log10_n(n):
    lg = make(n=n, alpha=5, beta=6, probe_interval=1.0)
    assert lg.suspicion_min() == pytest.approx(5 * math.log10(n))
    assert lg.suspicion_max() == pytest.approx(6 * lg.suspicion_min())
    assert make(n=n + 1).suspicion_min() > make(n=n).suspicion_min()


def test_suspicion_expires_on_injected_clock_and_shrinks_with_confirmations():
    clock = FakeClock(0.0)
    lg = make(clock=clock, alpha=2, beta=2, k_confirm=3)
    lg.start_suspicion("gpu2")
    clock.now = lg.suspicion_min() + 0.01
    assert lg.expired() == []
    for peer in ("gpu0", "gpu1", "gpu4"):
        assert lg.add_confirmation("gpu2", peer) is True
    assert lg.add_confirmation("gpu2", "gpu0") is False
    assert lg.expired() == ["gpu2"]
    lg.clear("gpu2")
    assert lg.expired() == []


# --- Self-check, classification, lateness ------------------------------------

def test_exceeds_needs_both_factor_and_jitter_floor():
    assert exceeds(0.30, 0.10, 1.1, 0.05)
    assert not exceeds(0.105, 0.10, 1.1, 0.0)
    assert not exceeds(0.13, 0.10, 1.1, 0.05)


def test_self_check_blocks_accusation_when_this_node_is_the_outlier():
    lg = make()
    s = stats(compute=[0.10, 0.10, 0.30, 0.10, 0.10], wait=[0.2, 0.2, 0.0, 0.2, 0.2])
    assert lg.is_self_outlier(s, "gpu2", "gpu3", floor=0.02)
    assert not lg.is_self_outlier(s, "gpu3", "gpu4", floor=0.02)


def test_self_check_flags_the_node_that_waits_least():
    lg = make()
    s = stats(compute=[0.1] * 5, wait=[0.20, 0.21, 0.01, 0.19, 0.20])
    assert lg.waits_least(s, "gpu2", floor=0.02)
    assert lg.is_self_outlier(s, "gpu2", "gpu3", floor=0.02)
    assert not lg.waits_least(s, "gpu3", floor=0.02)


def test_self_check_flags_a_node_whose_successor_reports_lag_from_it():
    lg = make()
    s = stats(compute=[0.1] * 5, wait=[0.1] * 5)
    s["pred_lag"] = {"gpu0": 0.0, "gpu1": 0.0, "gpu2": 0.0, "gpu3": 0.075, "gpu4": 0.001}
    assert lg.lag_from(s, "gpu3", floor=0.02)
    assert lg.implicated(s, "gpu2", "gpu3", floor=0.02)
    assert not lg.implicated(s, "gpu3", "gpu4", floor=0.02)


def test_classify_high_compute_is_compute():
    lg = make()
    window = [stats([0.1, 0.1, 0.3, 0.1, 0.1], [0.2] * 5)] * 4
    assert lg.classify(window, "gpu2", floor=0.02) == COMPUTE


def test_classify_normal_compute_with_late_arrival_is_network():
    lg = make()
    window = [stats([0.1, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.05, 0.3, 0.2])] * 4
    assert lg.classify(window, "gpu2", floor=0.02) == NETWORK


def test_refute_only_when_not_the_outlier():
    assert Lifeguard.should_refute(False, ALIVE)
    assert not Lifeguard.should_refute(True, ALIVE)
    assert not Lifeguard.should_refute(False, FAILED)
    assert not Lifeguard.should_refute(False, LEFT)


# --- Buddy System -------------------------------------------------------------

def test_buddy_system_puts_suspect_about_recipient_first():
    u = [msg.gossip_update(msg.ALIVE, "gpu1", 1, "gpu1"),
         msg.gossip_update(msg.SUSPECT, "gpu4", 0, "gpu0"),
         msg.gossip_update(msg.SUSPECT, "gpu3", 0, "gpu4"),
         msg.gossip_update(msg.CONFIRM, "gpu3", 0, "gpu4")]
    ordered = Lifeguard.order_gossip(u, "gpu3")
    assert ordered[0] == u[2]
    assert ordered[1:] == [u[0], u[1], u[3]]
