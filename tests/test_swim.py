import math
import queue
import random
import threading
import time

import numpy as np
import pytest

from src import messages as msg
from src.membership import ALIVE, FAILED, SUSPECT
from src.metrics import Metrics
from src.node import Config, build_swim, ring_allreduce
from src.swim import GossipBuffer, max_gossip_sends


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


# --- Ring all-reduce over an in-process ring ----------------------------------

def run_ring(n, vectors, delays=None, steps=1):
    """Run `steps` all-reduces on n threads linked by queues (inbox[i] is
    fed by node i-1). `delays[i]` is extra compute time per step on node i."""
    inbox = [queue.Queue() for _ in range(n)]
    results, lags, excess = [None] * n, [None] * n, [None] * n
    delays = delays or [0.0] * n

    def worker(i):
        prev_last = 0.0
        for _ in range(steps):
            time.sleep(0.02 + delays[i])

            def send(phase, idx, arr, fwd_wait):
                inbox[(i + 1) % n].put(({"phase": phase, "chunk_idx": idx, "fwd_wait": fwd_wait},
                                        np.array(arr, copy=True)))

            out, lg, ex = ring_allreduce(vectors[i], n, i, send, inbox[i].get,
                                         prev_last_wait=prev_last)
            prev_last = lg[-1]
        results[i], lags[i], excess[i] = out, lg, ex

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results, lags, excess


@pytest.mark.parametrize("n", [3, 5, 8, 10])
@pytest.mark.parametrize("d", [1, 7, 100, 1001])
def test_ring_allreduce_sums_for_any_n_and_d(n, d):
    rng = np.random.default_rng(n * 1000 + d)
    vectors = [rng.standard_normal(d) for _ in range(n)]
    results, lags, _ = run_ring(n, vectors)
    expected = np.sum(vectors, axis=0)
    for out, lg in zip(results, lags):
        assert out.shape == (d,)
        np.testing.assert_allclose(out, expected)
        assert len(lg) == 2 * (n - 1)


@pytest.mark.parametrize("n,slow", [(5, 2), (8, 7)])
def test_straggler_waits_least_and_its_successor_sees_the_lag(n, slow):
    delays = [0.0] * n
    delays[slow] = 0.15
    _, lags, excess = run_ring(n, [np.ones(64)] * n, delays=delays, steps=3)
    wait = [sum(lg) for lg in lags]
    pred_lag = [sum(ex) for ex in excess]
    assert int(np.argmin(wait)) == slow
    assert int(np.argmax(pred_lag)) == (slow + 1) % n
    others = [pred_lag[i] for i in range(n) if i != (slow + 1) % n]
    assert max(others) < 0.5 * pred_lag[(slow + 1) % n]


# --- Swim harness -------------------------------------------------------------

def make_swim(n=5, me=3, request_fn=None, clock=None, seed=0, **overrides):
    params = dict(node_id=me, n=n, warmup_steps=0, n_consec=3, window=20,
                  probe_interval=0.01, alpha=1.0, beta=2.0)
    params.update(overrides)
    cfg = Config(**params)
    metrics = Metrics(cfg.name, n, "test", ".", enabled=False)
    clock = clock or FakeClock()
    swim = build_swim(cfg, metrics, request_fn=request_fn, rng=random.Random(seed),
                      spawn=lambda f: f(), clock=clock)
    return swim, metrics, clock


def feed(swim, step, step_t=0.11, compute=None, wait=None, lag=None):
    """Record one step for every node (others via the view, self via end_step)."""
    n = swim.n
    compute = compute or {}
    wait = wait or {}
    lag = lag or {}
    for i in range(n):
        node = msg.node_name(i)
        c, w, l = compute.get(node, 0.10), wait.get(node, 0.01), lag.get(node, 0.001)
        if node == swim.self_id:
            continue
        swim.view.record(node, step, compute_t=c, wait_t=w, pred_lag=l)
    me = swim.self_id
    swim.end_step(step, compute.get(me, 0.10), wait.get(me, 0.01), lag.get(me, 0.001), step_t)


def agree_all(peer, request, timeout):
    return msg.ack(request["about"], peer, True, {})


def events(metrics, kind):
    return [e for e in metrics._events if e["type"] == kind]


def quiet(swim, steps):
    for s in steps:
        feed(swim, s)


def slow_compute_step(swim, s, suspect="gpu2"):
    feed(swim, s, step_t=0.21, compute={suspect: 0.20}, wait={suspect: 0.0},
         lag={swim.self_id: 0.10})


# --- Lateness rule ------------------------------------------------------------

def test_lateness_fires_after_n_consec_late_steps():
    swim, metrics, _ = make_swim(request_fn=agree_all)
    quiet(swim, range(10))
    slow_compute_step(swim, 10)
    slow_compute_step(swim, 11)
    assert swim.late_streak == 2 and swim.membership.state("gpu2") == ALIVE
    slow_compute_step(swim, 12)
    raised = events(metrics, "SUSPECT_RAISED")
    assert len(raised) == 1 and raised[0]["about"] == "gpu2"
    assert raised[0]["class"] == "compute"
    assert swim.membership.state("gpu2") == SUSPECT


def test_late_streak_resets_on_an_on_time_step():
    swim, metrics, _ = make_swim(request_fn=agree_all)
    quiet(swim, range(10))
    slow_compute_step(swim, 10)
    slow_compute_step(swim, 11)
    feed(swim, 12)
    slow_compute_step(swim, 13)
    assert swim.late_streak == 1
    assert not events(metrics, "SUSPECT_RAISED")


def test_lateness_needs_more_than_1_1x_the_ring_median():
    swim, _, _ = make_swim()
    quiet(swim, range(10))
    lags = {msg.node_name(i): 0.10 for i in range(5)}
    feed(swim, 10, lag={**lags, swim.self_id: 0.105})
    assert swim.late_streak == 0
    feed(swim, 11, lag={**lags, swim.self_id: 0.20})
    assert swim.late_streak == 1


def test_changes_under_ten_percent_are_jitter():
    swim, metrics, _ = make_swim(request_fn=agree_all)
    quiet(swim, range(10))
    for s in range(10, 13):   # pred looks late but step time rose only ~5%
        feed(swim, s, step_t=0.115, lag={swim.self_id: 0.05})
    assert events(metrics, "JITTER_IGNORED")
    assert not events(metrics, "SUSPECT_RAISED")
    assert swim.membership.state("gpu2") == ALIVE


def test_small_lag_under_the_jitter_floor_is_not_late():
    swim, _, _ = make_swim()
    quiet(swim, range(10))
    feed(swim, 10, lag={swim.self_id: 0.005})   # 5x the median, but 5 ms on a 110 ms step
    assert swim.late_streak == 0


def test_self_check_holds_back_an_accusation_from_a_slow_node():
    swim, metrics, _ = make_swim(request_fn=agree_all)
    quiet(swim, range(10))
    for s in range(10, 13):   # this node is itself the compute outlier
        feed(swim, s, step_t=0.30, compute={swim.self_id: 0.25, "gpu2": 0.25},
             lag={swim.self_id: 0.10})
    assert events(metrics, "ACCUSATION_HELD")
    assert not events(metrics, "SUSPECT_RAISED")


# --- Indirect probe -------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 5, 8, 10])
def test_ping_req_goes_to_k_distinct_peers_never_accuser_or_suspect(n):
    for seed in range(20):
        swim, _, _ = make_swim(n=n, me=0, seed=seed, k=2)
        for target in range(1, n):
            suspect = msg.node_name(target)
            peers = swim.pick_probe_peers(suspect)
            assert len(peers) == 2 and len(set(peers)) == 2
            assert swim.self_id not in peers and suspect not in peers


def run_to_suspicion(request_fn):
    swim, metrics, clock = make_swim(request_fn=request_fn)
    quiet(swim, range(10))
    for s in range(10, 13):
        slow_compute_step(swim, s)
    return swim, metrics, clock


def test_peers_agree_gossips_suspect_and_starts_shortened_timer():
    swim, metrics, _ = run_to_suspicion(agree_all)
    assert [e["agree"] for e in events(metrics, "PING_REQ_RESULT")] == [2]
    assert swim.membership.state("gpu2") == SUSPECT
    s = swim.lifeguard.suspicion("gpu2")
    assert len(s.confirmers) == 2 and s.origin == swim.self_id
    sent = swim.gossip.select(10)
    assert any(u["type"] == msg.SUSPECT and u["about"] == "gpu2" for u in sent)
    assert swim.lifeguard.lhm == 0


def test_peers_disagree_drops_suspicion_and_raises_lhm():
    disagree = lambda p, r, t: msg.ack(r["about"], p, False, {})
    swim, metrics, _ = run_to_suspicion(disagree)
    assert events(metrics, "SUSPICION_DROPPED")
    assert swim.membership.state("gpu2") == ALIVE
    assert len(swim.gossip) == 0
    assert swim.lifeguard.lhm == 1


def test_missing_replies_raise_lhm():
    swim, metrics, _ = run_to_suspicion(lambda p, r, t: None)
    # two missed nacks + failed probe + dropped suspicion
    assert swim.lifeguard.lhm == 4
    assert swim.membership.state("gpu2") == ALIVE


def test_suspicion_timeout_confirms_with_class():
    swim, metrics, clock = run_to_suspicion(agree_all)
    deadline = swim.lifeguard.deadline("gpu2")
    clock.now = deadline + 0.001
    swim.tick()
    assert swim.membership.state("gpu2") == FAILED
    assert swim.membership.get("gpu2").cls == "compute"
    assert events(metrics, msg.CONFIRM)[0]["from"] == swim.self_id


def test_handle_ping_req_answers_from_own_view():
    swim, _, _ = make_swim(me=0)
    for s in range(5):
        feed(swim, s, compute={"gpu2": 0.25}, wait={"gpu2": 0.0}, lag={"gpu3": 0.15})
    reply = swim.handle_ping_req(msg.ping_req("gpu2", "gpu3", (0, 4)))
    assert reply["type"] == msg.ACK and reply["agree"] is True
    assert reply["evidence"]["class"] == "compute"
    healthy = swim.handle_ping_req(msg.ping_req("gpu4", "gpu0", (0, 4)))
    assert healthy["type"] == msg.ACK and healthy["agree"] is False
    assert swim.handle_ping_req(msg.ping_req("gpu2", "gpu3", (50, 60)))["type"] == msg.NACK


def test_healthy_node_refutes_suspicion_with_higher_incarnation():
    swim, metrics, _ = make_swim(me=2)
    quiet(swim, range(5))
    header = msg.make_header("gpu1", 5, msg.RS, 0, 0.0, 0.1, 0.0, [],
                             [msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu3")], 5, 6)
    swim.on_header(header)
    assert swim.membership.incarnation == 1
    assert swim.membership.state("gpu2") == ALIVE
    alive = [u for u in swim.gossip.select(10) if u["type"] == msg.ALIVE]
    assert alive and alive[0]["incarnation"] == 1
    assert swim.lifeguard.lhm == 1


def test_slow_node_accepts_suspicion_about_itself():
    swim, metrics, _ = make_swim(me=2)
    for s in range(5):
        feed(swim, s, compute={"gpu2": 0.25}, wait={"gpu2": 0.0}, lag={"gpu3": 0.15})
    header = msg.make_header("gpu1", 5, msg.RS, 0, 0.0, 0.1, 0.0, [],
                             [msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu3")], 5, 6)
    swim.on_header(header)
    assert swim.membership.incarnation == 0
    assert swim.membership.state("gpu2") == SUSPECT
    assert events(metrics, "SUSPECTED_SELF")


def test_suspected_node_refutes_once_it_recovers():
    swim, metrics, _ = make_swim(me=2)
    for s in range(5):
        feed(swim, s, compute={"gpu2": 0.25}, wait={"gpu2": 0.0}, lag={"gpu3": 0.15})
    header = msg.make_header("gpu1", 5, msg.RS, 0, 0.0, 0.1, 0.0, [],
                             [msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu3")], 5, 6)
    swim.on_header(header)
    assert swim.membership.state("gpu2") == SUSPECT
    feed(swim, 5)
    feed(swim, 6)
    assert swim.membership.state("gpu2") == SUSPECT      # not yet N_consec clean steps
    feed(swim, 7)
    assert swim.membership.state("gpu2") == ALIVE
    assert swim.membership.incarnation == 1
    assert events(metrics, "REFUTE")


def test_independent_suspicions_count_as_confirmations():
    swim, _, _ = make_swim(me=0)
    first = msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu3")
    swim.on_header(msg.make_header("gpu4", 1, msg.RS, 0, 0, 0.1, 0, [], [first], 5, 6))
    assert swim.lifeguard.suspicion("gpu2").confirmers == set()
    again = msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu1")
    swim.on_header(msg.make_header("gpu4", 1, msg.RS, 1, 0, 0.1, 0, [], [first, again], 5, 6))
    assert swim.lifeguard.suspicion("gpu2").confirmers == {"gpu1"}


# --- Gossip ---------------------------------------------------------------------

def test_gossip_sends_least_sent_first_and_respects_g():
    buf = GossipBuffer(max_sends=10)
    updates = [msg.gossip_update(msg.ALIVE, f"gpu{i}", 1, f"gpu{i}") for i in range(8)]
    for u in updates:
        buf.add(u)
    first = buf.select(6)
    assert first == updates[:6]
    second = buf.select(6)
    assert second[:2] == updates[6:]          # never-sent ones go first
    assert len(second) == 6


@pytest.mark.parametrize("n", [3, 5, 10, 100])
def test_each_update_is_sent_about_lambda_log_n_times(n):
    limit = max_gossip_sends(n, 3.0)
    assert limit == max(1, math.ceil(3.0 * math.log(n)))
    buf = GossipBuffer(limit)
    buf.add(msg.gossip_update(msg.SUSPECT, "gpu1", 0, "gpu0"))
    sends = 0
    while len(buf):
        sends += len(buf.select(6))
    assert sends == limit


def test_buddy_system_puts_suspicion_about_successor_first():
    swim, _, _ = make_swim(me=1)          # successor is gpu2
    for i in range(6):
        swim.gossip.add(msg.gossip_update(msg.ALIVE, "gpu4", i + 1, "gpu4"))
    swim.gossip.add(msg.gossip_update(msg.SUSPECT, "gpu2", 0, "gpu3"))
    header = swim.build_header(1, msg.RS, 0, 0.1, 0.0, 0.0)
    assert header["gossip"][0]["type"] == msg.SUSPECT and header["gossip"][0]["about"] == "gpu2"
    assert len(header["gossip"]) <= swim.cfg.gossip_max


def test_header_telemetry_is_bounded_by_n():
    swim, _, _ = make_swim(n=5)
    quiet(swim, range(30))
    header = swim.build_header(30, msg.RS, 0, 0.1, 0.0, 0.0)
    assert len(header["telemetry"]) <= 5
    assert {e["node_id"] for e in header["telemetry"]} == set(swim.ids)


def test_detection_off_sends_bare_headers():
    swim, _, _ = make_swim(detection=False)
    swim.gossip.add(msg.gossip_update(msg.ALIVE, "gpu4", 1, "gpu4"))
    quiet(swim, range(5))
    header = swim.build_header(5, msg.RS, 0, 0.1, 0.0, 0.0)
    assert header["telemetry"] == [] and header["gossip"] == []
