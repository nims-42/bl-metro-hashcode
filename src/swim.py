"""SWIM layer driven by the ring all-reduce (CONTEXT.md 3.3, 6.1, 6.4).

Mapping of SWIM terms onto this system:

* Direct probe: the ring all-reduce itself. The predecessor's chunks are the
  "ack". `pred_lag` is the part of this node's waiting that originates at the
  predecessor: for every chunk, the time blocked on it minus the time the
  predecessor was itself blocked before forwarding it (`fwd_wait` in the
  header). Delay inherited from further up the ring cancels out, so only the
  successor of a slow GPU or slow link sees a large pred_lag. A failed direct
  probe means the lateness rule (6.1) fired for N_consec steps in a row.
* Indirect probe: PING_REQ to k random peers (never this node or the
  suspect). Each peer answers from its own ring view: agree, disagree, or
  NACK when it has no data.
* Gossip: SUSPECT / ALIVE / CONFIRM / LEAVE ride on outgoing ring chunks,
  each sent about lambda*ln(n) times, least-sent first, at most G per chunk.

Only PING_REQ traffic is extra; everything else rides on frames that training
already sends. All public methods are thread-safe: the training loop, the
ping-req server and probe threads all call in here.
"""

import math
import random
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from src import membership as mb
from src import messages as msg
from src.lifeguard import exceeds


class RingView:
    """Everyone's per-step compute_t / wait_t / pred_lag, learned from the
    telemetry relayed in headers. Bounded to a few windows of steps."""

    FIELDS = ("compute_t", "wait_t", "pred_lag")

    def __init__(self, ids, keep_steps):
        self._data = {i: {} for i in ids}
        self._keep = keep_steps

    def record(self, node, step, **fields):
        if node not in self._data:
            return
        steps = self._data[node]
        steps.setdefault(step, {}).update(fields)
        if len(steps) > self._keep:
            for old in sorted(steps)[:-self._keep]:
                del steps[old]

    def stats_at(self, step):
        """{field: {node: value}} for one step, from whoever reported it."""
        out = {f: {} for f in self.FIELDS}
        for node, steps in self._data.items():
            for f, v in steps.get(step, {}).items():
                out[f][node] = v
        return out

    def steps(self, node):
        """Steps this view holds for `node`, newest first."""
        return sorted(self._data.get(node, {}), reverse=True)

    def latest(self, field):
        """Each node's most recent value of `field`: {node: value}."""
        out = {}
        for node, steps in self._data.items():
            for step in sorted(steps, reverse=True):
                if field in steps[step]:
                    out[node] = steps[step][field]
                    break
        return out

    def telemetry(self):
        """Latest complete entry per node, at most one each (bounded by N)."""
        entries = []
        for node, steps in self._data.items():
            for step in sorted(steps, reverse=True):
                e = steps[step]
                if all(f in e for f in self.FIELDS):
                    entries.append(msg.telemetry_entry(node, step, e["compute_t"],
                                                       e["wait_t"], e["pred_lag"]))
                    break
        return entries


class GossipBuffer:
    """Infection-style dissemination buffer (SWIM 4.1)."""

    def __init__(self, max_sends):
        self.max_sends = max_sends
        self._items = {}

    @staticmethod
    def key(update):
        return (update["type"], update["about"], update["incarnation"], update["from"])

    def add(self, update):
        k = self.key(update)
        if k in self._items:
            return False
        self._items[k] = [update, 0]
        return True

    def select(self, limit, recipient=None, order=None):
        """Pick up to `limit` updates, least-sent first (ties keep insertion
        order). `order` may reorder them for `recipient` (Buddy System)."""
        updates = [e[0] for e in sorted(self._items.values(), key=lambda e: e[1])]
        if order is not None:
            updates = order(updates, recipient)
        chosen = updates[:limit]
        for u in chosen:
            k = self.key(u)
            self._items[k][1] += 1
            if self._items[k][1] >= self.max_sends:
                del self._items[k]
        return chosen

    def __len__(self):
        return len(self._items)


def max_gossip_sends(n, lam):
    return max(1, math.ceil(lam * math.log(n)))


class Swim:
    def __init__(self, cfg, ids, membership, lifeguard, metrics, request_fn=None,
                 rng=None, spawn=None, clock=time.monotonic):
        self.cfg = cfg
        self.ids = list(ids)
        self.n = len(self.ids)
        self.self_id = membership.self_id
        idx = self.ids.index(self.self_id)
        self.pred = self.ids[(idx - 1) % self.n]
        self.succ = self.ids[(idx + 1) % self.n]
        self.membership = membership
        self.lifeguard = lifeguard
        self.metrics = metrics
        self.enabled = cfg.detection
        self._request = request_fn
        self._rng = rng or random.Random()
        self._spawn = spawn or (lambda f: threading.Thread(target=f, daemon=True).start())
        self._clock = clock
        self._lock = threading.RLock()
        self.view = RingView(self.ids, keep_steps=4 * cfg.window)
        self.gossip = GossipBuffer(max_gossip_sends(self.n, cfg.gossip_lambda))
        self._step_times = deque(maxlen=cfg.window)
        self._quiet_times = deque(maxlen=cfg.window)   # steps where pred was on time
        self._late_times = deque(maxlen=cfg.window)    # steps of the current late streak
        self._probing = set()
        self._last_probe = -math.inf
        self._quorum = max(2, self.n - 1)
        self.late_streak = 0

    # --- outgoing ------------------------------------------------------------
    def build_header(self, step, phase, chunk_idx, compute_t, prev_wait_t, fwd_wait,
                     wall=time.time):
        with self._lock:
            self.metrics.count("msgs_sent")
            if not self.enabled:
                return msg.make_header(self.self_id, step, phase, chunk_idx, wall(),
                                       compute_t, prev_wait_t, [], [], 0, 0, fwd_wait)
            gossip = self.gossip.select(self.cfg.gossip_max, self.succ,
                                        self.lifeguard.order_gossip)
            return msg.make_header(self.self_id, step, phase, chunk_idx, wall(), compute_t,
                                   prev_wait_t, self.view.telemetry(), gossip,
                                   self.n, self.cfg.gossip_max, fwd_wait)

    def leave(self):
        """Clean shutdown: gossip LEAVE about this node on the last chunks."""
        with self._lock:
            u = msg.gossip_update(msg.LEAVE, self.self_id, self.membership.incarnation,
                                  self.self_id)
            self.gossip.add(u)
            self.metrics.event("LEAVE", self.self_id, from_=self.self_id)

    # --- incoming ring frames -------------------------------------------------
    def on_header(self, header):
        with self._lock:
            self.metrics.count("msgs_recv")
            if not self.enabled:
                return
            src = header["src"]
            if src != self.self_id:
                self.view.record(src, header["step"], compute_t=header["compute_t"])
            for e in header.get("telemetry", []):
                if e["node_id"] != self.self_id:
                    self.view.record(e["node_id"], e["step"], compute_t=e["compute_t"],
                                     wait_t=e["wait_t"], pred_lag=e["pred_lag"])
            for u in header.get("gossip", []):
                self._on_gossip(u)

    def _on_gossip(self, u):
        about, kind = u["about"], u["type"]
        if about not in self.membership.ids():
            return
        if about == self.self_id:
            self._on_gossip_about_self(u)
            return
        before = self.membership.get(about)
        before_state, before_inc = before.state, before.incarnation
        if self.membership.apply(u):
            self.gossip.add(u)
            if kind == msg.SUSPECT:
                self.lifeguard.clear(about)
                self.lifeguard.start_suspicion(about, u.get("class"), origin=u["from"])
            elif kind in (msg.ALIVE, msg.CONFIRM, msg.LEAVE):
                self.lifeguard.clear(about)
                self._probing.discard(about)
            self.metrics.event(kind, about, u.get("class"), u["from"],
                               incarnation=u["incarnation"])
        elif (kind == msg.SUSPECT and before_state == mb.SUSPECT
              and u["incarnation"] == before_inc):
            # An independent suspicion about the same member shortens the
            # timeout (LHA-Suspicion); re-gossip the first K of them.
            if self.lifeguard.add_confirmation(about, u["from"]):
                s = self.lifeguard.suspicion(about)
                if len(s.confirmers) <= self.lifeguard.k_confirm:
                    self.gossip.add(u)
                self.metrics.event("SUSPECT_CONFIRMATION", about, u.get("class"), u["from"])

    def _on_gossip_about_self(self, u):
        kind = u["type"]
        me = self.membership.get(self.self_id)
        if kind == msg.CONFIRM:
            if self.membership.apply(u):
                self.gossip.add(u)
                self.metrics.event(kind, self.self_id, u.get("class"), u["from"])
            return
        if kind != msg.SUSPECT or me.state in (mb.FAILED, mb.LEFT):
            return
        if u["incarnation"] < me.incarnation:
            return
        if self.lifeguard.should_refute(self._self_outlier(), me.state):
            self._refute(u["incarnation"], u["from"])
        elif self.membership.apply(u):
            self.gossip.add(u)
            self.metrics.event("SUSPECTED_SELF", self.self_id, u.get("class"), u["from"])

    def _refute(self, seen_incarnation, accuser):
        new_inc = self.membership.refute(seen_incarnation)
        self.lifeguard.health.refuted_self()
        self.gossip.add(msg.gossip_update(msg.ALIVE, self.self_id, new_inc, self.self_id))
        self.metrics.event("REFUTE", self.self_id, from_=accuser, incarnation=new_inc)

    def _recheck_self(self):
        """While suspected, keep checking: once the ring view has not
        implicated this node for N_consec steps, refute before the suspicion
        times out (SWIM's suspect that answers again in time)."""
        me = self.membership.get(self.self_id)
        if me.state == mb.SUSPECT and self._self_clear():
            self._refute(me.incarnation, None)

    # --- end of a training step: direct probe ----------------------------------
    def end_step(self, step, compute_t, wait_t, pred_lag, step_t):
        with self._lock:
            self.view.record(self.self_id, step, compute_t=compute_t, wait_t=wait_t,
                             pred_lag=pred_lag)
            self._step_times.append(step_t)
            if not self.enabled:
                return
            self._check_suspicions()
            self._recheck_self()
            if step < self.cfg.warmup_steps:
                return
            if self.is_pred_late(pred_lag):
                self.late_streak += 1
                self._late_times.append(step_t)
            else:
                self.late_streak = 0
                self._late_times.clear()
                self._quiet_times.append(step_t)
            self._maybe_suspect(step)

    def is_pred_late(self, pred_lag):
        """Lateness rule (6.1): the predecessor's compute_t, or the part of
        my lag its compute does not explain, is above 1.1x the ring median
        and clears the 10% jitter floor; and the predecessor is on the
        critical path, not itself waiting on a slower node."""
        floor = self._floor()
        stats = {f: self.view.latest(f) for f in RingView.FIELDS}
        stats["pred_lag"][self.self_id] = pred_lag
        lags, computes = stats["pred_lag"], stats["compute_t"]
        if len(lags) < self._quorum or self.pred not in computes:
            return False
        if not self.lifeguard.critical(stats, self.pred, self.ids, floor):
            return False
        factor = self.lifeguard.factor
        residual = self.lifeguard.residual_lag(stats, self.pred, self.self_id)
        lag_late = exceeds(residual, statistics.median(lags.values()), factor, floor)
        compute_late = exceeds(computes[self.pred], statistics.median(computes.values()),
                               factor, floor)
        return lag_late or compute_late

    def step_time_changed(self):
        """Median step time during the late streak vs. over recent quiet
        steps: True if it rose by at least jitter_threshold (10%)."""
        if not self._quiet_times or not self._late_times:
            return True
        before = statistics.median(self._quiet_times)
        after = statistics.median(self._late_times)
        return after - before >= self.cfg.jitter_threshold * before

    def _maybe_suspect(self, step):
        if self.late_streak < self.lifeguard.required_consec(self.cfg.n_consec):
            return
        if self.membership.state(self.pred) != mb.ALIVE or self.pred in self._probing:
            return
        if self._clock() - self._last_probe < self.lifeguard.probe_interval_scaled():
            return
        streak = self.late_streak
        changed = self.step_time_changed()
        self.late_streak = 0
        self._late_times.clear()
        if not changed:
            # FALCON change-point check: a real straggler slows every step;
            # a step-time change under 10% is jitter.
            self.metrics.event("JITTER_IGNORED", self.pred, from_=self.self_id)
            return
        if self._self_outlier():
            # Lifeguard self-check: I am the slow one, so I do not accuse.
            self.metrics.event("ACCUSATION_HELD", self.pred, from_=self.self_id)
            return
        self.raise_suspicion(self.pred, (step - streak + 1, step))

    def raise_suspicion(self, suspect, window):
        """Mark `suspect` Suspect locally and confirm through k peers."""
        with self._lock:
            peers = self.pick_probe_peers(suspect)
            cls = self.lifeguard.classify(self._window_stats(window), suspect, self._floor())
            self.membership.mark_suspect_locally(suspect, cls)
            self._probing.add(suspect)
            self._last_probe = self._clock()
            self.metrics.event("SUSPECT_RAISED", suspect, cls, self.self_id,
                               peers=peers, window=list(window))
            timeout = self.lifeguard.probe_timeout()
        self._spawn(lambda: self._probe(suspect, peers, window, timeout))

    def pick_probe_peers(self, suspect):
        """k distinct random peers, never this node or the suspect."""
        pool = [m for m in self.ids if m not in (self.self_id, suspect)
                and self.membership.state(m) != mb.LEFT]
        return self._rng.sample(pool, min(self.cfg.k, len(pool)))

    def _probe(self, suspect, peers, window, timeout):
        request = msg.ping_req(suspect, self.self_id, window)
        if self._request is None or not peers:
            replies = [None] * len(peers)
        else:
            with ThreadPoolExecutor(max_workers=len(peers)) as pool:
                replies = list(pool.map(lambda p: self._request(p, request, timeout), peers))
        for _ in peers:
            self.metrics.count("ping_req_sent")
        self.on_probe_result(suspect, peers, replies)

    def on_probe_result(self, suspect, peers, replies):
        with self._lock:
            self._probing.discard(suspect)
            health = self.lifeguard.health
            missing = sum(r is None for r in replies)
            for _ in range(missing):
                health.missed_nack()
            if peers and missing == len(peers):
                health.probe_failed()
            elif missing == 0:
                health.probe_succeeded()
            acks = [r for r in replies if r and r.get("type") == msg.ACK]
            agree = [r["src"] for r in acks if r["agree"]]
            disagree = len(acks) - len(agree)
            nacks = sum(1 for r in replies if r and r.get("type") == msg.NACK)
            self.metrics.event("PING_REQ_RESULT", suspect, from_=self.self_id, agree=len(agree),
                               disagree=disagree, nack=nacks, missing=missing)
            if self.membership.state(suspect) != mb.SUSPECT:
                return
            if agree and len(agree) > disagree:
                member = self.membership.get(suspect)
                u = msg.gossip_update(msg.SUSPECT, suspect, member.incarnation, self.self_id,
                                      member.cls)
                self.gossip.add(u)
                self.lifeguard.start_suspicion(suspect, member.cls, origin=self.self_id,
                                               confirmers=agree)
                self.metrics.event(msg.SUSPECT, suspect, member.cls, self.self_id,
                                   incarnation=member.incarnation)
            else:
                self.membership.clear_local_suspicion(suspect)
                health.suspicion_dropped()
                self.metrics.event("SUSPICION_DROPPED", suspect, from_=self.self_id)

    def _check_suspicions(self):
        for about in self.lifeguard.expired():
            s = self.lifeguard.clear(about)
            member = self.membership.get(about)
            if member.state != mb.SUSPECT:
                continue
            # The class was decided over the late-streak window when the
            # suspicion was raised; recomputing over recent history would mix
            # in pre-fault steps.
            cls = s.cls or member.cls or self.lifeguard.classify(
                self._recent_stats(self.cfg.n_consec), about, self._floor())
            u = msg.gossip_update(msg.CONFIRM, about, member.incarnation, self.self_id, cls)
            self.membership.apply(u)
            self.gossip.add(u)
            self.metrics.event(msg.CONFIRM, about, cls, self.self_id)

    def tick(self):
        """Check suspicion timers outside the step loop."""
        with self._lock:
            if self.enabled:
                self._check_suspicions()

    # --- indirect probe: answering a PING_REQ -------------------------------
    def handle_ping_req(self, request):
        with self._lock:
            self.metrics.count("ping_req_recv")
            about = request["about"]
            if not self.enabled or about not in self.membership.ids():
                return msg.nack(about, self.self_id, "unknown")
            lo, hi = request["step_window"]
            window = [s for s in (self.view.stats_at(t) for t in range(lo, hi + 1))
                      if self._covered(s) and about in s["compute_t"]]
            if not window:
                return msg.nack(about, self.self_id, "no data")
            floor = self._floor()
            late = [self.lifeguard.implicated(s, about, self.ids, floor) for s in window]
            agree = sum(late) * 2 >= len(late)
            evidence = {"steps": len(window), "late_steps": sum(late),
                        "class": self.lifeguard.classify(window, about, floor)}
            return msg.ack(about, self.self_id, agree, evidence)

    # --- helpers -----------------------------------------------------------------
    def _floor(self):
        if not self._step_times:
            return 0.0
        return self.lifeguard.jitter_floor(statistics.median(self._step_times))

    def _window_stats(self, window):
        """Whatever this node has for the steps in `window`, covered or not."""
        lo, hi = window
        return [self.view.stats_at(t) for t in range(lo, hi + 1)]

    def _covered(self, stats):
        return len(stats["compute_t"]) >= self._quorum and len(stats["wait_t"]) >= self._quorum

    def _recent_stats(self, count):
        """Stats for the last `count` steps that every (but one) node reported."""
        out = []
        for t in self.view.steps(self.self_id):
            s = self.view.stats_at(t)
            if self._covered(s):
                out.append(s)
            if len(out) >= count:
                break
        return out[::-1]

    def _self_outlier(self):
        recent = self._recent_stats(self.cfg.n_consec)
        if not recent:
            return False
        floor = self._floor()
        votes = sum(self.lifeguard.is_self_outlier(s, self.self_id, self.ids, floor)
                    for s in recent)
        return votes * 2 > len(recent)

    def _self_clear(self):
        """True when each of the last N_consec covered steps clears this node."""
        recent = self._recent_stats(self.cfg.n_consec)
        if len(recent) < self.cfg.n_consec:
            return False
        floor = self._floor()
        return not any(self.lifeguard.is_self_outlier(s, self.self_id, self.ids, floor)
                       for s in recent)

    def snapshot(self):
        with self._lock:
            return self.membership.snapshot(), self.lifeguard.lhm
