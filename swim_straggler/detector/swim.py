"""The SWIM-style straggler detector of one worker.

How it maps onto SWIM:
  * The ping is replaced by the ring all-reduce itself: every step, each worker measures how long
    it waited for its predecessor, round by round. A worker that computes slowly makes the
    workers after it wait (wait_profile.py says which worker, from the pattern of waits).
  * Suspect: a worker that blames the same peer for `consecutive_steps` steps in a row suspects it.
  * Confirm: like SWIM's indirect probe through k random members, the suspecting worker consults
    `k_probes` randomly chosen peers. Each peer checks its own table (compute times it learned
    by gossip). The suspicion is confirmed if a majority of the consulted peers agree.
  * Gossip: the table rows, suspicion notices and verdicts ride along in the header of every
    ring message, so the load per message is constant however many workers there are.

How to use it from a training loop (one detector object per worker):
    detector.start_step(step, compute_seconds)        # before the all-reduce
    ... in every all-reduce round:
        send:    detector.make_gossip()               # put the numbers into the outgoing header
        receive: detector.receive_gossip(numbers)     # numbers from the incoming header
    events = detector.finish_step(step, wait_seconds_per_round)   # after the all-reduce
"""

import random
from dataclasses import dataclass, field

from swim_straggler.detector.base import Detector, trace_event
from swim_straggler.detector.gossip import (
    CLEARED_VERDICT,
    STRAGGLER_VERDICT,
    SUSPICION,
    Notice,
    decode_gossip,
    encode_gossip,
    gossip_length,
)
from swim_straggler.detector.health import is_locally_healthy
from swim_straggler.detector.table import ALIVE, STRAGGLER, SUSPECT, MembershipTable
from swim_straggler.detector.wait_profile import estimate_culprit

STALE_ROW_AFTER_STEPS = 3


@dataclass
class Probe:
    """A suspicion this worker started and is waiting to have confirmed or rejected."""

    start_step: int
    peers: list
    answers_heard: dict = field(default_factory=dict)  # peer -> its answer (True: slow), for the trace


class StragglerDetector(Detector):
    def __init__(self, rank, world_size, config, seed=0):
        """`config` is a DetectorConfig (any object with the same fields works)."""
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.random = random.Random(seed * 100003 + rank)
        self.table = MembershipTable(world_size)
        self.current_step = -1

        self.accused_rank = None  # the worker my waits point to
        self.accusation_streak = 0  # for how many steps in a row
        self.probes = {}  # subject -> Probe, the suspicions I started
        self.rejected_at_step = {}  # subject -> step; a rejected suspicion is not repeated at once
        self.suspicion_notices = {}  # subject -> Notice, being passed on
        self.verdict_notices = {}  # subject -> Notice, being passed on
        self.steps_without_slowness = {}  # subject -> consecutive steps a confirmed straggler looked normal
        self.next_row_position = 0  # rotates through the table when it has more rows than fit in a message
        self.other_ranks = [rank for rank in range(world_size) if rank != self.rank]
        self.events = []

        # Only for the trace (detector.trace): what was last reported, so that a line is written when
        # something changes instead of the same line every step.
        self.trace = getattr(config, "trace", False)
        self.gossip_sender = None  # the worker the gossip being absorbed came from
        self.distrusts_own_waits = False
        self.last_answer = None
        self.cooldown_reported = {}  # subject -> the rejection step already reported
        self.stale_reported = set()

        # EXPERIMENTAL network diagnosis (detector.diagnose_network)
        self.diagnose_network = getattr(config, "diagnose_network", False)
        self.normal_compute_rejections = []  # steps of rejected suspicions whose subject computed normally
        self.network_diagnosed_at_step = None

    @property
    def gossip_length(self):
        return gossip_length(self.config.gossip_rows, self.config.verdict_slots)

    def straggler_ranks(self):
        return [rank for rank in range(self.world_size) if self.table.states[rank] == STRAGGLER]

    def suspect_ranks(self):
        return [rank for rank in range(self.world_size) if self.table.states[rank] == SUSPECT]

    # ---------------------------------------------------------------- called by the training loop

    def start_step(self, step, compute_seconds):
        self.current_step = step
        self.table.update_own_compute(self.rank, step, compute_seconds, self.config.ewma_alpha)

    def make_gossip(self):
        """The numbers to put into the header of the next outgoing message."""
        return encode_gossip(
            self.rows_to_send(), self.notices_to_send(), self.config.gossip_rows, self.config.verdict_slots
        )

    def receive_gossip(self, numbers, sender=None):
        """Absorb the numbers from the header of an incoming message (`sender`: the worker it came from)."""
        rows, notices = decode_gossip(numbers, self.config.gossip_rows, self.config.verdict_slots)
        self.absorb_gossip(rows, notices, sender)

    def gossip_contents(self):
        """The rows and notices of the next message, as objects instead of numbers.

        Used by the simulator, which passes messages between detectors in one process and so can
        skip turning them into numbers and back. The contents are exactly what make_gossip would
        encode. My own row is copied because I keep changing it; the other rows and the notices
        are never changed after they are stored, so they can be shared.
        """
        rows = self.rows_to_send()
        return [rows[0].copy()] + rows[1:], self.notices_to_send()

    def absorb_gossip(self, rows, notices, sender=None):
        """What receive_gossip does once the numbers are turned back into rows and notices."""
        self.gossip_sender = sender
        for row in rows:
            if row.rank != self.rank:  # nobody knows my own state better than I do
                self.table.merge_row(row)
        for notice in notices:
            self.handle_notice(notice)

    def finish_step(self, step, wait_seconds_per_round, compute_seconds_of_every_worker=None):
        """Run the detection logic once the all-reduce of this step is done. Returns the events since the last call."""
        if step >= self.config.warmup_steps:
            self.update_accusation(wait_seconds_per_round)
            self.start_probe(step)
            self.resolve_probes(step)
            self.check_recoveries(step)
            self.answer_suspicions(step)
        self.drop_expired_notices(step)
        new_events, self.events = self.events, []  # includes events caused by gossip received during this step
        return new_events

    # ---------------------------------------------------------------- what goes into a message

    def rows_to_send(self):
        """My own row first (it is always the freshest), then a rotating selection of the others."""
        rows = self.table.rows
        other_ranks = self.other_ranks
        count = len(other_ranks)
        room = self.config.gossip_rows - 1
        start = self.next_row_position
        selected = []
        for offset in range(count):
            if len(selected) == room:
                break
            row = rows[other_ranks[(start + offset) % count]]
            if row.step >= 0:  # never heard of: nothing to pass on
                selected.append(row)
        self.next_row_position = (start + max(room, 1)) % max(count, 1)
        return [rows[self.rank]] + selected

    def notices_to_send(self):
        """Verdicts first, then suspicions, newest first. Anything beyond the slots has to wait."""
        verdicts = sorted(self.verdict_notices.values(), key=lambda notice: -notice.step)
        suspicions = sorted(self.suspicion_notices.values(), key=lambda notice: -notice.step)
        return (verdicts + suspicions)[: self.config.verdict_slots]

    # ---------------------------------------------------------------- notices from other workers

    def handle_notice(self, notice):
        if notice.step + self.config.verdict_repeat_steps < self.current_step:
            return  # old news; ignoring it is what makes notices die out instead of circulating forever
        if notice.kind == SUSPICION:
            known = self.suspicion_notices.get(notice.subject)
            if known is None or notice.step > known.step:
                self.suspicion_notices[notice.subject] = notice
                if notice.subject != self.rank and self.table.states[notice.subject] == ALIVE:  # news to me
                    self.add_trace("heard_suspicion", notice.subject, self.current_step,
                                   suspected_at_step=notice.step, via=self.gossip_sender)
            return

        known = self.verdict_notices.get(notice.subject)
        if known is not None and known.step >= notice.step:
            return
        self.verdict_notices[notice.subject] = notice
        new_state = STRAGGLER if notice.kind == STRAGGLER_VERDICT else ALIVE
        if self.table.states[notice.subject] != new_state:
            self.table.states[notice.subject] = new_state
            self.probes.pop(notice.subject, None)
            self.steps_without_slowness[notice.subject] = 0
            action = "learned_straggler" if new_state == STRAGGLER else "learned_cleared"
            self.add_event(action, notice.subject, self.current_step, decided_at_step=notice.step, via=self.gossip_sender)

    # ---------------------------------------------------------------- the logic of one step

    def update_accusation(self, wait_seconds_per_round):
        """Decide whom my own waiting times point to, and for how many steps in a row."""
        step = self.current_step
        own_compute = self.table.rows[self.rank].compute_seconds
        typical_compute = self.table.median_compute_seconds()
        healthy = is_locally_healthy(own_compute, typical_compute, self.config.health_factor)
        if healthy == self.distrusts_own_waits:  # the health check changed its answer
            self.distrusts_own_waits = not healthy
            self.add_trace("distrusts_own_waits" if not healthy else "trusts_own_waits_again", self.rank, step,
                           own_compute_ms=milliseconds(own_compute), typical_compute_ms=milliseconds(typical_compute),
                           health_factor=self.config.health_factor)

        accused = None
        reason = "its waits show nobody standing out"
        if not healthy:
            reason = "it is slow itself, so it does not trust its own waits"
        else:
            accused = estimate_culprit(
                self.rank, self.world_size, wait_seconds_per_round, self.config.wait_floor_seconds
            )
        pointed_to = accused
        if accused is not None and self.table.states[accused] != ALIVE:
            accused = None  # already suspected or confirmed, nothing new to learn
            reason = f"its waits now point to w{pointed_to}, already {self.table.states[pointed_to]}, nothing more to count"

        previous_accused, previous_streak = self.accused_rank, self.accusation_streak
        if accused is not None and accused == self.accused_rank:
            self.accusation_streak += 1
        else:
            self.accused_rank = accused
            self.accusation_streak = 1 if accused is not None else 0

        if accused is not None and self.accusation_streak <= self.config.consecutive_steps:
            self.add_trace("points_to", accused, step, streak=self.accusation_streak,
                           needed=self.config.consecutive_steps, **wait_details(self.rank, self.world_size, accused,
                                                                                   wait_seconds_per_round))
        if previous_accused is not None and previous_accused != accused:
            self.add_trace("stopped_pointing", previous_accused, step, after_steps=previous_streak, reason=reason)

    def start_probe(self, step):
        """Suspect the accused worker once my waits have pointed to it for long enough."""
        subject = self.accused_rank
        if subject is None or self.accusation_streak < self.config.consecutive_steps:
            return
        if subject in self.probes:
            return
        recently_rejected = step - self.rejected_at_step.get(subject, -10**9) < self.config.recovery_steps
        if recently_rejected:
            rejected_at = self.rejected_at_step[subject]
            if self.cooldown_reported.get(subject) != rejected_at:
                self.cooldown_reported[subject] = rejected_at
                self.add_trace("waits_after_rejection", subject, step, rejected_at_step=rejected_at,
                               may_suspect_again_at_step=rejected_at + self.config.recovery_steps)
            return

        candidates = [rank for rank in range(self.world_size) if rank not in (self.rank, subject)]
        peers = self.random.sample(candidates, min(self.config.k_probes, len(candidates)))
        self.probes[subject] = Probe(start_step=step, peers=peers)
        self.table.states[subject] = SUSPECT
        self.suspicion_notices[subject] = Notice(SUSPICION, subject, step)
        self.add_event("suspected", subject, step, consulted_peers=peers)

    def resolve_probes(self, step):
        """Count the answers of the consulted peers and confirm or reject each suspicion."""
        for subject, probe in list(self.probes.items()):
            votes_for = 0
            votes_against = 0
            for peer in probe.peers:
                row = self.table.rows[peer]
                has_answered = row.probe_subject == subject and row.probe_answer_step >= probe.start_step
                if has_answered and probe.answers_heard.get(peer) != row.probe_says_slow:
                    probe.answers_heard[peer] = row.probe_says_slow
                    self.add_trace("got_answer", subject, step, peer=peer, says_slow=row.probe_says_slow,
                                   answered_at_step=row.probe_answer_step)
                if has_answered and row.probe_says_slow:
                    votes_for += 1
                elif has_answered:
                    votes_against += 1
            number_of_peers = len(probe.peers)
            not_yet_answered = number_of_peers - votes_for - votes_against

            if votes_for * 2 > number_of_peers:
                self.confirm(subject, step, votes_for, number_of_peers)
            elif (votes_for + not_yet_answered) * 2 <= number_of_peers:
                self.reject(subject, step, "majority disagrees", votes_for, number_of_peers)
            elif step - probe.start_step >= self.config.verdict_repeat_steps:
                self.reject(subject, step, "not enough answers in time", votes_for, number_of_peers)

    def compute_ratio(self, subject):
        """The subject's compute time against the typical one, from my table (None if unknown)."""
        reference = self.table.reference_median(subject)
        if self.table.rows[subject].step < 0 or not reference:
            return None
        return round(self.table.rows[subject].compute_seconds / reference, 3)

    def confirm(self, subject, step, votes_for, number_of_peers):
        del self.probes[subject]
        self.table.states[subject] = STRAGGLER
        self.verdict_notices[subject] = Notice(STRAGGLER_VERDICT, subject, step)
        self.steps_without_slowness[subject] = 0
        self.add_event("confirmed", subject, step, votes_for=votes_for, consulted=number_of_peers,
                       diagnosis="compute", compute_ratio=self.compute_ratio(subject))

    def reject(self, subject, step, reason, votes_for, number_of_peers):
        del self.probes[subject]
        self.table.states[subject] = ALIVE
        self.suspicion_notices.pop(subject, None)
        self.rejected_at_step[subject] = step
        ratio = self.compute_ratio(subject)
        self.add_event("rejected", subject, step, reason=reason, votes_for=votes_for, consulted=number_of_peers,
                       compute_ratio=ratio)
        if self.diagnose_network and ratio is not None and ratio <= self.config.factor:
            self.diagnose_delay_without_slow_compute(step)

    def diagnose_delay_without_slow_compute(self, step):
        """EXPERIMENTAL. My waits pointed at a worker for a whole streak, yet it computes normally. Once that has
        happened `diagnosis_rejections` times within `diagnosis_window_steps`, say the delay is not compute.
        A slow link delays messages, which shows up as waits like a slow worker's, but nobody computes slowly."""
        window = self.config.diagnosis_window_steps
        self.normal_compute_rejections = [earlier for earlier in self.normal_compute_rejections if step - earlier < window]
        self.normal_compute_rejections.append(step)
        recently_said = self.network_diagnosed_at_step is not None and step - self.network_diagnosed_at_step < window
        if len(self.normal_compute_rejections) >= self.config.diagnosis_rejections and not recently_said:
            self.network_diagnosed_at_step = step
            self.add_event("delay_not_compute", self.rank, step, diagnosis="network (experimental)",
                           rejections=len(self.normal_compute_rejections), window_steps=window)

    def check_recoveries(self, step):
        """A confirmed straggler that has looked normal for `recovery_steps` steps is cleared."""
        for subject in self.straggler_ranks():
            row = self.table.rows[subject]
            is_fresh = row.step >= step - STALE_ROW_AFTER_STEPS
            if not is_fresh:
                if subject not in self.stale_reported:
                    self.stale_reported.add(subject)
                    self.add_trace("no_recent_news", subject, step, last_heard_step=row.step)
                continue  # no recent news from that worker, so no evidence either way
            self.stale_reported.discard(subject)
            if self.table.looks_slow(subject, self.config.factor):
                if self.steps_without_slowness.get(subject, 0) > 0:
                    self.add_trace("recovery_restarted", subject, step,
                                   normal_steps_before=self.steps_without_slowness[subject])
                self.steps_without_slowness[subject] = 0
                continue
            self.steps_without_slowness[subject] = self.steps_without_slowness.get(subject, 0) + 1
            if self.steps_without_slowness[subject] >= self.config.recovery_steps:
                self.table.states[subject] = ALIVE
                self.verdict_notices[subject] = Notice(CLEARED_VERDICT, subject, step)
                self.add_event("cleared", subject, step)

    def answer_suspicions(self, step):
        """If anybody suspects a worker, say whether my own table agrees (one answer slot: the newest suspicion)."""
        suspicions = [notice for notice in self.suspicion_notices.values() if notice.subject != self.rank]
        if not suspicions:
            return
        newest = max(suspicions, key=lambda notice: notice.step)
        own_compute = self.table.rows[self.rank].compute_seconds
        healthy = is_locally_healthy(own_compute, self.table.median_compute_seconds(), self.config.health_factor)
        own_row = self.table.rows[self.rank]
        own_row.probe_subject = newest.subject
        own_row.probe_answer_step = step
        own_row.probe_says_slow = healthy and self.table.looks_slow(newest.subject, self.config.factor)
        answer = (newest.subject, own_row.probe_says_slow, healthy)
        if answer != self.last_answer:
            self.last_answer = answer
            subject_row = self.table.rows[newest.subject]
            self.add_trace("answers", newest.subject, step, says_slow=own_row.probe_says_slow, trusts_itself=healthy,
                           subject_compute_ms=milliseconds(subject_row.compute_seconds if subject_row.step >= 0 else None),
                           typical_compute_ms=milliseconds(self.table.reference_median(newest.subject)),
                           factor=self.config.factor)

    def drop_expired_notices(self, step):
        oldest_step_to_keep = step - self.config.verdict_repeat_steps
        for notices in (self.suspicion_notices, self.verdict_notices):
            for subject in [subject for subject, notice in notices.items() if notice.step < oldest_step_to_keep]:
                del notices[subject]

    def add_event(self, action, subject, step, **details):
        self.events.append({"action": action, "subject": subject, "step": step, **details})

    def add_trace(self, action, subject, step, **details):
        if self.trace:
            self.events.append(trace_event(action, subject, step, **details))


def milliseconds(seconds):
    return None if seconds is None else round(seconds * 1000, 2)


def wait_details(rank, world_size, accused, wait_seconds_per_round):
    """Which round's wait points to `accused`, how long it was, and the average of the other rounds (trace only)."""
    waits = list(wait_seconds_per_round[: world_size - 1])
    round_number = (rank - 1 - accused) % world_size
    others = waits[:round_number] + waits[round_number + 1:]
    return {"round": round_number, "longest_wait_ms": milliseconds(waits[round_number]),
            "typical_wait_ms": milliseconds(sum(others) / len(others)) if others else None}
