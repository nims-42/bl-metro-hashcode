"""Member table and SWIM precedence rules (CONTEXT.md section 6.4).

Pure state, no I/O. `Failed` means "confirmed straggler" (fail-slow): the node
stays in the ring and keeps training, and its `cls` says compute or network.
`Left` is a clean, voluntary exit.
"""

import time
from dataclasses import dataclass

ALIVE = "Alive"
SUSPECT = "Suspect"
FAILED = "Failed"
LEFT = "Left"


@dataclass
class Member:
    state: str = ALIVE
    incarnation: int = 0
    cls: str | None = None
    last_update: float = 0.0


class Membership:
    def __init__(self, self_id, member_ids, clock=time.monotonic):
        self.self_id = self_id
        self._clock = clock
        now = clock()
        self._members = {m: Member(last_update=now) for m in member_ids}
        if self_id not in self._members:
            self._members[self_id] = Member(last_update=now)

    @property
    def incarnation(self):
        return self._members[self.self_id].incarnation

    def get(self, member_id):
        return self._members[member_id]

    def ids(self):
        return list(self._members)

    def state(self, member_id):
        return self._members[member_id].state

    def snapshot(self):
        """{node_id: {state, incarnation, class}} for the metrics log."""
        return {m: {"state": v.state, "incarnation": v.incarnation, "class": v.cls}
                for m, v in self._members.items()}

    def apply(self, update):
        """Apply one gossip update. Returns True if the local table changed."""
        about = update["about"]
        if about not in self._members:
            return False
        member = self._members[about]
        kind = update["type"]
        inc = int(update["incarnation"])

        if member.state == LEFT:
            return False
        if kind == "LEAVE":
            return self._set(member, LEFT, member.incarnation, member.cls)
        if member.state == FAILED:
            # CONFIRM overrides ALIVE and SUSPECT at any incarnation, so a
            # confirmed straggler stays confirmed. A later CONFIRM may only
            # fill in a class that was missing.
            if kind == "CONFIRM" and member.cls is None and update.get("class"):
                member.cls = update["class"]
                member.last_update = self._clock()
                return True
            return False
        if kind == "CONFIRM":
            return self._set(member, FAILED, max(inc, member.incarnation), update.get("class"))

        # Only the node itself may raise its own incarnation (see refute()).
        if about == self.self_id and inc > member.incarnation:
            if kind == "ALIVE":
                return False
            inc = member.incarnation

        if kind == "ALIVE":
            if inc > member.incarnation:
                return self._set(member, ALIVE, inc, None)
            return False
        if kind == "SUSPECT":
            if member.state == SUSPECT and inc > member.incarnation:
                return self._set(member, SUSPECT, inc, update.get("class"))
            if member.state == ALIVE and inc >= member.incarnation:
                return self._set(member, SUSPECT, inc, update.get("class"))
            return False
        raise ValueError(f"unknown update type {kind!r}")

    def refute(self, seen_incarnation):
        """Refute a suspicion about this node: raise its own incarnation past
        the suspected one and return to Alive. Returns the new incarnation."""
        me = self._members[self.self_id]
        me.incarnation = max(me.incarnation, int(seen_incarnation)) + 1
        me.state = ALIVE
        me.cls = None
        me.last_update = self._clock()
        return me.incarnation

    def mark_suspect_locally(self, member_id, cls=None):
        """Raise this node's own suspicion (not received through gossip)."""
        member = self._members[member_id]
        if member.state != ALIVE:
            return False
        return self._set(member, SUSPECT, member.incarnation, cls)

    def clear_local_suspicion(self, member_id):
        """Drop a suspicion this node raised but never gossiped (peers
        disagreed). Received suspicions are only cleared by a higher ALIVE."""
        member = self._members[member_id]
        if member.state != SUSPECT:
            return False
        return self._set(member, ALIVE, member.incarnation, None)

    def _set(self, member, state, incarnation, cls):
        member.state = state
        member.incarnation = incarnation
        member.cls = cls
        member.last_update = self._clock()
        return True
