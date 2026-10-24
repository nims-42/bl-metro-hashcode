import pytest

from src import messages as msg
from src.membership import ALIVE, FAILED, LEFT, SUSPECT, Membership

N = 6
IDS = [msg.node_name(i) for i in range(N)]
ME, OTHER = IDS[0], IDS[2]


def table():
    return Membership(ME, IDS, clock=lambda: 0.0)


def upd(kind, inc, about=OTHER, cls=None, src=None):
    return msg.gossip_update(kind, about, inc, src or IDS[1], cls)


def test_starts_alive_at_incarnation_zero():
    m = table()
    assert all(m.get(i).state == ALIVE and m.get(i).incarnation == 0 for i in IDS)


@pytest.mark.parametrize("held,inc,expect", [(0, 1, True), (1, 1, False), (2, 1, False)])
def test_alive_overrides_suspect_only_with_higher_incarnation(held, inc, expect):
    m = table()
    m.apply(upd(msg.SUSPECT, held))
    assert m.apply(upd(msg.ALIVE, inc)) is expect
    assert m.state(OTHER) == (ALIVE if expect else SUSPECT)


def test_alive_overrides_alive_only_with_higher_incarnation():
    m = table()
    assert m.apply(upd(msg.ALIVE, 0)) is False
    assert m.apply(upd(msg.ALIVE, 3)) is True
    assert m.get(OTHER).incarnation == 3


@pytest.mark.parametrize("held,inc,expect", [(0, 1, True), (1, 1, False), (2, 1, False)])
def test_suspect_overrides_suspect_only_with_higher_incarnation(held, inc, expect):
    m = table()
    m.apply(upd(msg.ALIVE, held))
    m.apply(upd(msg.SUSPECT, held))
    assert m.apply(upd(msg.SUSPECT, inc)) is expect


@pytest.mark.parametrize("held,inc,expect", [(1, 2, True), (1, 1, True), (1, 0, False)])
def test_suspect_overrides_alive_with_equal_or_higher_incarnation(held, inc, expect):
    m = table()
    m.apply(upd(msg.ALIVE, held))
    assert m.apply(upd(msg.SUSPECT, inc)) is expect
    assert m.state(OTHER) == (SUSPECT if expect else ALIVE)


@pytest.mark.parametrize("prior", [None, msg.ALIVE, msg.SUSPECT])
def test_confirm_overrides_alive_and_suspect_at_any_incarnation(prior):
    m = table()
    if prior:
        m.apply(upd(prior, 5))
    assert m.apply(upd(msg.CONFIRM, 0, cls="compute")) is True
    assert m.state(OTHER) == FAILED


def test_confirm_keeps_class_and_later_alive_cannot_undo_it():
    m = table()
    m.apply(upd(msg.CONFIRM, 0, cls="network"))
    assert m.get(OTHER).cls == "network"
    assert m.apply(upd(msg.ALIVE, 99)) is False
    assert m.apply(upd(msg.SUSPECT, 99)) is False
    assert m.state(OTHER) == FAILED and m.get(OTHER).cls == "network"


def test_confirm_fills_in_missing_class_only():
    m = table()
    m.apply(upd(msg.CONFIRM, 0))
    assert m.apply(upd(msg.CONFIRM, 0, cls="compute")) is True
    assert m.apply(upd(msg.CONFIRM, 0, cls="network")) is False
    assert m.get(OTHER).cls == "compute"


def test_failed_member_is_still_a_member():
    m = table()
    m.apply(upd(msg.CONFIRM, 0, cls="compute"))
    assert OTHER in m.ids()
    assert m.snapshot()[OTHER] == {"state": FAILED, "incarnation": 0, "class": "compute"}


def test_leave_sets_left_and_is_final():
    m = table()
    assert m.apply(upd(msg.LEAVE, 0)) is True
    assert m.state(OTHER) == LEFT
    assert m.apply(upd(msg.ALIVE, 10)) is False
    assert m.apply(upd(msg.CONFIRM, 10)) is False


def test_only_self_can_raise_own_incarnation():
    m = table()
    assert m.apply(upd(msg.ALIVE, 7, about=ME)) is False
    assert m.incarnation == 0
    m.apply(upd(msg.SUSPECT, 4, about=ME))
    assert m.incarnation == 0 and m.state(ME) == SUSPECT
    assert m.refute(4) == 5
    assert m.incarnation == 5 and m.state(ME) == ALIVE


def test_refuting_alive_overrides_the_suspicion_elsewhere():
    accused = Membership(OTHER, IDS, clock=lambda: 0.0)
    observer = table()
    observer.apply(upd(msg.SUSPECT, 0))
    new_inc = accused.refute(0)
    assert observer.apply(msg.gossip_update(msg.ALIVE, OTHER, new_inc, OTHER)) is True
    assert observer.state(OTHER) == ALIVE


def test_local_suspicion_can_be_cleared_without_gossip():
    m = table()
    assert m.mark_suspect_locally(OTHER, "compute") is True
    assert m.state(OTHER) == SUSPECT
    assert m.clear_local_suspicion(OTHER) is True
    assert m.state(OTHER) == ALIVE and m.get(OTHER).incarnation == 0


def test_updates_about_unknown_members_are_ignored():
    assert table().apply(upd(msg.SUSPECT, 0, about="gpu99")) is False
