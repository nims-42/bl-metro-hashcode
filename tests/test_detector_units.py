"""Unit tests for the small pieces of the detector: table, wait profile, health check, gossip."""

import pytest

from swim_straggler.detector.gossip import (
    CLEARED_VERDICT, STRAGGLER_VERDICT, SUSPICION, Notice, decode_gossip, encode_gossip, gossip_length,
)
from swim_straggler.detector.health import is_locally_healthy
from swim_straggler.detector.table import ALIVE, STRAGGLER, SUSPECT, MembershipTable, TableRow
from swim_straggler.detector.wait_profile import estimate_culprit
from synthetic_ring import waits_in_ring


# ---- wait profile -------------------------------------------------------------------------

@pytest.mark.parametrize("world_size", [3, 4, 8])
def test_every_worker_names_the_slow_worker(world_size):
    for slow_rank in range(world_size):
        compute = [0.02] * world_size
        compute[slow_rank] = 0.06
        waits = waits_in_ring(compute)
        for rank in range(world_size):
            expected = None if rank == slow_rank else slow_rank
            assert estimate_culprit(rank, world_size, waits[rank], 0.002) == expected


def test_flat_waits_name_nobody():
    assert estimate_culprit(1, 4, [0.01, 0.01, 0.01, 0.01, 0.01, 0.01], 0.002) is None


def test_small_differences_below_the_floor_are_noise():
    assert estimate_culprit(1, 4, [0.0005, 0.0015, 0.0005, 0, 0, 0], 0.002) is None


def test_the_formula_matches_the_real_measurement_from_milestone_1():
    """Real gloo run, slow worker 2: workers 3, 0, 1 had their longest wait in rounds 0, 1, 2."""
    assert estimate_culprit(3, 4, [0.1, 0.0, 0.0, 0, 0, 0], 0.002) == 2
    assert estimate_culprit(0, 4, [0.0, 0.1, 0.0, 0, 0, 0], 0.002) == 2
    assert estimate_culprit(1, 4, [0.0, 0.0, 0.1, 0, 0, 0], 0.002) == 2


def test_two_workers_cannot_be_told_apart():
    assert estimate_culprit(0, 2, [0.5, 0.5], 0.002) is None


# ---- health check -------------------------------------------------------------------------

def test_slow_worker_is_not_healthy_and_unknown_baseline_is_trusted():
    assert is_locally_healthy(0.02, 0.02, 1.5)
    assert is_locally_healthy(0.029, 0.02, 1.5)
    assert not is_locally_healthy(0.031, 0.02, 1.5)
    assert is_locally_healthy(0.5, None, 1.5)


# ---- table --------------------------------------------------------------------------------

def test_own_compute_is_smoothed():
    table = MembershipTable(3)
    table.update_own_compute(0, 0, 0.10, ewma_alpha=0.5)
    table.update_own_compute(0, 1, 0.20, ewma_alpha=0.5)
    assert table.rows[0].compute_seconds == pytest.approx(0.15)
    assert table.rows[0].step == 1


def test_only_newer_rows_replace_older_ones():
    table = MembershipTable(3)
    assert table.merge_row(TableRow(1, step=5, compute_seconds=0.1))
    assert not table.merge_row(TableRow(1, step=5, compute_seconds=0.9))
    assert not table.merge_row(TableRow(1, step=4, compute_seconds=0.9))
    assert table.merge_row(TableRow(1, step=6, compute_seconds=0.2))
    assert table.rows[1].compute_seconds == 0.2


def fill(table, compute_by_rank):
    for rank, seconds in enumerate(compute_by_rank):
        table.merge_row(TableRow(rank, step=1, compute_seconds=seconds))


def test_the_subject_is_left_out_of_the_reference_median():
    table = MembershipTable(5)
    fill(table, [0.02, 0.02, 0.02, 0.10, 0.10])
    # With the subject (rank 3) left out the reference is median(0.02, 0.02, 0.02, 0.10) = 0.02.
    assert table.reference_median(3) == pytest.approx(0.02)
    assert table.looks_slow(3, factor=1.3)
    assert not table.looks_slow(0, factor=1.3)


def test_suspected_workers_are_left_out_of_the_reference_too():
    table = MembershipTable(6)
    fill(table, [0.02, 0.02, 0.10, 0.10, 0.10, 0.02])
    table.states[3] = SUSPECT
    table.states[4] = STRAGGLER
    assert table.reference_median(2) == pytest.approx(0.02)


def test_nothing_looks_slow_without_enough_information():
    table = MembershipTable(4)
    table.merge_row(TableRow(1, step=1, compute_seconds=0.5))
    assert not table.looks_slow(1, factor=1.3)


# ---- gossip encoding ----------------------------------------------------------------------

def test_gossip_round_trip():
    rows = [TableRow(2, step=7, compute_seconds=0.0213, probe_subject=4, probe_answer_step=6, probe_says_slow=True),
            TableRow(0, step=8, compute_seconds=0.019)]
    notices = [Notice(STRAGGLER_VERDICT, 4, 9), Notice(SUSPICION, 5, 10), Notice(CLEARED_VERDICT, 1, 3)]
    numbers = encode_gossip(rows, notices, gossip_rows=4, notice_slots=4)
    assert len(numbers) == gossip_length(4, 4)
    decoded_rows, decoded_notices = decode_gossip(numbers, 4, 4)
    assert [(row.rank, row.step, row.probe_subject, row.probe_answer_step, row.probe_says_slow) for row in decoded_rows] == [
        (2, 7, 4, 6, True), (0, 8, -1, -1, False)]
    assert decoded_rows[0].compute_seconds == pytest.approx(0.0213)
    assert decoded_notices == notices


def test_gossip_survives_float32_encoding():
    """The header stores numbers as float32; ranks and steps must come back exact."""
    import torch

    rows = [TableRow(7, step=123456, compute_seconds=0.0213, probe_subject=3, probe_answer_step=123455, probe_says_slow=True)]
    numbers = encode_gossip(rows, [Notice(SUSPICION, 3, 123456)], 2, 2)
    through_float32 = torch.tensor(numbers, dtype=torch.float32).tolist()
    decoded_rows, decoded_notices = decode_gossip(through_float32, 2, 2)
    assert (decoded_rows[0].rank, decoded_rows[0].step, decoded_rows[0].probe_subject) == (7, 123456, 3)
    assert decoded_notices == [Notice(SUSPICION, 3, 123456)]


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        decode_gossip([0.0] * 5, 4, 4)
