"""Protocol tests for StragglerDetector: voting, rejection, expiry, health. Rows are fed in by hand."""

import ast
import pathlib
import subprocess
import sys

from swim_straggler.config import DetectorConfig
from swim_straggler.detector.gossip import STRAGGLER_VERDICT, SUSPICION, Notice, encode_gossip
from swim_straggler.detector.swim import Probe, StragglerDetector
from swim_straggler.detector.table import ALIVE, STRAGGLER, SUSPECT, TableRow

WORLD_SIZE = 6
CONFIG = DetectorConfig(consecutive_steps=3, warmup_steps=0, k_probes=3, verdict_repeat_steps=5, recovery_steps=4)


def waits_pointing_at(rank, culprit):
    """A wait profile whose longest wait says `culprit` is slow, as seen by `rank`."""
    profile = [0.0] * (2 * (WORLD_SIZE - 1))
    profile[(rank - 1 - culprit) % WORLD_SIZE] = 0.05
    return profile


def give_row(detector, rank, step, compute_seconds, **answer):
    detector.receive_gossip(encode_gossip([TableRow(rank, step=step, compute_seconds=compute_seconds, **answer)], [],
                                          CONFIG.gossip_rows, CONFIG.verdict_slots))


def prepare(rank=0, slow_rank=3, steps=0):
    """A detector at the start of `steps`, knowing everybody's compute time (worker `slow_rank` is slow)."""
    detector = StragglerDetector(rank, WORLD_SIZE, CONFIG, seed=1)
    for step in range(steps + 1):
        detector.start_step(step, 0.02)
        for other in range(WORLD_SIZE):
            if other != rank:
                give_row(detector, other, step, 0.05 if other == slow_rank else 0.02)
    return detector


def accuse_until_suspected(detector, culprit):
    step = 10
    while not detector.suspect_ranks():
        detector.start_step(step, 0.02)
        for other in range(WORLD_SIZE):
            if other != detector.rank:
                give_row(detector, other, step, 0.05 if other == culprit else 0.02)
        detector.finish_step(step, waits_pointing_at(detector.rank, culprit))
        step += 1
        assert step < 30
    return step


def answer(detector, peer, subject, step, says_slow):
    give_row(detector, peer, step, 0.02, probe_subject=subject, probe_answer_step=step, probe_says_slow=says_slow)


def test_suspicion_needs_consecutive_steps():
    detector = prepare()
    for step in (10, 11):
        detector.start_step(step, 0.02)
        detector.finish_step(step, waits_pointing_at(0, 3))
    assert detector.suspect_ranks() == []
    detector.start_step(12, 0.02)
    events = detector.finish_step(12, waits_pointing_at(0, 3))
    assert detector.suspect_ranks() == [3]
    assert events[0]["action"] == "suspected" and len(events[0]["consulted_peers"]) == 3
    assert 0 not in events[0]["consulted_peers"] and 3 not in events[0]["consulted_peers"]


def test_a_gap_resets_the_streak():
    detector = prepare()
    for step, culprit in ((10, 3), (11, 3), (12, None), (13, 3), (14, 3)):
        detector.start_step(step, 0.02)
        detector.finish_step(step, waits_pointing_at(0, culprit) if culprit is not None else [0.0] * 10)
    assert detector.suspect_ranks() == []


def test_majority_of_consulted_peers_confirms():
    detector = prepare()
    step = accuse_until_suspected(detector, 3)
    peers = detector.probes[3].peers
    answer(detector, peers[0], 3, step, True)
    answer(detector, peers[1], 3, step, True)
    detector.start_step(step, 0.02)
    events = detector.finish_step(step, waits_pointing_at(0, 3))
    assert [event["action"] for event in events] == ["confirmed"]
    assert detector.table.states[3] == STRAGGLER
    assert detector.verdict_notices[3].kind == STRAGGLER_VERDICT


def test_majority_against_rejects_and_returns_to_alive():
    detector = prepare()
    step = accuse_until_suspected(detector, 3)
    peers = detector.probes[3].peers
    answer(detector, peers[0], 3, step, False)
    answer(detector, peers[1], 3, step, False)
    detector.start_step(step, 0.02)
    events = detector.finish_step(step, waits_pointing_at(0, 3))
    assert events[0]["action"] == "rejected" and events[0]["reason"] == "majority disagrees"
    assert detector.table.states[3] == ALIVE


def test_no_answers_means_rejection_after_the_timeout_not_a_confirmation():
    detector = prepare()
    step = accuse_until_suspected(detector, 3)
    actions = []
    for later in range(step, step + CONFIG.verdict_repeat_steps + 1):
        detector.start_step(later, 0.02)
        actions += [event["action"] for event in detector.finish_step(later, waits_pointing_at(0, 3))]
    assert actions == ["rejected"]


def test_answers_from_before_the_suspicion_do_not_count():
    detector = prepare()
    step = accuse_until_suspected(detector, 3)
    peers = detector.probes[3].peers
    for peer in peers:
        answer(detector, peer, 3, step - 5, True)  # old answers
    detector.start_step(step, 0.02)
    assert detector.finish_step(step, waits_pointing_at(0, 3)) == []


def test_a_rejected_suspicion_is_not_repeated_immediately():
    detector = prepare()
    step = accuse_until_suspected(detector, 3)
    for peer in detector.probes[3].peers:
        answer(detector, peer, 3, step, False)
    detector.start_step(step, 0.02)
    detector.finish_step(step, waits_pointing_at(0, 3))
    for later in range(step + 1, step + 4):
        detector.start_step(later, 0.02)
        detector.finish_step(later, waits_pointing_at(0, 3))
    assert detector.suspect_ranks() == []


def test_a_slow_worker_does_not_accuse_anybody():
    detector = prepare(rank=0, slow_rank=0)
    for step in range(10, 20):
        detector.start_step(step, 0.05)  # this worker itself computes 2.5x slower than the others
        for other in range(1, WORLD_SIZE):
            give_row(detector, other, step, 0.02)
        detector.finish_step(step, waits_pointing_at(0, 3))
    assert detector.suspect_ranks() == []


def test_peers_answer_a_suspicion_from_their_own_table():
    peer = prepare(rank=1, slow_rank=3)
    peer.receive_gossip(encode_gossip([], [Notice(SUSPICION, 3, 10)], CONFIG.gossip_rows, CONFIG.verdict_slots))
    peer.start_step(10, 0.02)
    peer.finish_step(10, [0.0] * 10)
    row = peer.table.rows[1]
    assert (row.probe_subject, row.probe_says_slow, row.probe_answer_step) == (3, True, 10)

    healthy_subject = prepare(rank=1, slow_rank=4)  # worker 4 is the slow one, so worker 3 looks fine
    healthy_subject.receive_gossip(encode_gossip([], [Notice(SUSPICION, 3, 10)], CONFIG.gossip_rows, CONFIG.verdict_slots))
    healthy_subject.start_step(10, 0.02)
    healthy_subject.finish_step(10, [0.0] * 10)
    assert healthy_subject.table.rows[1].probe_says_slow is False


def test_verdicts_are_learned_passed_on_and_expire():
    detector = prepare(rank=2)
    detector.receive_gossip(encode_gossip([], [Notice(STRAGGLER_VERDICT, 3, 10)], CONFIG.gossip_rows, CONFIG.verdict_slots))
    assert detector.table.states[3] == STRAGGLER
    assert 3 in detector.verdict_notices
    for step in range(1, 20):
        detector.start_step(step, 0.02)
        detector.finish_step(step, [0.0] * 10)
    assert detector.verdict_notices == {}  # no longer passed on after verdict_repeat_steps
    assert detector.table.states[3] == STRAGGLER  # but still remembered

    detector.receive_gossip(encode_gossip([], [Notice(STRAGGLER_VERDICT, 4, 2)], CONFIG.gossip_rows, CONFIG.verdict_slots))
    assert detector.table.states[4] == ALIVE  # an old notice arriving late is ignored


def test_a_worker_never_accepts_gossip_about_its_own_row():
    detector = prepare(rank=2)
    give_row(detector, 2, 99, 9.9)
    assert detector.table.rows[2].compute_seconds < 1


def test_gossip_has_a_constant_size_however_large_the_table():
    sizes = set()
    for world_size in (4, 16, 64):
        detector = StragglerDetector(0, world_size, CONFIG)
        detector.start_step(0, 0.02)
        for other in range(1, world_size):
            detector.table.rows[other] = TableRow(other, step=0, compute_seconds=0.02)
        sizes.add(len(detector.make_gossip()))
    assert len(sizes) == 1


def test_rows_rotate_so_that_every_worker_is_eventually_sent():
    world_size = 20
    detector = StragglerDetector(0, world_size, CONFIG)
    detector.start_step(0, 0.02)
    for other in range(1, world_size):
        detector.table.rows[other] = TableRow(other, step=0, compute_seconds=0.02)
    seen = set()
    for _ in range(10):
        seen |= {row.rank for row in detector.rows_to_send()}
    assert seen == set(range(world_size))


def test_during_warmup_nothing_happens():
    config = DetectorConfig(consecutive_steps=1, warmup_steps=10)
    detector = StragglerDetector(0, WORLD_SIZE, config)
    for step in range(10):
        detector.start_step(step, 0.02)
        assert detector.finish_step(step, waits_pointing_at(0, 3)) == []


# ---- purity -------------------------------------------------------------------------------

DETECTOR_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent / "swim_straggler" / "detector"
ALLOWED_MODULES = {"dataclasses", "random", "statistics", "swim_straggler"}


def test_detector_source_imports_only_plain_python():
    for source_file in DETECTOR_DIRECTORY.glob("*.py"):
        for node in ast.walk(ast.parse(source_file.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module]
            for name in names:
                assert name.split(".")[0] in ALLOWED_MODULES, f"{source_file.name} imports {name}"


def test_importing_the_detector_does_not_load_torch_or_networking():
    code = (
        "import sys, swim_straggler.detector.swim;"
        "bad = [m for m in ('torch', 'numpy', 'socket', 'multiprocessing', 'threading') if m in sys.modules];"
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=DETECTOR_DIRECTORY.parent.parent)
    assert result.returncode == 0, result.stdout + result.stderr


def test_experimental_network_diagnosis_after_repeated_rejections_of_normal_workers():
    from swim_straggler.config import DetectorConfig as Settings

    detector = StragglerDetector(0, 6, Settings(diagnose_network=True, diagnosis_rejections=2, diagnosis_window_steps=50))
    for row_rank in range(6):
        detector.table.rows[row_rank].step = 1
        detector.table.rows[row_rank].compute_seconds = 0.02  # everybody computes normally
    events = []
    for step in (20, 30):
        detector.probes[3] = Probe(start_step=step, peers=[1, 2])
        detector.reject(3, step, "majority disagrees", 0, 2)
        events += detector.finish_step(step, [0.0] * 10)
    diagnoses = [event for event in events if event["action"] == "delay_not_compute"]
    assert len(diagnoses) == 1 and diagnoses[0]["step"] == 30
    rejected = [event for event in events if event["action"] == "rejected"]
    assert rejected[0]["compute_ratio"] == 1.0

    quiet = StragglerDetector(0, 6, Settings())  # switched off by default
    quiet.probes[3] = Probe(start_step=20, peers=[1, 2])
    quiet.reject(3, 20, "majority disagrees", 0, 2)
    quiet.probes[3] = Probe(start_step=30, peers=[1, 2])
    quiet.reject(3, 30, "majority disagrees", 0, 2)
    assert not [event for event in quiet.finish_step(30, [0.0] * 10) if event["action"] == "delay_not_compute"]
