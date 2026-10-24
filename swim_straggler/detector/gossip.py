"""Turning table rows and notices into a flat list of numbers, and back.

The numbers travel in the header of every ring message. The size is fixed by the
configuration, so the detector adds the same small load to every message however large the
group is (the "constant load" property of SWIM).

A row is 6 numbers: rank, step, compute_seconds, probe_subject, probe_answer_step, probe_says_slow.
A notice is 3 numbers: kind, subject, step. An unused row has rank -1, an unused notice kind 0.
"""

from dataclasses import dataclass

from swim_straggler.detector.table import TableRow

NUMBERS_PER_ROW = 6
NUMBERS_PER_NOTICE = 3

SUSPICION = "suspicion"  # "I suspect this worker; please check and answer"
STRAGGLER_VERDICT = "straggler"  # "this worker is confirmed slow"
CLEARED_VERDICT = "cleared"  # "this worker is healthy again"
NOTICE_CODES = {SUSPICION: 1, STRAGGLER_VERDICT: 2, CLEARED_VERDICT: 3}
NOTICE_KINDS = {code: kind for kind, code in NOTICE_CODES.items()}


@dataclass
class Notice:
    kind: str
    subject: int
    step: int  # the step at which the notice was created


def gossip_length(gossip_rows, notice_slots):
    return gossip_rows * NUMBERS_PER_ROW + notice_slots * NUMBERS_PER_NOTICE


def encode_gossip(rows, notices, gossip_rows, notice_slots):
    """Flatten up to `gossip_rows` rows and `notice_slots` notices; unused space is marked empty."""
    numbers = []
    for position in range(gossip_rows):
        if position < len(rows):
            row = rows[position]
            numbers += [row.rank, row.step, row.compute_seconds, row.probe_subject,
                        row.probe_answer_step, 1.0 if row.probe_says_slow else 0.0]
        else:
            numbers += [-1.0] * NUMBERS_PER_ROW
    for position in range(notice_slots):
        if position < len(notices):
            notice = notices[position]
            numbers += [NOTICE_CODES[notice.kind], notice.subject, notice.step]
        else:
            numbers += [0.0] * NUMBERS_PER_NOTICE
    return numbers


def decode_gossip(numbers, gossip_rows, notice_slots):
    """The reverse of encode_gossip. Returns (rows, notices)."""
    if len(numbers) != gossip_length(gossip_rows, notice_slots):
        raise ValueError("gossip has the wrong length")
    rows = []
    for position in range(gossip_rows):
        start = position * NUMBERS_PER_ROW
        rank, step, compute_seconds, probe_subject, probe_answer_step, probe_says_slow = numbers[start : start + NUMBERS_PER_ROW]
        if round(rank) < 0:
            continue
        rows.append(TableRow(
            rank=round(rank),
            step=round(step),
            compute_seconds=compute_seconds,
            probe_subject=round(probe_subject),
            probe_answer_step=round(probe_answer_step),
            probe_says_slow=probe_says_slow > 0.5,
        ))
    notices = []
    notice_start = gossip_rows * NUMBERS_PER_ROW
    for position in range(notice_slots):
        start = notice_start + position * NUMBERS_PER_NOTICE
        code, subject, step = numbers[start : start + NUMBERS_PER_NOTICE]
        if round(code) in NOTICE_KINDS:
            notices.append(Notice(NOTICE_KINDS[round(code)], round(subject), round(step)))
    return rows, notices
