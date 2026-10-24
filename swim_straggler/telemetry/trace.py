"""trace.log: everything that happened in a run, step by step, in plain sentences.

Built from the merged event log (events.jsonl), so it can be rebuilt for any finished run.
Without detector.trace the log holds only the decisions (suspected, confirmed, ...); with it,
also the reasoning behind them (whom each worker's waits point to, every answer, every vote).

Ordering: lines are ordered by training step, and within a step by worker. Worker clocks are never
compared, so the order between two workers inside one step says nothing about which came first.
"""

from pathlib import Path

from swim_straggler.telemetry.events import read_events, summarize_detection

# The belief a worker holds about another one after each decision.
STATE_AFTER_ACTION = {
    "suspected": "suspect",
    "rejected": "alive",
    "confirmed": "straggler",
    "cleared": "alive",
    "learned_straggler": "straggler",
    "learned_cleared": "alive",
}


def worker_name(rank):
    return f"w{rank}"


def workers_text(ranks):
    return ", ".join(worker_name(rank) for rank in ranks) or "nobody"


def number(value, digits=1):
    return "?" if value is None else f"{value:.{digits}f}"


def describe(event, recovery_steps):
    """One sentence for a detector decision or a trace event."""
    action = event["action"]
    subject = worker_name(event["subject"])
    decided_by = event.get("decided_by")

    # decisions
    if action == "suspected":
        return f"SUSPECTS {subject}, asks {workers_text(event.get('consulted_peers', []))} to confirm"
    if action == "confirmed":
        if decided_by == "own step time":
            return "CONFIRMS ITSELF as slow (its own step time; it cannot tell who is really slow)"
        if decided_by == "central monitor":
            return f"CONFIRMS {subject} is a straggler (central monitor, alone)"
        ratio = f", computes x{event['compute_ratio']:.2f} the typical" if event.get("compute_ratio") else ""
        return f"CONFIRMS {subject} is a straggler ({event['votes_for']} of {event['consulted']} consulted agree{ratio})"
    if action == "rejected":
        return (f"drops its suspicion of {subject}: {event['reason']} "
                f"({event['votes_for']} of {event['consulted']} agree)")
    if action == "cleared":
        if decided_by == "own step time":
            return f"clears itself: its step time looked normal for {recovery_steps} steps"
        return f"CLEARS {subject}: looked normal for {recovery_steps} steps"
    if action == "delay_not_compute":
        return (f"DIAGNOSIS (experimental): {event['rejections']} suspicions in {event['window_steps']} steps were rejected "
                "because the accused computed normally: the delay is probably not compute, possibly the network")
    if action in ("learned_straggler", "learned_cleared"):
        verdict = "a straggler" if action == "learned_straggler" else "cleared"
        via = f" via gossip from {worker_name(event['via'])}" if event.get("via") is not None else " via gossip"
        decided = f", decided at step {event['decided_at_step']}" if event.get("decided_at_step") is not None else ""
        return f"learns {subject} is {verdict}{via}{decided}"

    # trace events: SWIM
    if action == "points_to":
        return (f"waits point to {subject} (round {event['round']} waited {number(event['longest_wait_ms'], 2)} ms, "
                f"other rounds {number(event['typical_wait_ms'], 2)} ms): {event['streak']} of {event['needed']} steps")
    if action == "stopped_pointing":
        return f"stops counting {subject} after {event['after_steps']} step(s): {event['reason']}"
    if action == "distrusts_own_waits":
        return (f"is slow itself (computes {number(event['own_compute_ms'])} ms, typical {number(event['typical_compute_ms'])} ms, "
                f"limit x{event['health_factor']}), so it accuses nobody and answers 'not slow'")
    if action == "trusts_own_waits_again":
        return f"computes normally again ({number(event['own_compute_ms'])} ms), trusts its own waits again"
    if action == "waits_after_rejection":
        return (f"would suspect {subject}, but that suspicion was rejected at step {event['rejected_at_step']}; "
                f"waits until step {event['may_suspect_again_at_step']}")
    if action == "heard_suspicion":
        via = f" from {worker_name(event['via'])}" if event.get("via") is not None else ""
        return f"hears{via} that {subject} is suspected (since step {event['suspected_at_step']})"
    if action == "answers":
        if not event["trusts_itself"]:
            return f"answers about {subject}: not slow (it does not trust its own view while slow itself)"
        verdict = "SLOW" if event["says_slow"] else "not slow"
        return (f"answers about {subject}: {verdict} ({subject} computes {number(event['subject_compute_ms'])} ms, "
                f"typical {number(event['typical_compute_ms'])} ms, limit x{event['factor']})")
    if action == "got_answer":
        verdict = "slow" if event["says_slow"] else "not slow"
        return f"hears {worker_name(event['peer'])}'s answer about {subject}: {verdict} (given at step {event['answered_at_step']})"
    if action == "no_recent_news":
        return f"has no recent news of {subject} (last heard at step {event['last_heard_step']}), cannot judge its recovery"
    if action == "recovery_restarted":
        return f"sees {subject} slow again after {event['normal_steps_before']} normal step(s); recovery count starts over"

    # trace events: baselines
    if action == "looks_slow":
        return (f"(central monitor) {subject} computes {number(event['compute_ms'])} ms, typical "
                f"{number(event['typical_compute_ms'])} ms (limit x{event['factor']}): {event['streak']} of {event['needed']} steps")
    if action == "looks_normal_again":
        return f"(central monitor) {subject} looks normal again after {event['after_steps']} slow step(s)"
    if action == "learned_normal_step":
        return f"learned its normal step time: {number(event['normal_step_ms'])} ms"
    if action == "own_step_slow":
        return (f"own step {number(event['step_ms'])} ms, normal {number(event['normal_step_ms'])} ms "
                f"(limit x{event['factor']}): {event['streak']} of {event['needed']} steps")
    if action == "own_step_normal_again":
        return f"own step time normal again after {event['after_steps']} slow step(s)"
    return f"{action} {subject}"


def fault_text(fault):
    kind = fault["kind"]
    if kind in ("compute", "two", "intermittent", "spike"):
        return f"compute slowed down x{fault['multiplier']}" + (" (switches on and off)" if kind == "intermittent" else "")
    if kind == "link":
        return f"every message delayed by {fault['link_delay_seconds'] * 1000:.1f} ms"
    if kind == "container_cpu":
        return f"container processor share cut to 1/{fault['multiplier']}"
    if kind == "container_stress":
        return f"{fault['stress_workers']} processor-hungry programs started next to it"
    if kind == "container_network_delay":
        return f"every outgoing packet delayed by {fault['network_delay_ms']} ms"
    if kind == "container_pause":
        return f"container frozen for {fault['pause_seconds']} s"
    return kind


def fault_lines_by_step(events, fault):
    """{step: [(worker, text)]} for the fault: worker-side faults from the step events, container faults from the controller."""
    lines = {}

    def add(step, worker, text):
        lines.setdefault(step, []).append((worker, "FAULT " + text))

    active = {}
    for event in sorted((event for event in events if event["event"] == "step"), key=lambda event: (event["rank"], event["step"])):
        was_active = active.get(event["rank"], False)
        if event.get("fault_active") and not was_active:
            add(event["step"], event["rank"], f"{worker_name(event['rank'])}: {fault_text(fault)} (active from this step)")
        elif was_active and not event.get("fault_active"):
            add(event["step"], event["rank"], f"{worker_name(event['rank'])}: fault no longer active")
        active[event["rank"]] = bool(event.get("fault_active"))

    for event in events:
        if event.get("source") != "controller":
            continue
        name = worker_name(event["rank"])
        if event["event"] == "fault_started":
            tool = event.get("tool", "docker")
            add(event["step"], event["rank"], f"{name}: {fault_text(fault)}: {tool} command issued "
                                              f"(training was at step {event['step']})")
            add(event.get("step_after_command", event["step"]), event["rank"],
                f"{name}: fault command returned, the fault is in place")
        elif event["event"] == "fault_ended":
            add(event["step"], event["rank"], f"{name}: fault removed")
    return lines


def parse_step_range(text):
    """"60-90" -> (60, 90); "60" -> (60, 60); "60-" -> (60, None); None -> (None, None)."""
    if not text:
        return None, None
    first, _, last = text.partition("-")
    if not _:
        return int(first), int(first)
    return (int(first) if first else None), (int(last) if last else None)


def trace_lines(events, worker=None, about=None, first_step=None, last_step=None):
    """The lines of trace.log. `worker`: only what this worker did; `about`: only what concerns this worker."""
    run_start = next((event for event in events if event["event"] == "run_start"), None)
    config = run_start["config"] if run_start else None
    fault = config["fault"] if config else {"kind": "none"}
    detector = config["detector"] if config else {}
    recovery_steps = detector.get("recovery_steps", "?")
    world_size = run_start["world_size"] if run_start else len({event["rank"] for event in events if event["event"] == "step"})
    has_detail = any(event["event"] == "trace" for event in events)
    filtering = worker is not None or about is not None
    simulated = bool(run_start and run_start.get("simulated"))

    lines = []
    if simulated:
        lines.append("SIMULATION: the timings come from a random model, not from measurement; "
                     "the detector decisions are those of the real detector code (see docs/simulation.md)")
    if config:
        fault_part = "no fault" if fault["kind"] == "none" else (
            f"fault {fault['kind']} on {workers_text(fault['workers'])} from step {fault['start_step']} ({fault_text(fault)})")
        lines += [
            f"trace of '{config['name']}': {world_size} workers, backend {config['cluster']['backend']}, {fault_part}",
            f"detector {detector['mode']}: factor x{detector['factor']}, {detector['consecutive_steps']} steps in a row, "
            f"{detector['k_probes']} peers consulted, recovery after {recovery_steps} steps, first {detector['warmup_steps']} steps ignored",
        ]
    lines += [
        "detail: " + ("full (detector.trace on: the reasoning behind every decision)" if has_detail
                      else "decisions only (run with --trace to also see the reasoning behind them)"),
        "Steps are training steps. Times are " + ("simulated" if simulated else "each worker's own measurements")
        + ", in milliseconds.",
        "Within a step, lines are grouped by worker: the order between workers inside one step means nothing.",
    ]
    if filtering:
        lines.append(f"filter: {'only what ' + worker_name(worker) + ' did' if worker is not None else ''}"
                     f"{'; ' if worker is not None and about is not None else ''}"
                     f"{'only what concerns ' + worker_name(about) if about is not None else ''}")
    lines.append("")

    step_events = {}
    decision_events = {}
    for event in events:
        if event["event"] == "step":
            step_events.setdefault(event["step"], {})[event["rank"]] = event
        elif event["event"] in ("detector", "trace"):
            decision_events.setdefault(event["step"], []).append(event)
    fault_lines = fault_lines_by_step(events, fault)

    beliefs = {rank: {} for rank in range(world_size)}
    last_beliefs_text = NOBODY_SUSPECTED
    printed_a_step = False
    all_steps = sorted(set(step_events) | set(decision_events) | set(fault_lines))
    for step in all_steps:
        happenings = []
        for fault_worker, text in fault_lines.get(step, []):
            if (worker is None or worker == fault_worker) and (about is None or about == fault_worker):
                happenings.append("  " + text)
        for event in sorted(decision_events.get(step, []), key=lambda event: event["rank"]):  # keeps each worker's order
            if event["event"] == "detector" and event["action"] in STATE_AFTER_ACTION:
                beliefs.setdefault(event["rank"], {})[event["subject"]] = STATE_AFTER_ACTION[event["action"]]
            concerns_about = about is None or event["subject"] == about or event.get("peer") == about
            if (worker is None or event["rank"] == worker) and concerns_about:
                happenings.append(f"  {worker_name(event['rank'])}: {describe(event, recovery_steps)}")

        current_beliefs = beliefs_text(beliefs, worker, about)
        beliefs_changed = current_beliefs != last_beliefs_text
        last_beliefs_text = current_beliefs

        in_range = (first_step is None or step >= first_step) and (last_step is None or step <= last_step)
        if not in_range or (filtering and not happenings):
            continue

        per_worker = step_events.get(step, {})
        header = f"step {step}"
        if per_worker:
            step_ms = max(event["step_seconds"] for event in per_worker.values()) * 1000
            header += f"   step {step_ms:.1f} ms"
            if 0 in per_worker and per_worker[0].get("loss") is not None:
                header += f"   loss {per_worker[0]['loss']:.3f}"
        lines.append(header)
        shown_workers = sorted(per_worker) if worker is None else [rank for rank in sorted(per_worker) if rank == worker]
        if len(shown_workers) > MOST_WORKERS_LISTED:
            lines += compact_measurement_lines(per_worker)
        elif shown_workers:
            lines.append("  compute ms : " + "  ".join(
                f"{worker_name(rank)} {per_worker[rank]['compute_seconds'] * 1000:6.1f}" for rank in shown_workers))
            lines.append("  wait ms    : " + "  ".join(
                f"{worker_name(rank)} {per_worker[rank]['wait_seconds'] * 1000:6.1f}" for rank in shown_workers))
        lines += happenings
        if beliefs_changed or (not printed_a_step and current_beliefs != NOBODY_SUSPECTED):
            lines.append(f"  beliefs now: {current_beliefs}")
        printed_a_step = True

    lines += [""] + summary_lines(events)
    return lines


NOBODY_SUSPECTED = "nobody suspects anybody"
MOST_WORKERS_LISTED = 16  # with more workers, the measurement lines show a summary instead of every worker


def compact_measurement_lines(per_worker):
    """For large groups: the median, and the most unusual worker, instead of every worker."""
    import statistics

    computes = {rank: event["compute_seconds"] * 1000 for rank, event in per_worker.items()}
    waits = {rank: event["wait_seconds"] * 1000 for rank, event in per_worker.items()}
    slowest = max(computes, key=computes.get)
    least_waiting = min(waits, key=waits.get)
    return [f"  compute ms : median {statistics.median(computes.values()):.1f}, "
            f"slowest {worker_name(slowest)} {computes[slowest]:.1f}",
            f"  wait ms    : median {statistics.median(waits.values()):.1f}, "
            f"least {worker_name(least_waiting)} {waits[least_waiting]:.1f}"]


def ranges_text(ranks):
    """[0, 1, 2, 5, 7, 8] -> "w0-w2, w5, w7-w8"."""
    parts = []
    ranks = sorted(ranks)
    start = previous = ranks[0]
    for rank in ranks[1:] + [None]:
        if rank is not None and rank == previous + 1:
            previous = rank
            continue
        parts.append(worker_name(start) if start == previous else f"{worker_name(start)}-{worker_name(previous)}")
        if rank is not None:
            start = previous = rank
    return ", ".join(parts)


def beliefs_text(beliefs, worker=None, about=None):
    """Who thinks whom is suspect or a straggler, for example "w0 thinks w2 straggler; w3 thinks w2 suspect"."""
    believers_of = {}  # the same opinion held by several workers is written once
    for believer in sorted(beliefs):
        if worker is not None and believer != worker:
            continue
        held = tuple((subject, state) for subject, state in sorted(beliefs[believer].items())
                     if state != "alive" and (about is None or subject == about))
        if held:
            believers_of.setdefault(held, []).append(believer)
    parts = [f"{ranges_text(believers)} {'thinks' if len(believers) == 1 else 'think'} "
             + ", ".join(f"{worker_name(subject)} {state}" for subject, state in held)
             for held, believers in believers_of.items()]
    return "; ".join(parts) or NOBODY_SUSPECTED


def summary_lines(events):
    """What the detectors concluded, compared with the injected fault."""
    summary = summarize_detection(events)
    lines = ["summary"]
    if not summary["faulty_workers"]:
        lines.append("  no fault was injected: every confirmation is wrong")
    for culprit, report in summary["per_fault"].items():
        lines.append(f"  fault on {worker_name(culprit)}: issued at step {report['fault_issued_step']}, "
                     f"first suspected at step {report['first_suspected_step']}, "
                     f"first confirmed at step {report['first_confirmed_step']} (delay {report['delay_steps']} steps), "
                     f"every worker knew by step {report['all_workers_knew_at_step']}")
    lines.append(f"  healthy workers wrongly suspected: {summary['false_suspicions']}, "
                 f"wrongly confirmed: {len(summary['false_confirmations'])}")
    for rank, subject, step in summary["false_confirmations"]:
        lines.append(f"    {worker_name(rank)} confirmed {worker_name(subject)} at step {step}")
    for event in events:
        if event["event"] == "run_end":
            lines.append(f"  at the end {worker_name(event['rank'])} believes stragglers are: "
                         f"{workers_text(event.get('believed_stragglers', []))}")
    return lines


def trace_text(events, **filters):
    return "\n".join(trace_lines(events, **filters)) + "\n"


def write_trace(run_directory):
    """Write trace.log next to events.jsonl and return its path."""
    run_directory = Path(run_directory)
    trace_path = run_directory / "trace.log"
    trace_path.write_text(trace_text(read_events(run_directory / "events.jsonl")), encoding="utf-8")
    return trace_path
