"""The Streamlit dashboard. Start it with:  swim-straggler dashboard   (or: streamlit run this file -- --runs runs)

It only reads event logs, so it can watch a run while it is going (turn on auto refresh).
"""

import argparse
import os
import sys
import time

import altair
import pandas
import streamlit

from swim_straggler.dashboard.data import (
    belief_rows, chart_markers, comparison_with_other_detectors, detector_event_rows, latest_step, list_runs, load_run_events, round_wait_rows,
    run_overview, step_rows, world_size_of,
)
from swim_straggler.telemetry.events import summarize_detection
from swim_straggler.telemetry.trace import trace_text

BELIEF_COLOURS = {"alive": "#4c9a6a", "suspect": "#e0a030", "straggler": "#c8463c"}


def runs_directory_from_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.environ.get("SWIM_RUNS_DIRECTORY", "runs"))
    arguments, _ = parser.parse_known_args(sys.argv[1:])
    return arguments.runs


def line_chart_with_markers(frame, field, markers, title, faulty_workers):
    """One line per worker; faulty workers drawn thicker; dashed vertical lines at the markers."""
    lines = altair.Chart(frame).mark_line().encode(
        x=altair.X(field="step", type="quantitative", title="training step"),
        y=altair.Y(field=field, type="quantitative", title=title),
        color=altair.Color(field="worker", type="nominal", title="worker"),
        strokeWidth=altair.condition(altair.FieldOneOfPredicate(field="worker", oneOf=list(faulty_workers)),
                                     altair.value(3), altair.value(1.2)),
        tooltip=["step", "worker", altair.Tooltip(field=field, type="quantitative", format=".1f")],
    )
    if not markers:
        return lines.properties(height=260)
    rules = altair.Chart(pandas.DataFrame(markers)).mark_rule(strokeDash=[4, 4], color="gray").encode(
        x=altair.X(field="step", type="quantitative"), tooltip=["label", "step"],
    )
    return (lines + rules).properties(height=260)


def smoothed(frame, columns, window):
    """Rolling median per worker, so single noisy steps do not hide the trend."""
    if window <= 1:
        return frame
    frame = frame.sort_values(["worker", "step"]).copy()
    for column in columns:
        frame[column] = frame.groupby("worker")[column].transform(lambda values: values.rolling(window, min_periods=1).median())
    return frame


def dash_if_none(value):
    return "-" if value is None else value


def show_detection_summary(events):
    summary = summarize_detection(events)
    if summary["detector_mode"] in (None, "off"):
        streamlit.info("The detector was switched off in this run.")
        return
    if not summary["faulty_workers"]:
        streamlit.write("No fault was injected in this run (so far).")
    for culprit, report in summary["per_fault"].items():
        streamlit.markdown(f"**Worker {culprit}** — fault issued at step {report['fault_issued_step']}, "
                           f"in place at step {report['fault_applied_step']}")
        columns = streamlit.columns(4)
        ratio = report["compute_ratio"]
        columns[0].metric("measured effect on its compute", "-" if ratio is None else f"x{ratio:.2f}")
        columns[1].metric("first suspected at step", dash_if_none(report["first_suspected_step"]))
        columns[2].metric("first confirmed at step", dash_if_none(report["first_confirmed_step"]),
                          help="delay counted from the step where the fault was issued")
        columns[3].metric("delay (steps)", dash_if_none(report["delay_steps"]))
        if ratio is not None and ratio < 1.15:
            streamlit.warning("The fault barely changed this worker's compute time, so there was nothing to detect.")
    columns = streamlit.columns(3)
    columns[0].metric("healthy workers wrongly confirmed", len(summary["false_confirmations"]))
    columns[1].metric("healthy workers wrongly suspected", summary["false_suspicions"])
    columns[2].metric("suspicions rejected by the vote", summary["rejected_suspicions"])
    if summary["detector_mode"] == "steptime":
        streamlit.caption("Step-time detector: each worker can only judge itself, and in synchronous training all step "
                          "times rise together, so healthy workers accuse themselves.")


def show_comparison(runs_directory, events):
    """The detector comparison. Built only from runs that already exist in the runs folder."""
    rows, missing = comparison_with_other_detectors(runs_directory, events)
    if rows:
        streamlit.dataframe(pandas.DataFrame(rows), hide_index=True, use_container_width=True)
        streamlit.caption("One run per detector, so timing noise matters; treat differences of a few steps as noise. "
                          "Step-time cannot name the culprit: its 'wrongly confirmed' are healthy workers accusing themselves.")
    if missing:
        streamlit.info(f"No finished run found for: {', '.join(missing)}, on this same experiment. "
                       "Create them with:  swim-straggler demo --compare   (same config, one run per detector)")


def show_trace(events, world_size, last_step):
    if not any(event["event"] == "trace" for event in events):
        streamlit.caption("This run logged decisions only. Run with --trace to also see the reasoning behind them.")
    choices = ["everyone"] + [f"worker {rank}" for rank in range(world_size)]
    left, middle, right = streamlit.columns(3)
    done_by = left.selectbox("what was done by", choices, key="trace_worker")
    about = middle.selectbox("what concerns", choices, key="trace_about")
    first_step, last_shown_step = right.slider("steps", 0, max(last_step, 1), (0, max(last_step, 1)), key="trace_steps")
    text = trace_text(events, worker=None if done_by == "everyone" else int(done_by.split()[1]),
                      about=None if about == "everyone" else int(about.split()[1]),
                      first_step=first_step, last_step=last_shown_step)
    streamlit.code(text, language=None)


def main():
    streamlit.set_page_config(page_title="swim-straggler", layout="wide")
    streamlit.title("swim-straggler")

    runs_directory = streamlit.sidebar.text_input("runs folder", runs_directory_from_arguments())
    runs = list_runs(runs_directory)
    if not runs:
        streamlit.info(f"No runs found in '{runs_directory}'. Start one with: swim-straggler demo --workers 4")
        return
    names = [name for name, _ in runs]
    chosen_name = streamlit.sidebar.selectbox("run (newest first)", names)
    run_directory = dict(runs)[chosen_name]
    events, is_finished = load_run_events(run_directory)
    auto_refresh = streamlit.sidebar.checkbox("auto refresh every 2 s", value=not is_finished)
    smoothing_window = streamlit.sidebar.slider("smooth charts (median over steps)", 1, 15, 1)

    overview = run_overview(events)
    streamlit.subheader(chosen_name + ("" if is_finished else "  (still running)"))
    if any(event["event"] == "run_start" and event.get("simulated") for event in events):
        streamlit.warning("SIMULATION: nothing was trained or timed. The detectors are the real code; the compute and "
                          "message times come from the simulator's random model (docs/simulation.md).")
    if overview:
        streamlit.dataframe(pandas.DataFrame([overview]), hide_index=True)
    steps = step_rows(events)
    if not steps:
        streamlit.info("No steps logged yet.")
    else:
        frame = smoothed(pandas.DataFrame(steps), ["compute (ms)", "wait (ms)", "step (ms)"], smoothing_window)
        markers = chart_markers(events)

        streamlit.header("What the detectors found")
        show_detection_summary(events)

        streamlit.header("The signal")
        faulty_workers = sorted({event["rank"] for event in events if event["event"] == "fault_started"})
        if markers:
            marker_text = "; ".join(f"step {marker['step']}: {marker['label']}" for marker in sorted(markers, key=lambda m: m["step"]))
            streamlit.caption(f"Dashed lines — {marker_text}. Thick line: the worker with the injected fault.")
        left, right = streamlit.columns(2)
        left.markdown("**Compute time** — a slow worker computes longer.")
        left.altair_chart(line_chart_with_markers(frame, "compute (ms)", markers, "compute (ms)", faulty_workers), use_container_width=True)
        right.markdown("**Wait time** — and waits least, because everybody else waits for it.")
        right.altair_chart(line_chart_with_markers(frame, "wait (ms)", markers, "wait (ms)", faulty_workers), use_container_width=True)
        left.markdown("**Step time** — the same for everyone, so step time alone cannot name the culprit.")
        left.altair_chart(line_chart_with_markers(frame, "step (ms)", markers, "step (ms)", faulty_workers), use_container_width=True)
        right.markdown("**Training loss** — the average over all workers.")
        loss_frame = frame[frame["worker"] == 0]
        right.altair_chart(altair.Chart(loss_frame).mark_line().encode(
            x=altair.X(field="step", type="quantitative", title="training step"),
            y=altair.Y(field="loss", type="quantitative", title="loss")).properties(height=260), use_container_width=True)

        streamlit.header("One step up close")
        last_step = latest_step(events)
        confirmed_steps = [marker["step"] for marker in markers if "confirmed" in marker["label"]]
        default_step = min(confirmed_steps[0] if confirmed_steps else last_step, max(last_step, 0))
        chosen_step = streamlit.slider("step (starts at the first confirmation, if any)", 0, max(last_step, 0), default_step)
        world_size = world_size_of(events)
        left, right = streamlit.columns(2)
        left.markdown("**Waiting per ring round.** The worker after the slow one waits in round 0, the next one in "
                      "round 1, and so on. The detector reads the culprit from this diagonal.")
        rounds = pandas.DataFrame(round_wait_rows(events, chosen_step))
        if not rounds.empty:
            left.altair_chart(altair.Chart(rounds).mark_rect().encode(
                x=altair.X(field="round", type="ordinal", title="round (reduce-scatter first, then all-gather)"),
                y=altair.Y(field="worker", type="ordinal"),
                color=altair.Color(field="wait (ms)", type="quantitative", scale=altair.Scale(scheme="oranges")),
                tooltip=["worker", "round", altair.Tooltip(field="wait (ms)", type="quantitative", format=".2f")],
            ).properties(height=220), use_container_width=True)
        right.markdown("**Who believes what** at the end of this step (row: the believer, column: the worker it is about).")
        beliefs = pandas.DataFrame(belief_rows(events, chosen_step, world_size))
        right.altair_chart(altair.Chart(beliefs).mark_rect(stroke="white").encode(
            x=altair.X(field="about worker", type="ordinal"),
            y=altair.Y(field="worker (believer)", type="ordinal"),
            color=altair.Color(field="belief", type="nominal",
                               scale=altair.Scale(domain=list(BELIEF_COLOURS), range=list(BELIEF_COLOURS.values()))),
            tooltip=["worker (believer)", "about worker", "belief"],
        ).properties(height=220), use_container_width=True)

        streamlit.header("Detector events")
        detector_rows = detector_event_rows(events)
        if detector_rows:
            streamlit.dataframe(pandas.DataFrame(detector_rows), hide_index=True, use_container_width=True)
        else:
            streamlit.write("No detector events.")

        with streamlit.expander("Trace: every event, step by step (the same text as trace.log)"):
            show_trace(events, world_size, last_step)

        # Hidden until asked for. The choice is kept in session_state so it survives the auto refresh.
        showing_comparison = streamlit.session_state.get("show_comparison", False)
        if streamlit.button("Hide comparison with baseline detectors" if showing_comparison
                            else "Compare with baseline detectors"):
            streamlit.session_state["show_comparison"] = not showing_comparison
            streamlit.rerun()
        if showing_comparison:
            streamlit.header("Comparison with baseline detectors")
            show_comparison(runs_directory, events)

    if auto_refresh:
        time.sleep(2)
        streamlit.rerun()


main()
