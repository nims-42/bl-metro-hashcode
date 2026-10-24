"""results.md, generated from runs.jsonl. No number is typed by hand."""

import json
from pathlib import Path

from swim_straggler.evaluation.metrics import step_time_overhead, summarize_sweep
from swim_straggler.evaluation.sweep import read_finished_rows


def number_text(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def delay_text(summary):
    if summary["median_delay_steps"] is None:
        return "-"
    return (f"{number_text(summary['median_delay_steps'])} "
            f"({number_text(summary['min_delay_steps'])}–{number_text(summary['max_delay_steps'])})")


def results_markdown(results_directory):
    """The whole results section for one sweep folder."""
    results_directory = Path(results_directory)
    rows = read_finished_rows(results_directory / "runs.jsonl")
    if not rows:
        return "No finished runs yet."
    summaries = summarize_sweep(rows)
    overhead = step_time_overhead(summaries)
    machine = json.loads((results_directory / "machine.json").read_text(encoding="utf-8"))
    sample = rows[0]
    repeats = max(row["repeat"] for row in rows) + 1

    parts = [
        f"Measured on {machine['date']}: {sample['workers']} workers, backend `{sample['backend']}`, "
        f"{machine['processors']} processors, {machine['platform']}, Python {machine['python']}, torch {machine['torch']}. "
        f"Up to {repeats} repeats per cell (seed changes per repeat; every detector sees the same seeds). "
        f"Sweep definition: `{results_directory.as_posix()}/sweep.yaml`; every run: `{results_directory.as_posix()}/runs.jsonl`.",
        "",
        "Timing on a shared machine is noisy, and the number of runs per cell is small: treat differences of a few "
        "steps, or of one false confirmation, as noise.",
        "",
    ]

    fault_summaries = [summary for summary in summaries if summary["fault"] != "healthy" and summary["detector"] != "off"]
    if fault_summaries:
        parts += ["**Detection with a slow worker** (delay = steps from fault start to the first confirmation, "
                  "median and range over the runs where the culprit was found)", ""]
        parts.append(markdown_table(
            ["detector", "fault", "culprit found", "delay (steps)", "everyone knew after (steps)", "runs with a false confirmation",
             "measured slowdown"],
            [[summary["detector"], summary["fault"], f"{summary['found']} of {summary['runs']}", delay_text(summary),
              number_text(summary["median_everyone_knew_delay_steps"]),
              f"{summary['runs_with_false_confirmation']} of {summary['runs']}",
              "-" if summary["median_compute_ratio"] is None else f"x{summary['median_compute_ratio']:.2f}"]
             for summary in fault_summaries]))
        parts += ["", f"![detection delay](detection_delay.png)", ""]

    healthy_summaries = [summary for summary in summaries if summary["fault"] == "healthy" and summary["detector"] != "off"]
    if healthy_summaries:
        parts += ["**Healthy runs** (no fault: every confirmation is wrong)", ""]
        parts.append(markdown_table(
            ["detector", "runs", "runs with a false confirmation", "false confirmations", "false suspicions"],
            [[summary["detector"], str(summary["runs"]), f"{summary['runs_with_false_confirmation']} of {summary['runs']}",
              str(summary["false_confirmations"]), str(summary["false_suspicions"])] for summary in healthy_summaries]))
        parts += ["", f"![false confirmations](false_confirmations.png)", ""]

    if overhead:
        parts += ["**Cost** (mean step time on healthy runs)", ""]
        parts.append(markdown_table(["detector", "mean step (ms)", "change against off"],
                                    [[row["detector"], number_text(row["mean_step_ms"]), f"{row['overhead_percent']:+.1f}%"]
                                     for row in overhead]))
        parts += ["", f"![step time overhead](step_time_overhead.png)", ""]

    parts += ["Notes: the step-time detector can only accuse its own worker; in synchronous training every step "
              "time rises together, so its false confirmations are healthy workers accusing themselves. "
              "The central monitor sees every compute time directly and does not vote, which makes it faster, "
              "but one worker receives every report and nobody decides if it fails."]
    return "\n".join(parts)


def write_results(results_directory):
    """Summary file, plots and results.md for a sweep folder. Returns the path of results.md."""
    from swim_straggler.evaluation.plots import make_all_plots

    results_directory = Path(results_directory)
    rows = read_finished_rows(results_directory / "runs.jsonl")
    summaries = summarize_sweep(rows)
    (results_directory / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    if rows:
        make_all_plots(rows, summaries, step_time_overhead(summaries), results_directory)
    results_path = results_directory / "results.md"
    results_path.write_text(f"# Results: {results_directory.name}\n\n" + results_markdown(results_directory) + "\n",
                            encoding="utf-8")
    return results_path
