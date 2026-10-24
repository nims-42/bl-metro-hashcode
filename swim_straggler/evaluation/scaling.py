"""Results of a sweep over group sizes (the simulator sweeps): how detection changes with the number of workers.

Written to results.md and three plots by write_results when the runs have more than one group size.
Every number comes from runs.jsonl, except the message load, which is computed from the
detector settings (it is a property of the design, not something that varies between runs).
"""

import json
from pathlib import Path

import yaml

from swim_straggler.config import DetectorConfig
from swim_straggler.detector.gossip import gossip_length
from swim_straggler.evaluation.metrics import summarize_sweep
from swim_straggler.evaluation.plots import (
    GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, detector_colours, new_figure, style_axes,
)
from swim_straggler.evaluation.sweep import read_finished_rows


def number_text(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def detection_load(workers, detector_config=None):
    """Numbers each worker sends per step for detection, by detector (from the design, not measured)."""
    detector_config = detector_config or DetectorConfig()
    numbers_per_message = gossip_length(detector_config.gossip_rows, detector_config.verdict_slots)
    messages_per_step = 2 * (workers - 1)
    return {
        "swim_per_message": numbers_per_message,
        # the gossip rides in the header of every ring message
        "swim_per_step": numbers_per_message * messages_per_step,
        # N report slots are added to the all-reduced vector; each worker sends 2(N-1) chunks of 1/N of it
        "central_per_step": messages_per_step,
        "steptime_per_step": 0,
    }


def cell(summary):
    """"3/3, 6 (5-10)": runs where the culprit was found, then median delay and range."""
    if summary is None:
        return "-"
    text = f"{summary['found']}/{summary['runs']}"
    if summary["median_delay_steps"] is not None:
        text += f", {summary['median_delay_steps']:.0f}"
        if summary["runs"] > 1:
            text += f" ({summary['min_delay_steps']:.0f}-{summary['max_delay_steps']:.0f})"
    return text


def scaling_markdown(rows, measured_rows=None, detector_config=None):
    summaries = summarize_sweep(rows)
    lookup = {(summary["detector"], summary["fault"], summary["workers"]): summary for summary in summaries}
    detectors = [detector for detector in dict.fromkeys(row["detector"] for row in rows) if detector != "off"]
    faults = list(dict.fromkeys(row["fault"] for row in rows))
    sizes = sorted({row["workers"] for row in rows})
    repeats = max(row["repeat"] for row in rows) + 1

    parts = [
        "**SIMULATION.** These numbers come from the simulator (`cluster.backend: simulated`): the detectors are the "
        "real code, but compute and message times are drawn from a random model, not measured. They show how the "
        "protocol behaves as the group grows, under that model; they are not measurements of a real cluster. "
        "See docs/simulation.md for the model and its limits.",
        "",
        f"Group sizes {', '.join(str(size) for size in sizes)}; up to {repeats} runs per cell "
        "(each repeat has a different seed; every detector gets the same seeds).",
        "",
    ]
    for fault in faults:
        if fault == "healthy":
            continue
        parts += [f"**Fault {fault}**: culprit found in m of n runs, then the median delay in steps (range)", ""]
        parts.append(markdown_table(["workers"] + detectors,
                                    [[size] + [cell(lookup.get((detector, fault, size))) for detector in detectors]
                                     for size in sizes]))
        parts.append("")
        if "swim" in detectors:
            knew = [[size, number_text((lookup.get(("swim", fault, size)) or {}).get("median_everyone_knew_delay_steps"), 0)]
                    for size in sizes]
            parts += [f"SWIM, fault {fault}: median steps from the fault start until every worker knew", ""]
            parts += [markdown_table(["workers", "everyone knew after (steps)"], knew), ""]
    if "healthy" in faults:
        parts += ["**Healthy runs**: runs with at least one false confirmation, of all runs", ""]
        parts.append(markdown_table(["workers"] + detectors, [
            [size] + [("-" if (summary := lookup.get((detector, "healthy", size))) is None
                       else f"{summary['runs_with_false_confirmation']}/{summary['runs']}") for detector in detectors]
            for size in sizes]))
        parts.append("")

    parts += ["**Detection load**: numbers each worker sends per step for detection (from the design, not measured; "
              "the default gossip_rows and verdict_slots)", ""]
    load_rows = []
    for size in sizes:
        load = detection_load(size, detector_config)
        load_rows.append([size, load["swim_per_message"], load["swim_per_step"], load["central_per_step"]])
    parts.append(markdown_table(["workers", "SWIM per message", "SWIM per step", "central per step"], load_rows))
    parts += ["", "SWIM adds the same small header to every message (constant per message), but a ring step has "
              "2(N-1) messages, so per step its load grows with N, and it is larger than the central monitor's, "
              "which adds only N numbers to the vector that is all-reduced anyway. The central monitor's cost is "
              "elsewhere: one worker decides for everyone, and nobody decides if it fails.", ""]

    if measured_rows:
        parts += validation_markdown(rows, measured_rows)

    parts += ["![detection delay by group size](scaling_delay.png)", "",
              "![false confirmations by group size](scaling_false_confirmations.png)", "",
              "![detection load by group size](scaling_load.png)", ""]
    return "\n".join(parts)


def validation_markdown(simulated_rows, measured_rows):
    """The simulation at the measured group size, next to the measurement."""
    measured = {(summary["detector"], summary["fault"]): summary for summary in summarize_sweep(measured_rows)}
    measured_size = measured_rows[0]["workers"]
    simulated = {(summary["detector"], summary["fault"]): summary for summary in summarize_sweep(simulated_rows)
                 if summary["workers"] == measured_size}
    pairs = [key for key in simulated if key in measured and key[0] != "off"]
    if not pairs:
        return []
    table = []
    for detector, fault in pairs:
        real, model = measured[(detector, fault)], simulated[(detector, fault)]
        if fault == "healthy":
            table.append([detector, fault, "-", "-", f"{real['runs_with_false_confirmation']}/{real['runs']}",
                          f"{model['runs_with_false_confirmation']}/{model['runs']}"])
        else:
            table.append([detector, fault, cell(real), cell(model), f"{real['runs_with_false_confirmation']}/{real['runs']}",
                          f"{model['runs_with_false_confirmation']}/{model['runs']}"])
    return [f"**Check against measurement** ({measured_size} workers, the same faults and detectors): if the simulator "
            "is far off here, trust its large-group numbers less", "",
            markdown_table(["detector", "fault", "measured: found, delay", "simulated: found, delay",
                            "measured: runs with false confirmation", "simulated: runs with false confirmation"], table),
            ""]


def plot_scaling_delay(summaries, detectors, faults, output_path):
    """One panel per fault: median delay against group size; hollow markers where the culprit was not always found."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    panels = [fault for fault in faults if fault != "healthy"]
    if not panels:
        return
    figure, axes_grid = pyplot.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), dpi=130, squeeze=False)
    axes_list = list(axes_grid[0])
    colours = detector_colours(detectors)
    for axes, fault in zip(axes_list, panels):
        for detector in detectors:
            points = sorted((summary["workers"], summary) for summary in summaries
                            if summary["detector"] == detector and summary["fault"] == fault)
            found_points = [(size, summary) for size, summary in points if summary["median_delay_steps"] is not None]
            if not found_points:
                continue
            sizes = [size for size, _ in found_points]
            medians = [summary["median_delay_steps"] for _, summary in found_points]
            axes.plot(sizes, medians, color=colours[detector], linewidth=2, label=detector, zorder=3)
            for size, summary in found_points:
                always = summary["found"] == summary["runs"]
                axes.scatter([size], [summary["median_delay_steps"]], s=34, zorder=4, linewidths=1.8,
                             color=colours[detector] if always else SURFACE, edgecolors=colours[detector])
                axes.vlines(size, summary["min_delay_steps"], summary["max_delay_steps"], color=colours[detector],
                            linewidth=1, alpha=0.5, zorder=2)
        axes.set_xscale("log", base=2)
        axes.set_ylim(bottom=0)
        style_axes(axes, f"fault {fault}", "workers (log scale)", "steps to first confirmation", title_pad=8)
        axes.xaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{int(value)}" if value >= 1 else ""))
    axes_list[0].legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8, loc="upper left")
    figure.suptitle("SIMULATION: delay by group size (median, range; hollow: culprit not found in every run)",
                    x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    figure.patch.set_facecolor(SURFACE)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def plot_scaling_false_confirmations(summaries, detectors, output_path):
    pyplot, figure, axes = new_figure(height=3.8)
    colours = detector_colours(detectors)
    for detector in detectors:
        points = sorted((summary["workers"], summary) for summary in summaries
                        if summary["detector"] == detector and summary["fault"] == "healthy")
        if not points:
            continue
        sizes = [size for size, _ in points]
        shares = [100 * summary["runs_with_false_confirmation"] / summary["runs"] for _, summary in points]
        axes.plot(sizes, shares, color=colours[detector], linewidth=2, marker="o", markersize=5, label=detector)
    axes.set_xscale("log", base=2)
    axes.xaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{int(value)}" if value >= 1 else ""))
    axes.set_ylim(-3, 105)
    style_axes(axes, "SIMULATION: healthy runs with a false confirmation", "workers (log scale)", "share of runs (%)")
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def plot_scaling_load(sizes, output_path, detector_config=None):
    pyplot, figure, axes = new_figure(height=3.6)
    colours = detector_colours(["swim", "central"])
    loads = [detection_load(size, detector_config) for size in sizes]
    axes.plot(sizes, [load["swim_per_step"] for load in loads], color=colours["swim"], linewidth=2, marker="o",
              markersize=4, label="swim (header on every message)")
    axes.plot(sizes, [load["central_per_step"] for load in loads], color=colours["central"], linewidth=2, marker="o",
              markersize=4, label="central (N slots in the all-reduced vector)")
    axes.set_xscale("log", base=2)
    axes.set_yscale("log")
    axes.xaxis.set_major_formatter(pyplot.FuncFormatter(lambda value, _: f"{int(value)}" if value >= 1 else ""))
    style_axes(axes, "Numbers each worker sends per step for detection (from the design)", "workers (log scale)",
               "numbers per step (log scale)")
    axes.grid(axis="y", which="minor", color=GRID, linewidth=0.4)
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def measured_rows_for(results_directory):
    """Rows of the measured sweep named by `compare_with` in this sweep's copy of its sweep file, if any."""
    spec_copy = Path(results_directory) / "sweep.yaml"
    if not spec_copy.exists():
        return None
    compare_with = (yaml.safe_load(spec_copy.read_text(encoding="utf-8")) or {}).get("compare_with")
    if not compare_with:
        return None
    rows = read_finished_rows(Path(compare_with) / "runs.jsonl")
    return rows or None


def write_scaling_results(results_directory, rows):
    results_directory = Path(results_directory)
    summaries = summarize_sweep(rows)
    (results_directory / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    detectors = [detector for detector in dict.fromkeys(row["detector"] for row in rows) if detector != "off"]
    faults = list(dict.fromkeys(row["fault"] for row in rows))
    sizes = sorted({row["workers"] for row in rows})
    plot_scaling_delay(summaries, detectors, faults, results_directory / "scaling_delay.png")
    plot_scaling_false_confirmations(summaries, detectors, results_directory / "scaling_false_confirmations.png")
    plot_scaling_load(sizes, results_directory / "scaling_load.png")
    results_path = results_directory / "results.md"
    results_path.write_text(f"# Results: {results_directory.name} (simulation)\n\n"
                            + scaling_markdown(rows, measured_rows_for(results_directory)) + "\n", encoding="utf-8")
    return results_path
