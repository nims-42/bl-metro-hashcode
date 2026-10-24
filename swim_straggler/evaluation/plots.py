"""Plots for the results (PNG). Every number shown here is also in results.md as a table.

Colours: a validated categorical palette, assigned to detectors in a fixed order (never cycled),
so a detector keeps its colour in every plot.
"""

from pathlib import Path

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


NEUTRAL = "#8f8e8a"  # the detector switched off is a reference, not a series


def detector_colours(detectors):
    """A fixed colour per detector, the same in every plot. "off" is grey and takes no palette slot."""
    detecting = [detector for detector in detectors if detector != "off"]
    colours = {detector: CATEGORICAL[position % len(CATEGORICAL)] for position, detector in enumerate(detecting)}
    colours["off"] = NEUTRAL
    return colours


def style_axes(axes, title, x_label, y_label, title_pad=10):
    axes.set_facecolor(SURFACE)
    axes.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=title_pad)
    axes.set_xlabel(x_label, color=TEXT_SECONDARY)
    axes.set_ylabel(y_label, color=TEXT_SECONDARY)
    axes.tick_params(colors=TEXT_SECONDARY, length=0)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)


def new_figure(width=8, height=4.2):
    import matplotlib

    matplotlib.use("Agg")  # no window, files only
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(figsize=(width, height), dpi=130)
    figure.patch.set_facecolor(SURFACE)
    return pyplot, figure, axes


def plot_detection_delay(rows, detectors, faults, output_path, colour_order=None):
    """Each dot is one run in which the culprit was found; the short bar is the median."""
    import statistics

    pyplot, figure, axes = new_figure()
    colours = detector_colours(colour_order or detectors)
    width = 0.8 / max(len(detectors), 1)
    for fault_position, fault in enumerate(faults):
        for detector_position, detector in enumerate(detectors):
            centre = fault_position - 0.4 + width * (detector_position + 0.5)
            group = [row for row in rows if row["fault"] == fault and row["detector"] == detector]
            delays = [row["delay_steps"] for row in group if row["culprit_found"]]
            if delays:
                axes.scatter([centre] * len(delays), delays, s=28, color=colours[detector], zorder=3,
                             edgecolors=SURFACE, linewidths=1.5, label=detector if fault_position == 0 else None)
                axes.plot([centre - width * 0.35, centre + width * 0.35], [statistics.median(delays)] * 2,
                          color=TEXT_PRIMARY, linewidth=2, zorder=4)
            if group:
                axes.annotate(f"{len(delays)}/{len(group)}", (centre, 0), xytext=(0, -14), textcoords="offset points",
                              ha="center", fontsize=7, color=TEXT_SECONDARY, annotation_clip=False)
    axes.set_xticks(range(len(faults)), faults)
    axes.tick_params(axis="x", pad=18)
    axes.set_ylim(bottom=0)
    style_axes(axes, "Steps from fault start to first confirmation", "fault (n/m under each dot column: runs where the culprit was found)",
               "delay (training steps)", title_pad=26)
    # Above the plot area, so it never covers a data point.
    axes.legend(frameon=False, labelcolor=TEXT_SECONDARY, ncol=len(detectors), loc="lower left",
                bbox_to_anchor=(0, 1.0), fontsize=8, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def plot_false_confirmations(summaries, detectors, output_path, colour_order=None):
    """Share of runs with at least one healthy worker confirmed slow: healthy runs and fault runs side by side."""
    pyplot, figure, axes = new_figure(height=3.8)
    colours = detector_colours(colour_order or detectors)

    def share(detector, healthy):
        groups = [summary for summary in summaries if summary["detector"] == detector and (summary["fault"] == "healthy") == healthy]
        runs = sum(group["runs"] for group in groups)
        with_false = sum(group["runs_with_false_confirmation"] for group in groups)
        return (with_false / runs if runs else None), with_false, runs

    positions = range(len(detectors))
    for offset, healthy, label in ((-0.18, True, "healthy runs"), (0.18, False, "runs with a fault")):
        for position, detector in zip(positions, detectors):
            value, with_false, runs = share(detector, healthy)
            if value is None:
                continue
            axes.bar(position + offset, value * 100, width=0.34, color=colours[detector],
                     alpha=1.0 if healthy else 0.55, edgecolor=SURFACE, linewidth=2)
            axes.annotate(f"{with_false}/{runs}", (position + offset, value * 100), xytext=(0, 3),
                          textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    axes.set_xticks(list(positions), detectors)
    axes.set_ylim(0, 105)
    style_axes(axes, "Runs where a healthy worker was wrongly confirmed slow", "detector  (left bar: healthy runs, right bar, lighter: runs with a fault)",
               "share of runs (%)")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def plot_overhead(overhead_rows, detectors, output_path, colour_order=None):
    pyplot, figure, axes = new_figure(height=3.4)
    colours = detector_colours(colour_order or detectors)
    rows = [row for detector in detectors for row in overhead_rows if row["detector"] == detector]
    for position, row in enumerate(rows):
        axes.bar(position, row["mean_step_ms"], width=0.55, color=colours[row["detector"]], edgecolor=SURFACE, linewidth=2)
        axes.annotate(f"{row['mean_step_ms']:.1f} ms ({row['overhead_percent']:+.0f}%)", (position, row["mean_step_ms"]),
                      xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY)
    axes.set_xticks(range(len(rows)), [row["detector"] for row in rows])
    style_axes(axes, "Mean step time on healthy runs (change against detector off)", "detector", "step time (ms)")
    figure.tight_layout()
    figure.savefig(output_path, facecolor=SURFACE)
    pyplot.close(figure)


def make_all_plots(rows, summaries, overhead_rows, results_directory):
    detectors = list(dict.fromkeys(row["detector"] for row in rows))
    colour_order = detectors  # every plot below gets its colours from this one list
    faults_with_culprit = [fault for fault in dict.fromkeys(row["fault"] for row in rows) if fault != "healthy"]
    detecting = [detector for detector in detectors if detector != "off"]
    paths = {}
    if faults_with_culprit:
        paths["delay"] = Path(results_directory) / "detection_delay.png"
        plot_detection_delay(rows, detecting, faults_with_culprit, paths["delay"], colour_order)
    paths["false"] = Path(results_directory) / "false_confirmations.png"
    plot_false_confirmations(summaries, detecting, paths["false"], colour_order)
    if overhead_rows:
        paths["overhead"] = Path(results_directory) / "step_time_overhead.png"
        plot_overhead(overhead_rows, detectors, paths["overhead"], colour_order)
    return paths
