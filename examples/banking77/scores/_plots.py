from collections.abc import Sequence
from math import isfinite
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import NullFormatter, PercentFormatter, ScalarFormatter


METHOD_STYLES = {
    "lac": ("LAC", "#0072B2", "o"),
    "aps": ("APS", "#A05195", "s"),
    "socop_fixed": ("SOCOP, fixed λ = 0.25", "#009E73", "^"),
    "socop_tuned": ("SOCOP, tuned λ", "#D55E00", "D"),
    "naive": ("Naive confidence", "#E69F00", "x"),
}
CONFORMAL_METHODS = ("lac", "aps", "socop_fixed", "socop_tuned")


# Keep the evaluated parameter order, including curves that turn back.
def _method_rows(
    rows: list[dict[str, str | float]], method: str
) -> list[dict[str, str | float]]:
    parameter = "confidence_threshold" if method == "naive" else "alpha"
    method_rows = [row for row in rows if row["method"] == method]
    ordered_rows = sorted(method_rows, key=lambda row: float(row[parameter]))
    return ordered_rows


# Show split variation without treating repeated test records as independent data.
def _plot_method(
    axis: Axes,
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    method: str,
    horizontal_metric: str,
    vertical_metric: str,
) -> None:
    label, color, marker = METHOD_STYLES[method]
    method_rows = _method_rows(rows, method)
    seeds = sorted({float(row["seed"]) for row in method_rows})

    for seed in seeds:
        seed_rows = [row for row in method_rows if float(row["seed"]) == seed]
        horizontal_values = [float(row[horizontal_metric]) for row in seed_rows]
        vertical_values = [float(row[vertical_metric]) for row in seed_rows]
        axis.plot(
            horizontal_values,
            vertical_values,
            color=color,
            alpha=0.18,
            linewidth=0.9,
        )

    mean_rows = _method_rows(summary_rows, method)
    horizontal_means = [float(row[horizontal_metric]) for row in mean_rows]
    vertical_means = [float(row[vertical_metric]) for row in mean_rows]
    axis.plot(
        horizontal_means,
        vertical_means,
        color=color,
        marker=marker,
        markersize=5,
        linewidth=2,
        label=label,
    )


def _style_axis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)


# Error is undefined when nothing is automated; do not plot it as zero.
def save_automation_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    methods: Sequence[str],
    output_path: Path,
) -> None:
    plot_rows = []
    for row in rows:
        error_rate = float(row["automated_error_rate"])
        if float(row["automation_rate"]) > 0.0 and isfinite(error_rate):
            plot_rows.append(row)

    plot_summary = []
    for row in summary_rows:
        error_rate = float(row["automated_error_rate"])
        if float(row["automation_rate"]) > 0.0 and isfinite(error_rate):
            plot_summary.append(row)

    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")
    for method in methods:
        _plot_method(
            axis,
            plot_rows,
            plot_summary,
            method,
            "automation_rate",
            "automated_error_rate",
        )

    axis.set_title("Automation versus routing error")
    axis.set_xlabel("Requests handled automatically")
    axis.set_ylabel("Errors among automated requests")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="upper center", frameon=False)
    _style_axis(axis)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# Compare prediction-set coverage with its target, separately from routing error.
def save_coverage_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")

    for method in CONFORMAL_METHODS:
        _plot_method(
            axis,
            rows,
            summary_rows,
            method,
            "target_coverage",
            "coverage",
        )

    axis.plot([0.45, 1.0], [0.45, 1.0], "--", color="0.35", linewidth=1)
    axis.set_title("Prediction-set coverage against its target")
    axis.set_xlabel("Target coverage")
    axis.set_ylabel("Observed coverage")
    axis.set_xlim(0.45, 1.01)
    axis.set_ylim(0.45, 1.01)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(loc="upper left", frameon=False)
    _style_axis(axis)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# A logarithmic axis keeps small sets and large deferred sets visible together.
def save_set_size_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5), layout="constrained")

    for method in CONFORMAL_METHODS:
        _plot_method(
            axis,
            rows,
            summary_rows,
            method,
            "coverage",
            "average_set_size",
        )

    axis.set_title("Prediction-set size versus coverage")
    axis.set_xlabel("Observed coverage")
    axis.set_ylabel("Average set size (log scale)")
    axis.set_xlim(0.45, 1.01)
    axis.set_yscale("log")
    axis.set_ylim(0.45, 80.0)
    axis.set_yticks([0.5, 1, 2, 5, 10, 20, 50, 77])
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(ScalarFormatter())
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.legend(loc="upper left", frameon=False)
    _style_axis(axis)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
