from math import isfinite
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


MODEL_STYLES = {
    "tfidf": ("Word TF-IDF", "#0072B2", "o"),
    "encoder": ("Frozen encoder", "#D55E00", "s"),
    "qwen": ("Qwen 4B", "#009E73", "^"),
}
METHOD_STYLES = {
    "lac": ("LAC", "#0072B2", "o"),
    "socop_tuned": ("SOCOP, tuned λ", "#D55E00", "D"),
    "naive": ("Naive confidence", "#009E73", "^"),
}


# Retain the parameter order, including curves that turn back in automation.
def _plot_curve(
    axis: Axes,
    rows: list[dict[str, str | float]],
    parameter: str,
    style: tuple[str, str, str],
) -> None:
    plot_rows = []
    for row in rows:
        automation_rate = float(row["automation_rate"])
        error_rate = float(row["automated_error_rate"])
        if automation_rate > 0.0 and isfinite(error_rate):
            plot_rows.append(row)

    plot_rows = sorted(plot_rows, key=lambda row: float(row[parameter]))
    automation = [float(row["automation_rate"]) for row in plot_rows]
    errors = [float(row["automated_error_rate"]) for row in plot_rows]
    label, color, marker = style
    axis.plot(
        automation,
        errors,
        color=color,
        marker=marker,
        markersize=5,
        linewidth=1.8,
        label=label,
    )


def _style_axis(axis: Axes, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Requests handled automatically")
    axis.set_ylabel("Errors among automated requests")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(bottom=0.0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False)


# Show Qwen's coverage uncertainty and the set-size cost of retaining more answers.
def save_qwen_diagnostic_figures(
    conformal_rows: list[dict[str, str | float]],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    coverage_figure, coverage_axis = plt.subplots(
        figsize=(7, 5.2), layout="constrained"
    )
    size_figure, size_axis = plt.subplots(figsize=(7, 5.2), layout="constrained")

    for method in ("lac", "socop_tuned"):
        rows = [
            row
            for row in conformal_rows
            if row["model"] == "qwen" and row["method"] == method
        ]
        rows = sorted(rows, key=lambda row: float(row["target_coverage"]))
        targets = [float(row["target_coverage"]) for row in rows]
        coverage = [float(row["coverage"]) for row in rows]
        set_sizes = [float(row["average_set_size"]) for row in rows]
        lower_errors = [
            float(row["coverage"]) - float(row["coverage_lower"]) for row in rows
        ]
        upper_errors = [
            float(row["coverage_upper"]) - float(row["coverage"]) for row in rows
        ]
        label, color, marker = METHOD_STYLES[method]
        line_style = "-" if method == "lac" else "--"

        # Intervals describe the fixed test sample, not variation across splits.
        coverage_axis.errorbar(
            targets,
            coverage,
            yerr=[lower_errors, upper_errors],
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=6,
            linestyle=line_style,
            linewidth=1.8,
            elinewidth=1,
            capsize=3,
            label=label,
        )
        size_axis.plot(
            coverage,
            set_sizes,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markersize=6,
            linestyle=line_style,
            linewidth=1.8,
            label=label,
        )

    coverage_axis.plot(
        [0.45, 1.0], [0.45, 1.0], ":", color="0.4", linewidth=1, label="Target"
    )
    coverage_axis.set_title("Qwen 4B: prediction-set coverage")
    coverage_axis.set_xlabel("Target coverage")
    coverage_axis.set_ylabel("Observed coverage")
    coverage_axis.set_xlim(0.47, 1.01)
    coverage_axis.set_ylim(0.47, 1.01)
    coverage_axis.yaxis.set_major_formatter(PercentFormatter(1.0))

    size_axis.set_title("Qwen 4B: prediction-set size")
    size_axis.set_xlabel("Observed coverage")
    size_axis.set_ylabel("Mean prediction-set size")
    size_axis.set_xlim(0.5, 1.01)
    size_axis.set_ylim(bottom=0.0)

    for figure, axis, filename in (
        (coverage_figure, coverage_axis, "qwen_coverage_vs_target.png"),
        (size_figure, size_axis, "qwen_set_size_vs_coverage.png"),
    ):
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(alpha=0.18)
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", frameon=False)
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)


# Compare models at seed 42, then compare routing rules on Qwen alone.
def save_comparison_figures(
    conformal_rows: list[dict[str, str | float]],
    naive_rows: list[dict[str, str | float]],
    output_directory: Path,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    error_rates = []
    for row in conformal_rows:
        error_rate = float(row["automated_error_rate"])
        if float(row["automation_rate"]) > 0.0 and isfinite(error_rate):
            error_rates.append(error_rate)
    highest_error_rate = max(error_rates)

    # The two model comparisons share axis limits for side-by-side reading.
    for method in ("lac", "socop_tuned"):
        figure, axis = plt.subplots(figsize=(5.8, 4.8), layout="constrained")
        for model, style in MODEL_STYLES.items():
            selected_rows = [
                row
                for row in conformal_rows
                if row["model"] == model and row["method"] == method
            ]
            _plot_curve(axis, selected_rows, "alpha", style)

        _style_axis(axis, METHOD_STYLES[method][0])
        axis.set_ylim(0.0, highest_error_rate * 1.05)
        filename = "automation_vs_error_lac.png"
        if method == "socop_tuned":
            filename = "automation_vs_error_socop.png"
        figure.savefig(output_directory / filename, dpi=180)
        plt.close(figure)

    all_rows = conformal_rows + naive_rows
    qwen_rows = [row for row in all_rows if row["model"] == "qwen"]
    figure, axis = plt.subplots(figsize=(7, 5.2), layout="constrained")
    for method, style in METHOD_STYLES.items():
        selected_rows = [row for row in qwen_rows if row["method"] == method]
        parameter = "confidence_threshold" if method == "naive" else "alpha"
        _plot_curve(axis, selected_rows, parameter, style)

    _style_axis(axis, "Qwen 4B")
    figure.savefig(output_directory / "qwen_automation_vs_error.png", dpi=180)
    plt.close(figure)
    save_qwen_diagnostic_figures(conformal_rows, output_directory)
