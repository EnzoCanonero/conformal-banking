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
