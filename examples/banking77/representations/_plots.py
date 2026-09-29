from math import isfinite
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import PercentFormatter


REPRESENTATION_STYLES = {
    "tfidf": ("Word TF-IDF", "#0072B2", "o"),
    "encoder": ("Frozen encoder", "#D55E00", "s"),
}
METHOD_TITLES = {"lac": "LAC", "socop_tuned": "SOCOP, tuned λ"}


# Keep the evaluated alpha order, including routing curves that turn back.
def _ordered_rows(
    rows: list[dict[str, str | float]], method: str, representation: str
) -> list[dict[str, str | float]]:
    selected_rows = []
    for row in rows:
        if row["method"] == method and row["representation"] == representation:
            selected_rows.append(row)
    return sorted(selected_rows, key=lambda row: float(row["alpha"]))


# Show individual splits faintly and their arithmetic mean as the main curve.
def _plot_curve(
    axis: Axes,
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    style: tuple[str, str, str],
    horizontal_metric: str,
    vertical_metric: str,
) -> None:
    label, color, marker = style
    seeds = sorted({float(row["seed"]) for row in rows})
    for seed in seeds:
        seed_rows = [row for row in rows if float(row["seed"]) == seed]
        horizontal_values = [float(row[horizontal_metric]) for row in seed_rows]
        vertical_values = [float(row[vertical_metric]) for row in seed_rows]
        axis.plot(
            horizontal_values, vertical_values, color=color, alpha=0.18, linewidth=0.9
        )

    horizontal_means = [float(row[horizontal_metric]) for row in summary_rows]
    vertical_means = [float(row[vertical_metric]) for row in summary_rows]
    axis.plot(
        horizontal_means,
        vertical_means,
        color=color,
        marker=marker,
        markersize=5,
        linewidth=2,
        label=label,
    )


# Undefined error rates have no routing point; they are not zero-error outcomes.
def _routing_rows(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    selected_rows = []
    for row in rows:
        error_rate = float(row["automated_error_rate"])
        if float(row["automation_rate"]) > 0.0 and isfinite(error_rate):
            selected_rows.append(row)
    return selected_rows


def _style_axis(axis: Axes, title: str, horizontal_label: str) -> None:
    axis.set_title(title)
    axis.set_xlabel(horizontal_label)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.18)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=False)


# Save one method's routing figure, retaining the same scale across methods.
def save_automation_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    method: str,
    output_path: Path,
) -> None:
    plot_rows = _routing_rows(rows)
    plot_summary = _routing_rows(summary_rows)
    highest_error_rate = max(float(row["automated_error_rate"]) for row in plot_rows)
    figure, axis = plt.subplots(figsize=(5.8, 4.8), layout="constrained")
    for representation in REPRESENTATION_STYLES:
        method_rows = _ordered_rows(plot_rows, method, representation)
        mean_rows = _ordered_rows(plot_summary, method, representation)
        _plot_curve(
            axis, method_rows, mean_rows, REPRESENTATION_STYLES[representation],
            "automation_rate", "automated_error_rate",
        )
    _style_axis(axis, METHOD_TITLES[method], "Requests handled automatically")

    axis.set_ylabel("Errors among automated requests")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, highest_error_rate * 1.05)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# Coverage includes automated and deferred requests; the diagonal marks the target.
def save_coverage_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    method: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.8, 4.8), layout="constrained")
    axis.plot([0.45, 1.0], [0.45, 1.0], "--", color="0.35", linewidth=1)
    for representation in REPRESENTATION_STYLES:
        method_rows = _ordered_rows(rows, method, representation)
        mean_rows = _ordered_rows(summary_rows, method, representation)
        _plot_curve(
            axis, method_rows, mean_rows, REPRESENTATION_STYLES[representation],
            "target_coverage", "coverage",
        )
    _style_axis(axis, METHOD_TITLES[method], "Target coverage")

    axis.set_ylabel("Observed prediction-set coverage")
    axis.set_xlim(0.45, 1.01)
    axis.set_ylim(0.45, 1.01)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


# Compare routing rules on the same frozen-encoder predictions and test requests.
def save_encoder_automation_figure(
    rows: list[dict[str, str | float]],
    summary_rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    encoder_rows = [row for row in rows if row["representation"] == "encoder"]
    encoder_summary = [row for row in summary_rows if row["representation"] == "encoder"]
    plot_rows = _routing_rows(encoder_rows)
    plot_summary = _routing_rows(encoder_summary)
    styles = {
        "lac": ("LAC", "#0072B2", "o"),
        "socop_tuned": ("SOCOP, tuned λ", "#D55E00", "D"),
        "naive": ("Naive confidence", "#009E73", "^"),
    }
    figure, axis = plt.subplots(figsize=(7, 5.2), layout="constrained")

    for method, style in styles.items():
        parameter = "confidence_threshold" if method == "naive" else "alpha"
        method_rows = [row for row in plot_rows if row["method"] == method]
        method_rows = sorted(method_rows, key=lambda row: float(row[parameter]))
        mean_rows = [row for row in plot_summary if row["method"] == method]
        mean_rows = sorted(mean_rows, key=lambda row: float(row[parameter]))

        _plot_curve(
            axis, method_rows, mean_rows, style,
            "automation_rate", "automated_error_rate",
        )

    _style_axis(axis, "Frozen encoder", "Requests handled automatically")
    axis.set_ylabel("Errors among automated requests")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(bottom=0.0)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
