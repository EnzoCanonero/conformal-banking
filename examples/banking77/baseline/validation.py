import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray

from conformal_selective_prediction import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    binomial_confidence_interval,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
    singleton_mask,
)
from conformal_selective_prediction.data import load_banking77
from conformal_selective_prediction.models import fit_tfidf_classifier


DATA_DIRECTORY = Path("data/raw/banking77")
OUTPUT_DIRECTORY = Path("outputs/banking77")
RANDOM_SEEDS = (7, 21, 42, 84, 123)
MIS_COVERAGE_RATES = (0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
SET_SIZE_PLOT_ALPHAS = (0.01, 0.05, 0.10, 0.30)
HIGHLIGHTED_ALPHA = 0.30
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CONFIDENCE_LEVEL = 0.95


@dataclass
class ValidationResults:
    lac: list[dict[str, float]] = field(default_factory=list)
    naive: list[dict[str, float]] = field(default_factory=list)
    class_coverage: list[dict[str, str | float]] = field(default_factory=list)
    set_sizes: list[dict[str, float]] = field(default_factory=list)


# Measure automation and its error, using only automated cases for the error interval.
def selection_metrics(
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    automation_mask: NDArray[np.bool_],
) -> dict[str, float]:
    automated_count = int(np.sum(automation_mask))
    incorrect_predictions = predictions != labels
    automated_errors = incorrect_predictions & automation_mask
    error_count = int(np.sum(automated_errors))

    selected_fraction = automation_rate(automation_mask)
    error_rate = automated_error_rate(predictions, labels, automation_mask)
    error_lower, error_upper = binomial_confidence_interval(
        error_count, automated_count, CONFIDENCE_LEVEL
    )

    return {
        "sample_count": labels.size,
        "automated_count": automated_count,
        "error_count": error_count,
        "automation_rate": selected_fraction,
        "abstention_rate": 1.0 - selected_fraction,
        "automated_error_rate": error_rate,
        "error_lower": error_lower,
        "error_upper": error_upper,
    }


# Retain per-intent counts and the complete distribution, including empty sets.
def record_set_diagnostics(
    results: ValidationResults,
    prediction_sets: NDArray[np.bool_],
    labels: NDArray[np.int64],
    class_names: list[str],
    random_seed: int,
    alpha: float,
) -> None:
    sample_indices = np.arange(labels.size)
    covered_samples = prediction_sets[sample_indices, labels]

    for class_index, class_name in enumerate(class_names):
        class_mask = labels == class_index
        sample_count = int(np.sum(class_mask))
        covered_count = int(np.sum(covered_samples[class_mask]))
        coverage = covered_count / sample_count
        lower_bound, upper_bound = binomial_confidence_interval(
            covered_count, sample_count, CONFIDENCE_LEVEL
        )
        results.class_coverage.append(
            {
                "seed": random_seed,
                "alpha": alpha,
                "class_index": class_index,
                "class_name": class_name,
                "sample_count": sample_count,
                "covered_count": covered_count,
                "coverage": coverage,
                "coverage_lower": lower_bound,
                "coverage_upper": upper_bound,
            }
        )

    set_sizes = np.sum(prediction_sets, axis=1)
    size_counts = np.bincount(set_sizes, minlength=len(class_names) + 1)

    for set_size, count in enumerate(size_counts):
        results.set_sizes.append(
            {
                "seed": random_seed,
                "alpha": alpha,
                "set_size": set_size,
                "count": int(count),
                "fraction": float(count / labels.size),
            }
        )


# Refit once per split and evaluate fixed policy grids on the unchanged official test set.
def run_validation(
    data_directory: str | Path,
    random_seeds: Sequence[int],
    alphas: Sequence[float],
    confidence_thresholds: Sequence[float],
) -> ValidationResults:
    results = ValidationResults()

    for random_seed in random_seeds:
        # Changing the seed changes training/calibration, never the test records.
        data = load_banking77(data_directory, random_seed=random_seed)
        model = fit_tfidf_classifier(data.train)
        expected_classes = np.arange(len(data.class_names))

        if not np.array_equal(model.classes_, expected_classes):
            raise ValueError("model classes must match the dataset label indices")

        calibration_probabilities = model.predict_proba(data.calibration.texts)
        calibration_scores = lac_scores(
            calibration_probabilities, data.calibration.labels
        )
        test_probabilities = model.predict_proba(data.test.texts)
        predicted_indices = np.argmax(test_probabilities, axis=1)
        predictions = model.classes_[predicted_indices]
        maximum_probabilities = np.max(test_probabilities, axis=1)
        accuracy = float(np.mean(predictions == data.test.labels))
        sample_indices = np.arange(data.test.labels.size)

        # Each alpha recalibrates a cutoff, without another model fit or test-based tuning.
        for alpha in sorted(alphas):
            threshold = conformal_quantile(calibration_scores, alpha)
            prediction_sets = lac_prediction_sets(test_probabilities, threshold)
            covered_samples = prediction_sets[sample_indices, data.test.labels]
            covered_count = int(np.sum(covered_samples))
            coverage_lower, coverage_upper = binomial_confidence_interval(
                covered_count, data.test.labels.size, CONFIDENCE_LEVEL
            )
            set_sizes = np.sum(prediction_sets, axis=1)
            automation_mask = singleton_mask(prediction_sets)

            row = {
                "seed": random_seed,
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "threshold": threshold,
                "accuracy": accuracy,
                "covered_count": covered_count,
                "coverage": empirical_coverage(prediction_sets, data.test.labels),
                "coverage_lower": coverage_lower,
                "coverage_upper": coverage_upper,
                "average_set_size": average_set_size(prediction_sets),
                "empty_set_rate": float(np.mean(set_sizes == 0)),
            }
            row.update(
                selection_metrics(predictions, data.test.labels, automation_mask)
            )
            results.lac.append(row)
            record_set_diagnostics(
                results,
                prediction_sets,
                data.test.labels,
                data.class_names,
                random_seed,
                alpha,
            )

        # Naive thresholds use the same probabilities, but no conformal calibration.
        for confidence_threshold in sorted(confidence_thresholds):
            automation_mask = maximum_probabilities >= confidence_threshold
            row = {
                "seed": random_seed,
                "confidence_threshold": confidence_threshold,
                "accuracy": accuracy,
            }
            row.update(
                selection_metrics(predictions, data.test.labels, automation_mask)
            )
            results.naive.append(row)

        print(f"Completed seed {random_seed}")

    return results


# Save each diagnostic table without pooling repeated observations across seeds.
def write_csv(file_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with file_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Keep the per-split coverage intervals separate from variability across splits.
def save_coverage_figure(rows: list[dict[str, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    random_seeds = sorted({row["seed"] for row in rows})

    for random_seed in random_seeds:
        seed_rows = [row for row in rows if row["seed"] == random_seed]
        seed_rows.sort(key=lambda row: row["target_coverage"])
        targets = [row["target_coverage"] for row in seed_rows]
        coverage = [row["coverage"] for row in seed_rows]
        lower_bounds = [row["coverage_lower"] for row in seed_rows]
        upper_bounds = [row["coverage_upper"] for row in seed_rows]

        (line,) = axis.plot(
            targets, coverage, marker="o", label=f"Seed {random_seed:g}"
        )
        axis.vlines(
            targets, lower_bounds, upper_bounds, color=line.get_color(), alpha=0.4
        )

    minimum_target = min(row["target_coverage"] for row in rows)
    axis.plot(
        [minimum_target, 1.0], [minimum_target, 1.0], color="black", linestyle="--"
    )
    axis.set_xlabel("Nominal coverage")
    axis.set_ylabel("Empirical coverage")
    axis.set_title(f"LAC coverage: pointwise {CONFIDENCE_LEVEL:.0%} Wilson intervals")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_coverage_size_figure(rows: list[dict[str, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    random_seeds = sorted({row["seed"] for row in rows})

    for random_seed in random_seeds:
        seed_rows = [row for row in rows if row["seed"] == random_seed]
        seed_rows.sort(key=lambda row: row["alpha"])
        average_sizes = [row["average_set_size"] for row in seed_rows]
        coverage = [row["coverage"] for row in seed_rows]
        axis.plot(average_sizes, coverage, marker="o", label=f"Seed {random_seed:g}")

    axis.set_xlabel("Average prediction-set size")
    axis.set_ylabel("Empirical coverage")
    axis.set_title("LAC coverage and set size by split")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# Follow policy-parameter order: singleton automation need not change monotonically.
def save_automation_figure(results: ValidationResults, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    methods = (
        ("LAC", results.lac, "alpha", "tab:blue", "o", "-"),
        ("Naive", results.naive, "confidence_threshold", "tab:orange", "s", "--"),
    )

    for method_name, rows, parameter, color, marker, line_style in methods:
        random_seeds = sorted({row["seed"] for row in rows})

        for seed_index, random_seed in enumerate(random_seeds):
            seed_rows = [row for row in rows if row["seed"] == random_seed]
            seed_rows.sort(key=lambda row: row[parameter])
            selected_rows = [
                row for row in seed_rows if np.isfinite(row["automated_error_rate"])
            ]
            automation = [row["automation_rate"] for row in selected_rows]
            error_rates = [row["automated_error_rate"] for row in selected_rows]
            label = method_name if seed_index == 0 else None

            axis.plot(
                automation,
                error_rates,
                color=color,
                marker=marker,
                linestyle=line_style,
                alpha=0.6,
                label=label,
            )

    # Highlight an observed trade-off, not an automatically selected best alpha.
    highlighted_rows = [
        row for row in results.lac if row["alpha"] == HIGHLIGHTED_ALPHA
    ]
    highlighted_automation = [row["automation_rate"] for row in highlighted_rows]
    highlighted_errors = [row["automated_error_rate"] for row in highlighted_rows]
    axis.scatter(
        highlighted_automation,
        highlighted_errors,
        color="tab:red",
        edgecolors="white",
        s=70,
        zorder=3,
        label=f"LAC alpha={HIGHLIGHTED_ALPHA:g} (best trade off)",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Automation rate")
    axis.set_ylabel("Error rate on automated cases")
    axis.set_title("Automation and error: one trace per split")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# Average per-split fractions descriptively; repeated test records are not new samples.
def save_set_size_figure(rows: list[dict[str, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    selected_rows = [row for row in rows if row["alpha"] in SET_SIZE_PLOT_ALPHAS]
    alphas = sorted({row["alpha"] for row in selected_rows})
    nonempty_bins = [int(row["set_size"]) for row in selected_rows if row["count"] > 0]
    maximum_size = max(nonempty_bins)
    bin_edges = np.arange(maximum_size + 2) - 0.5

    for alpha in alphas:
        alpha_rows = [row for row in selected_rows if row["alpha"] == alpha]
        seed_count = len({row["seed"] for row in alpha_rows})
        mean_fractions = np.zeros(maximum_size + 1)

        for row in alpha_rows:
            set_size = int(row["set_size"])
            if set_size <= maximum_size:
                mean_fractions[set_size] += row["fraction"]

        mean_fractions = mean_fractions / seed_count
        axis.stairs(mean_fractions, bin_edges, label=f"Alpha {alpha:g}")

    axis.set_xlim(-0.5, maximum_size + 0.5)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("Prediction-set size (0 means empty)")
    axis.set_ylabel("Mean fraction of test examples across splits")
    axis.set_title("LAC set-size distribution: descriptive split average")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_figures(results: ValidationResults, output_directory: Path) -> None:
    plt.switch_backend("Agg")
    save_coverage_figure(
        results.lac, output_directory / "nominal_vs_empirical_coverage.png"
    )
    save_coverage_size_figure(
        results.lac, output_directory / "coverage_vs_set_size.png"
    )
    save_automation_figure(results, output_directory / "automation_vs_error.png")
    save_set_size_figure(
        results.set_sizes, output_directory / "set_size_distribution.png"
    )


def main() -> None:
    results = run_validation(
        data_directory=DATA_DIRECTORY,
        random_seeds=RANDOM_SEEDS,
        alphas=MIS_COVERAGE_RATES,
        confidence_thresholds=CONFIDENCE_THRESHOLDS,
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIRECTORY / "lac_metrics.csv", results.lac)
    write_csv(OUTPUT_DIRECTORY / "naive_metrics.csv", results.naive)
    write_csv(OUTPUT_DIRECTORY / "class_coverage.csv", results.class_coverage)
    write_csv(OUTPUT_DIRECTORY / "set_sizes.csv", results.set_sizes)

    configuration = {
        "data_directory": str(DATA_DIRECTORY),
        "random_seeds": RANDOM_SEEDS,
        "alphas": MIS_COVERAGE_RATES,
        "confidence_thresholds": CONFIDENCE_THRESHOLDS,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "Wilson (pointwise, per split)",
        "numpy_version": version("numpy"),
        "scikit_learn_version": version("scikit-learn"),
        "matplotlib_version": version("matplotlib"),
    }
    with (OUTPUT_DIRECTORY / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(configuration, config_file, indent=2)
        config_file.write("\n")

    save_figures(results, OUTPUT_DIRECTORY)
    print(f"Results saved to {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
