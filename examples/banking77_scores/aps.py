from importlib.metadata import version
from pathlib import Path

from conformal_selective_prediction import (
    aps_prediction_sets,
    aps_scores,
    conformal_quantile,
)

from ._artifacts import load_manifest, load_prepared_seed, save_csv, save_manifest
from ._diagnostics import class_coverage_rows, set_size_rows
from ._evaluation import naive_metrics, prediction_set_metrics


PREPARED_DIRECTORY = Path("outputs/banking77/score_comparison/prepared")
OUTPUT_DIRECTORY = Path("outputs/banking77/score_comparison/aps")


# Evaluate deterministic APS and the naive rule on the shared predictions.
def run_experiment(prepared_directory: Path, output_directory: Path) -> None:
    manifest = load_manifest(prepared_directory)
    preparation_id = manifest["preparation_id"]
    confidence_level = manifest["confidence_level"]
    alphas = manifest["alphas"]
    confidence_thresholds = manifest["confidence_thresholds"]

    metrics: list[dict[str, str | float]] = []
    naive_results: list[dict[str, str | float]] = []
    class_coverage: list[dict[str, str | float]] = []
    set_sizes: list[dict[str, str | float]] = []

    for random_seed in manifest["random_seeds"]:
        data = load_prepared_seed(prepared_directory, random_seed, preparation_id)

        # APS needs no tuning; only final calibration determines its thresholds.
        calibration_scores = aps_scores(
            data.calibration.probabilities, data.calibration.labels
        )

        for alpha in alphas:
            threshold = conformal_quantile(calibration_scores, alpha)

            # Keep the calibrated set as-is, without adding a threshold-crossing label.
            prediction_sets = aps_prediction_sets(data.test.probabilities, threshold)
            configuration: dict[str, str | float] = {
                "method": "aps",
                "seed": random_seed,
                "alpha": alpha,
            }

            row = configuration.copy()
            row["target_coverage"] = 1.0 - alpha
            row["threshold"] = threshold
            row["calibration_count"] = data.calibration.labels.size
            measured_metrics = prediction_set_metrics(
                prediction_sets, data.test, confidence_level
            )
            row.update(measured_metrics)
            metrics.append(row)

            class_rows = class_coverage_rows(
                prediction_sets, data.test.labels, data.class_names, confidence_level
            )
            for class_row in class_rows:
                diagnostic_row = configuration.copy()
                diagnostic_row.update(class_row)
                class_coverage.append(diagnostic_row)

            size_rows = set_size_rows(prediction_sets)
            for size_row in size_rows:
                diagnostic_row = configuration.copy()
                diagnostic_row.update(size_row)
                set_sizes.append(diagnostic_row)

        # Recompute the same cheap baseline so this experiment runs independently.
        naive_rows = naive_metrics(data.test, confidence_thresholds, confidence_level)
        for naive_row in naive_rows:
            row = {"method": "naive", "seed": random_seed}
            row.update(naive_row)
            naive_results.append(row)

        print(f"Completed APS seed {random_seed}")

    output_directory.mkdir(parents=True, exist_ok=True)
    save_csv(output_directory / "metrics.csv", metrics)
    save_csv(output_directory / "naive_metrics.csv", naive_results)
    save_csv(output_directory / "class_coverage.csv", class_coverage)
    save_csv(output_directory / "set_sizes.csv", set_sizes)

    result_manifest = manifest.copy()
    result_manifest["methods"] = ["aps", "naive"]
    result_manifest["prepared_directory"] = str(prepared_directory)
    result_manifest["evaluation_numpy_version"] = version("numpy")
    result_manifest["aps_score"] = "deterministic cumulative-through-label, stable ties"
    result_manifest["aps_set_rule"] = "score <= threshold, no crossing label added"
    save_manifest(output_directory, result_manifest)
    print(f"APS results saved to {output_directory}")


def main() -> None:
    run_experiment(PREPARED_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
