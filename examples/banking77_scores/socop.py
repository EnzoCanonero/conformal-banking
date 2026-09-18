from importlib.metadata import version
from pathlib import Path

from conformal_selective_prediction import (
    conformal_quantile,
    socop_prediction_sets,
    socop_scores,
)

from ._artifacts import load_manifest, load_prepared_seed, save_csv, save_manifest
from ._data import PreparedData
from ._diagnostics import class_coverage_rows, set_size_rows
from ._evaluation import naive_metrics, prediction_set_metrics
from ._socop_tuning import select_regularizations


PREPARED_DIRECTORY = Path("outputs/banking77/score_comparison/prepared")
OUTPUT_DIRECTORY = Path("outputs/banking77/score_comparison/socop")


# Evaluate one SOCOP method after its regularization choices have been frozen.
def _evaluate_method(
    data: PreparedData,
    random_seed: int,
    method: str,
    regularizations: dict[float, float],
    confidence_level: float,
) -> tuple[
    list[dict[str, str | float]],
    list[dict[str, str | float]],
    list[dict[str, str | float]],
]:
    metrics: list[dict[str, str | float]] = []
    class_coverage: list[dict[str, str | float]] = []
    set_sizes: list[dict[str, str | float]] = []

    for alpha, regularization in regularizations.items():
        # Recalibrate on the reserved partition, not on either tuning half.
        calibration_scores = socop_scores(
            data.calibration.probabilities,
            data.calibration.labels,
            regularization=regularization,
        )
        threshold = conformal_quantile(calibration_scores, alpha)
        prediction_sets = socop_prediction_sets(
            data.test.probabilities, threshold, regularization=regularization
        )
        configuration: dict[str, str | float] = {
            "method": method,
            "seed": random_seed,
            "alpha": alpha,
            "regularization": regularization,
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

    return metrics, class_coverage, set_sizes


# Tune on held-out training data, then evaluate fixed and tuned SOCOP separately.
def run_experiment(prepared_directory: Path, output_directory: Path) -> None:
    manifest = load_manifest(prepared_directory)
    preparation_id = manifest["preparation_id"]
    confidence_level = manifest["confidence_level"]
    alphas = manifest["alphas"]
    candidate_regularizations = manifest["socop_regularizations"]
    reference_regularization = manifest["socop_reference_regularization"]
    confidence_thresholds = manifest["confidence_thresholds"]

    metrics: list[dict[str, str | float]] = []
    naive_results: list[dict[str, str | float]] = []
    class_coverage: list[dict[str, str | float]] = []
    set_sizes: list[dict[str, str | float]] = []
    tuning_candidates: list[dict[str, float]] = []
    tuning_choices: list[dict[str, float]] = []

    for random_seed in manifest["random_seeds"]:
        data = load_prepared_seed(prepared_directory, random_seed, preparation_id)

        # This function can access only tuning data, never final calibration or test.
        selected_regularizations, candidate_rows = select_regularizations(
            data.tuning_a, data.tuning_b, alphas, candidate_regularizations
        )
        for candidate_row in candidate_rows:
            tuning_row = {"seed": random_seed}
            tuning_row.update(candidate_row)
            tuning_candidates.append(tuning_row)

        fixed_regularizations = {}
        for alpha in alphas:
            fixed_regularizations[alpha] = reference_regularization
            tuning_choices.append(
                {
                    "seed": random_seed,
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "regularization": selected_regularizations[alpha],
                }
            )

        # Freeze all per-target choices before evaluating either method on test data.
        methods = {
            "socop_fixed": fixed_regularizations,
            "socop_tuned": selected_regularizations,
        }
        for method, regularizations in methods.items():
            method_metrics, method_class_rows, method_size_rows = _evaluate_method(
                data, random_seed, method, regularizations, confidence_level
            )
            metrics.extend(method_metrics)
            class_coverage.extend(method_class_rows)
            set_sizes.extend(method_size_rows)

        naive_rows = naive_metrics(data.test, confidence_thresholds, confidence_level)
        for naive_row in naive_rows:
            row: dict[str, str | float] = {"method": "naive", "seed": random_seed}
            row.update(naive_row)
            naive_results.append(row)

        print(f"Completed SOCOP seed {random_seed}")

    output_directory.mkdir(parents=True, exist_ok=True)
    save_csv(output_directory / "metrics.csv", metrics)
    save_csv(output_directory / "naive_metrics.csv", naive_results)
    save_csv(output_directory / "class_coverage.csv", class_coverage)
    save_csv(output_directory / "set_sizes.csv", set_sizes)
    save_csv(output_directory / "tuning_candidates.csv", tuning_candidates)
    save_csv(output_directory / "tuning_choices.csv", tuning_choices)

    result_manifest = manifest.copy()
    result_manifest["methods"] = ["socop_fixed", "socop_tuned", "naive"]
    result_manifest["prepared_directory"] = str(prepared_directory)
    result_manifest["evaluation_numpy_version"] = version("numpy")
    result_manifest["socop_score"] = "full positive-regularization, lower convex hull"
    result_manifest["socop_set_rule"] = "score <= threshold, including empty sets"
    save_manifest(output_directory, result_manifest)
    print(f"SOCOP results saved to {output_directory}")


def main() -> None:
    run_experiment(PREPARED_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
