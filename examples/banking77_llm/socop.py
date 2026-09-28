from pathlib import Path

from conformal_selective_prediction import (
    conformal_quantile,
    socop_prediction_sets,
    socop_scores,
)
from examples.banking77_scores._diagnostics import set_size_rows
from examples.banking77_scores._socop_tuning import select_regularizations

from ._data import load_predictions
from ._evaluation import ALPHAS, SOCOP_REGULARIZATIONS, evaluate_sets, save_evaluation
from .prepare import CACHE_PATH, STUDY_DIRECTORY


OUTPUT_DIRECTORY = STUDY_DIRECTORY / "socop"


# Select lambda using A/B only, then calibrate and evaluate the frozen choices.
def run_experiment(
    cache_path: Path = CACHE_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> None:
    partitions, metadata = load_predictions(cache_path)
    regularizations, candidates = select_regularizations(
        partitions["tuning_a"],
        partitions["tuning_b"],
        ALPHAS,
        SOCOP_REGULARIZATIONS,
    )
    calibration = partitions["calibration"]
    test = partitions["test"]
    metrics: list[dict[str, str | float]] = []
    set_sizes: list[dict[str, float]] = []
    choices: list[dict[str, float]] = []

    for alpha in ALPHAS:
        regularization = regularizations[alpha]
        choices.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "regularization": regularization,
            }
        )

        # Final calibration never participates in selecting lambda.
        calibration_scores = socop_scores(
            calibration.probabilities,
            calibration.labels,
            regularization=regularization,
        )
        threshold = conformal_quantile(calibration_scores, alpha)
        prediction_sets = socop_prediction_sets(
            test.probabilities, threshold, regularization=regularization
        )
        row: dict[str, str | float] = {
            "method": "socop_tuned",
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
            "regularization": regularization,
            "threshold": threshold,
            "calibration_count": calibration.labels.size,
        }
        row.update(evaluate_sets(prediction_sets, test))
        metrics.append(row)

        for size_row in set_size_rows(prediction_sets):
            size_row["alpha"] = alpha
            set_sizes.append(size_row)

    save_evaluation(
        output_directory,
        cache_path,
        metadata,
        settings={
            "method": "socop_tuned",
            "alphas": ALPHAS,
            "regularizations": SOCOP_REGULARIZATIONS,
            "tuning": "swap calibration and selection roles of tuning A and B",
            "objective": "highest combined singleton count",
            "tie_breaks": ["smaller total set size", "smaller regularization"],
        },
        tables={
            "metrics": metrics,
            "set_sizes": set_sizes,
            "tuning_candidates": candidates,
            "tuning_choices": choices,
        },
    )


if __name__ == "__main__":
    run_experiment()
