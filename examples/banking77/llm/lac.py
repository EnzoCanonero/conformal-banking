from pathlib import Path

from llm_scorekit import (
    conformal_quantile,
    lac_prediction_sets,
    lac_scores,
)
from examples.banking77.scores._diagnostics import set_size_rows

from ._data import load_predictions
from ._evaluation import ALPHAS, evaluate_sets, save_evaluation
from .prepare import CACHE_PATH, STUDY_DIRECTORY


OUTPUT_DIRECTORY = STUDY_DIRECTORY / "lac"


# Calibrate LAC on the reserved examples and evaluate singleton routing on test.
def run_experiment(
    cache_path: Path = CACHE_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> None:
    partitions, metadata = load_predictions(cache_path)
    calibration = partitions["calibration"]
    test = partitions["test"]
    calibration_scores = lac_scores(calibration.probabilities, calibration.labels)
    metrics: list[dict[str, str | float]] = []
    set_sizes: list[dict[str, float]] = []

    for alpha in ALPHAS:
        threshold = conformal_quantile(calibration_scores, alpha)
        prediction_sets = lac_prediction_sets(test.probabilities, threshold)
        row: dict[str, str | float] = {
            "method": "lac",
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
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
        settings={"method": "lac", "alphas": ALPHAS},
        tables={"metrics": metrics, "set_sizes": set_sizes},
    )


if __name__ == "__main__":
    run_experiment()
