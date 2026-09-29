from pathlib import Path

from examples.banking77.scores._evaluation import naive_metrics

from ._data import load_predictions
from ._evaluation import CONFIDENCE_LEVEL, CONFIDENCE_THRESHOLDS, save_evaluation
from .prepare import CACHE_PATH, STUDY_DIRECTORY


OUTPUT_DIRECTORY = STUDY_DIRECTORY / "naive"


# Evaluate fixed confidence cutoffs without selecting a cutoff on the test results.
def run_experiment(
    cache_path: Path = CACHE_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> None:
    partitions, metadata = load_predictions(cache_path)
    metrics = naive_metrics(partitions["test"], CONFIDENCE_THRESHOLDS, CONFIDENCE_LEVEL)
    save_evaluation(
        output_directory,
        cache_path,
        metadata,
        settings={"method": "naive", "confidence_thresholds": CONFIDENCE_THRESHOLDS},
        tables={"metrics": metrics},
    )


if __name__ == "__main__":
    run_experiment()
