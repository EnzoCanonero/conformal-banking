from pathlib import Path

import numpy as np

from conformal_selective_prediction import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    conformal_quantile,
    empirical_coverage,
    lac_prediction_sets,
    lac_scores,
    singleton_mask,
)
from conformal_selective_prediction.data import load_banking77
from conformal_selective_prediction.models import fit_tfidf_classifier


DATA_DIRECTORY = Path("data/raw/banking77")
RANDOM_SEED = 42
MIS_COVERAGE_RATE = 0.1
CONFIDENCE_THRESHOLD = 0.5


# Compare LAC singleton selection and naive thresholding on one BANKING77 split.
def run_experiment(
    data_directory: str | Path,
    random_seed: int,
    alpha: float,
    confidence_threshold: float,
) -> dict[str, float]:
    # Fit both TF-IDF and the classifier on training data only.
    data = load_banking77(data_directory, random_seed=random_seed)
    model = fit_tfidf_classifier(data.train)

    # LAC uses label IDs as probability-column indices, so their order must match.
    expected_classes = np.arange(len(data.class_names))
    if not np.array_equal(model.classes_, expected_classes):
        raise ValueError("model classes must match the dataset label indices")

    # Estimate the LAC cutoff from true-label scores on calibration data only.
    calibration_probabilities = model.predict_proba(data.calibration.texts)
    calibration_scores = lac_scores(calibration_probabilities, data.calibration.labels)
    threshold = conformal_quantile(calibration_scores, alpha)

    # Reuse the same test probabilities and top-label predictions for both policies.
    test_probabilities = model.predict_proba(data.test.texts)
    predicted_class_indices = np.argmax(test_probabilities, axis=1)
    test_predictions = model.classes_[predicted_class_indices]
    correct_predictions = test_predictions == data.test.labels
    accuracy = float(np.mean(correct_predictions))

    # Set coverage measures true-label inclusion over all test cases, before selection.
    prediction_sets = lac_prediction_sets(test_probabilities, threshold)
    coverage = empirical_coverage(prediction_sets, data.test.labels)
    mean_set_size = average_set_size(prediction_sets)

    # Defer empty and multi-label sets; a LAC singleton contains the top-probability class.
    lac_automation_mask = singleton_mask(prediction_sets)
    lac_automated_fraction = automation_rate(lac_automation_mask)
    lac_automated_error = automated_error_rate(
        test_predictions,
        data.test.labels,
        lac_automation_mask,
    )

    # Naive selection uses a fixed confidence cutoff, not the LAC score threshold.
    maximum_probabilities = np.max(test_probabilities, axis=1)
    naive_automation_mask = maximum_probabilities >= confidence_threshold
    naive_automated_fraction = automation_rate(naive_automation_mask)
    naive_automated_error = automated_error_rate(
        test_predictions,
        data.test.labels,
        naive_automation_mask,
    )

    results = {
        "accuracy": accuracy,
        "lac_threshold": threshold,
        "lac_coverage": coverage,
        "lac_average_set_size": mean_set_size,
        "lac_automation_rate": lac_automated_fraction,
        "lac_automated_error_rate": lac_automated_error,
        "naive_automation_rate": naive_automated_fraction,
        "naive_automated_error_rate": naive_automated_error,
    }

    return results


def main() -> None:
    results = run_experiment(
        data_directory=DATA_DIRECTORY,
        random_seed=RANDOM_SEED,
        alpha=MIS_COVERAGE_RATE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )
    target_coverage = 1.0 - MIS_COVERAGE_RATE

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Classifier accuracy: {results['accuracy']:.3f}")
    print(f"LAC target coverage: {target_coverage:.3f}")
    print(f"LAC calibrated score threshold: {results['lac_threshold']:.3f}")
    print(f"LAC empirical coverage: {results['lac_coverage']:.3f}")
    print(f"LAC average set size: {results['lac_average_set_size']:.3f}")
    print(f"LAC automation rate: {results['lac_automation_rate']:.3f}")
    print(f"LAC automated-case error rate: {results['lac_automated_error_rate']:.3f}")
    print(f"Naive confidence threshold: {CONFIDENCE_THRESHOLD:.3f}")
    print(f"Naive automation rate: {results['naive_automation_rate']:.3f}")
    print(f"Naive automated-case error rate: {results['naive_automated_error_rate']:.3f}")


if __name__ == "__main__":
    main()
