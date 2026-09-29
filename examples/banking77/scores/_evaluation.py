from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from conformal_selective_prediction import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    binomial_confidence_interval,
    empirical_coverage,
    singleton_mask,
)

from ._data import PartitionPredictions


# Measure routing error and its interval using only automated requests.
def selection_metrics(
    predictions: NDArray[np.int64],
    labels: NDArray[np.int64],
    automation_mask: NDArray[np.bool_],
    confidence_level: float,
) -> dict[str, float]:
    automated_count = int(np.sum(automation_mask))
    incorrect_predictions = predictions != labels
    automated_errors = incorrect_predictions & automation_mask
    error_count = int(np.sum(automated_errors))

    selected_fraction = automation_rate(automation_mask)
    error_rate = automated_error_rate(predictions, labels, automation_mask)
    error_lower, error_upper = binomial_confidence_interval(
        error_count,
        automated_count,
        confidence_level,
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


# Measure prediction-set coverage separately from singleton-routing error.
def prediction_set_metrics(
    prediction_sets: NDArray[np.bool_],
    test_data: PartitionPredictions,
    confidence_level: float,
) -> dict[str, float]:
    labels = test_data.labels
    sample_indices = np.arange(labels.size)
    covered_samples = prediction_sets[sample_indices, labels]
    covered_count = int(np.sum(covered_samples))
    coverage_lower, coverage_upper = binomial_confidence_interval(
        covered_count,
        labels.size,
        confidence_level,
    )

    baseline_predictions = np.argmax(test_data.probabilities, axis=1)
    correct_predictions = baseline_predictions == labels
    accuracy = float(np.mean(correct_predictions))
    set_sizes = np.sum(prediction_sets, axis=1)
    empty_sets = set_sizes == 0
    empty_set_rate = float(np.mean(empty_sets))

    # Route the actual singleton label; mask out empty and multi-label sets.
    automation_mask = singleton_mask(prediction_sets)
    singleton_predictions = np.argmax(prediction_sets, axis=1)
    routing_metrics = selection_metrics(
        singleton_predictions,
        labels,
        automation_mask,
        confidence_level,
    )

    metrics = {
        "accuracy": accuracy,
        "covered_count": covered_count,
        "coverage": empirical_coverage(prediction_sets, labels),
        "coverage_lower": coverage_lower,
        "coverage_upper": coverage_upper,
        "average_set_size": average_set_size(prediction_sets),
        "empty_set_rate": empty_set_rate,
    }
    metrics.update(routing_metrics)

    return metrics


# Evaluate fixed confidence cutoffs on the same baseline probabilities.
def naive_metrics(
    test_data: PartitionPredictions,
    confidence_thresholds: Sequence[float],
    confidence_level: float,
) -> list[dict[str, float]]:
    predictions = np.argmax(test_data.probabilities, axis=1)
    maximum_probabilities = np.max(test_data.probabilities, axis=1)
    correct_predictions = predictions == test_data.labels
    accuracy = float(np.mean(correct_predictions))
    rows: list[dict[str, float]] = []

    for confidence_threshold in sorted(confidence_thresholds):
        automation_mask = maximum_probabilities >= confidence_threshold
        routing_metrics = selection_metrics(
            predictions,
            test_data.labels,
            automation_mask,
            confidence_level,
        )
        row = {
            "confidence_threshold": confidence_threshold,
            "accuracy": accuracy,
        }
        row.update(routing_metrics)
        rows.append(row)

    return rows
