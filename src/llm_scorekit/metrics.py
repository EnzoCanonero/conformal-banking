from math import sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike


# Measure the fraction of samples selected for automation.
def automation_rate(automation_mask: ArrayLike) -> float:
    automated_samples = np.asarray(automation_mask, dtype=np.bool_)

    if automated_samples.ndim != 1:
        raise ValueError("automation_mask must be one-dimensional")

    if automated_samples.size == 0:
        raise ValueError("automation_mask must contain at least one sample")

    rate = np.mean(automated_samples)

    return float(rate)


# Measure classification error among samples selected for automation.
# Measure the classification error rate on automated samples only.
def automated_error_rate(
    predictions: ArrayLike,
    labels: ArrayLike,
    automation_mask: ArrayLike,
) -> float:
    predicted_labels = np.asarray(predictions)
    true_labels = np.asarray(labels)
    automated_samples = np.asarray(automation_mask, dtype=np.bool_)

    if automated_samples.ndim != 1:
        raise ValueError("automation_mask must be one-dimensional")

    number_of_samples = automated_samples.size

    if number_of_samples == 0:
        raise ValueError("automation_mask must contain at least one sample")

    if predicted_labels.shape != automated_samples.shape:
        raise ValueError(
            "predictions and automation_mask must have the same shape"
        )

    if true_labels.shape != automated_samples.shape:
        raise ValueError("labels and automation_mask must have the same shape")

    # The error rate is undefined if no samples are automated.
    if not np.any(automated_samples):
        return float("nan")

    # Keep only the predictions and labels for automated samples.
    automated_predictions = predicted_labels[automated_samples]
    automated_labels = true_labels[automated_samples]

    # Check which automated predictions are incorrect.
    incorrect_predictions = automated_predictions != automated_labels

    # Compute the fraction of automated samples classified incorrectly.
    error_rate = np.mean(incorrect_predictions)

    return float(error_rate)


# Measure the fraction of samples whose prediction set contains the true class.
def empirical_coverage(prediction_sets: ArrayLike, labels: ArrayLike) -> float:
    included_classes = np.asarray(prediction_sets, dtype=np.bool_)
    true_labels = np.asarray(labels)

    if included_classes.ndim != 2:
        raise ValueError("prediction_sets must be two-dimensional")

    number_of_samples = included_classes.shape[0]

    if number_of_samples == 0:
        raise ValueError("prediction_sets must contain at least one sample")

    if true_labels.shape != (number_of_samples,):
        raise ValueError("labels must be one-dimensional with one label per sample")

    labels_below_range = np.any(true_labels < 0)

    if labels_below_range:
        raise ValueError("labels must refer to existing classes")

    # Check whether the true class is included for each sample.
    sample_indices = np.arange(number_of_samples)
    covered_samples = included_classes[sample_indices, true_labels]

    # Coverage is the fraction of samples for which this is true.
    coverage = np.mean(covered_samples)

    return float(coverage)


# Measure the average number of classes included per prediction set.
def average_set_size(prediction_sets: ArrayLike) -> float:
    included_classes = np.asarray(prediction_sets, dtype=np.bool_)

    if included_classes.ndim != 2:
        raise ValueError("prediction_sets must be two-dimensional")

    number_of_samples = included_classes.shape[0]

    if number_of_samples == 0:
        raise ValueError("prediction_sets must contain at least one sample")

    # Count how many classes are included in each prediction set.
    set_sizes = np.sum(included_classes, axis=1)

    # Average the set sizes over all samples.
    average_size = np.mean(set_sizes)

    return float(average_size)


# Estimate a binomial proportion's uncertainty with a Wilson score interval.
def binomial_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    if total < 0 or not 0 <= successes <= total:
        raise ValueError("counts must satisfy 0 <= successes <= total")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    if total == 0:
        return float("nan"), float("nan")

    proportion = successes / total
    upper_tail_probability = (1.0 + confidence_level) / 2.0
    normal_quantile = NormalDist().inv_cdf(upper_tail_probability)
    squared_quantile = normal_quantile**2

    denominator = 1.0 + squared_quantile / total
    adjusted_proportion = proportion + squared_quantile / (2.0 * total)
    interval_center = adjusted_proportion / denominator

    sampling_variance = proportion * (1.0 - proportion) / total
    variance_adjustment = squared_quantile / (4.0 * total**2)
    adjusted_standard_error = sqrt(sampling_variance + variance_adjustment)
    interval_radius = normal_quantile * adjusted_standard_error / denominator

    lower_bound = max(0.0, interval_center - interval_radius)
    upper_bound = min(1.0, interval_center + interval_radius)

    return lower_bound, upper_bound
