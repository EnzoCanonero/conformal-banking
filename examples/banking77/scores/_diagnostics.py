import numpy as np
from numpy.typing import NDArray

from conformal_selective_prediction import binomial_confidence_interval


# Keep class-level counts and coverage intervals separate from marginal coverage.
def class_coverage_rows(
    prediction_sets: NDArray[np.bool_],
    labels: NDArray[np.int64],
    class_names: list[str],
    confidence_level: float,
) -> list[dict[str, str | float]]:
    sample_indices = np.arange(labels.size)
    covered_samples = prediction_sets[sample_indices, labels]
    rows: list[dict[str, str | float]] = []

    for class_index, class_name in enumerate(class_names):
        class_mask = labels == class_index
        sample_count = int(np.sum(class_mask))
        covered_count = int(np.sum(covered_samples[class_mask]))
        coverage = covered_count / sample_count
        lower_bound, upper_bound = binomial_confidence_interval(
            covered_count, sample_count, confidence_level
        )

        rows.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "sample_count": sample_count,
                "covered_count": covered_count,
                "coverage": coverage,
                "coverage_lower": lower_bound,
                "coverage_upper": upper_bound,
            }
        )

    return rows


# Retain every possible set size, including empty sets and zero-count bins.
def set_size_rows(prediction_sets: NDArray[np.bool_]) -> list[dict[str, float]]:
    sample_count, class_count = prediction_sets.shape
    set_sizes = np.sum(prediction_sets, axis=1)
    size_counts = np.bincount(set_sizes, minlength=class_count + 1)
    rows = []

    for set_size, count in enumerate(size_counts):
        fraction = float(count / sample_count)
        rows.append(
            {
                "set_size": set_size,
                "count": int(count),
                "fraction": fraction,
            }
        )

    return rows
