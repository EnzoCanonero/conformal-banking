from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from conformal_selective_prediction import (
    conformal_quantile,
    singleton_mask,
    socop_prediction_sets,
    socop_scores,
)

from ._data import PartitionPredictions


# Calibrate on one tuning half and retain evaluation counts from the other.
def _fold_counts(
    calibration_scores: NDArray[np.float64],
    selection_data: PartitionPredictions,
    alpha: float,
    regularization: float,
) -> tuple[int, int, int]:
    threshold = conformal_quantile(calibration_scores, alpha)
    prediction_sets = socop_prediction_sets(
        selection_data.probabilities,
        threshold,
        regularization=regularization,
    )
    automation_mask = singleton_mask(prediction_sets)
    automated_count = int(np.sum(automation_mask))
    set_size_total = int(np.sum(prediction_sets))
    sample_indices = np.arange(selection_data.labels.size)
    covered_samples = prediction_sets[sample_indices, selection_data.labels]
    covered_count = int(np.sum(covered_samples))

    return automated_count, set_size_total, covered_count


# Prefer more singletons, then smaller sets, then smaller regularization.
def _selection_key(candidate: dict[str, float]) -> tuple[float, float, float]:
    negative_automated_count = -candidate["combined_automated_count"]
    set_size_total = candidate["combined_set_size_total"]
    regularization = candidate["regularization"]

    return negative_automated_count, set_size_total, regularization


# Select each alpha's regularization using only the two tuning partitions.
def select_regularizations(
    tuning_a: PartitionPredictions,
    tuning_b: PartitionPredictions,
    alphas: Sequence[float],
    regularizations: Sequence[float],
) -> tuple[dict[float, float], list[dict[str, float]]]:
    if tuning_a.labels.size != tuning_b.labels.size:
        raise ValueError("SOCOP tuning halves must have the same number of samples")

    combined_sample_count = tuning_a.labels.size + tuning_b.labels.size
    candidate_rows: list[dict[str, float]] = []

    for regularization in regularizations:
        scores_a = socop_scores(
            tuning_a.probabilities,
            tuning_a.labels,
            regularization=regularization,
        )
        scores_b = socop_scores(
            tuning_b.probabilities,
            tuning_b.labels,
            regularization=regularization,
        )

        for alpha in alphas:
            # Evaluate A-calibrated sets on B, then swap the halves.
            automated_b, size_total_b, covered_b = _fold_counts(
                scores_a, tuning_b, alpha, regularization
            )
            automated_a, size_total_a, covered_a = _fold_counts(
                scores_b, tuning_a, alpha, regularization
            )

            # Equal halves make combined counts equivalent to arithmetic fold means.
            combined_automated_count = automated_a + automated_b
            combined_set_size_total = size_total_a + size_total_b
            combined_covered_count = covered_a + covered_b
            mean_automation_rate = combined_automated_count / combined_sample_count
            mean_set_size = combined_set_size_total / combined_sample_count
            mean_coverage = combined_covered_count / combined_sample_count

            candidate_rows.append(
                {
                    "alpha": alpha,
                    "regularization": regularization,
                    "mean_automation_rate": mean_automation_rate,
                    "mean_set_size": mean_set_size,
                    "mean_coverage": mean_coverage,
                    "combined_automated_count": combined_automated_count,
                    "combined_set_size_total": combined_set_size_total,
                    "combined_covered_count": combined_covered_count,
                }
            )

    selected_regularizations: dict[float, float] = {}

    for alpha in alphas:
        alpha_candidates = [row for row in candidate_rows if row["alpha"] == alpha]
        selected_candidate = min(alpha_candidates, key=_selection_key)
        selected_regularizations[alpha] = selected_candidate["regularization"]

    return selected_regularizations, candidate_rows
