from math import ceil

import numpy as np
from numpy.typing import ArrayLike


# Calculate the finite-sample split-conformal threshold.
def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    calibration_scores = np.asarray(scores, dtype=np.float64)

    #some checks
    if calibration_scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")

    if calibration_scores.size == 0:
        raise ValueError("scores must contain at least one value")

    if not np.all(np.isfinite(calibration_scores)):
        raise ValueError("scores must contain only finite values")

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")

    corrected_sample_size = calibration_scores.size + 1
    target_coverage = 1.0 - alpha
    raw_quantile_rank = corrected_sample_size * target_coverage
    quantile_rank = ceil(raw_quantile_rank)

    if quantile_rank > calibration_scores.size:
        return float("inf")

    sorted_scores = np.sort(calibration_scores)

    # raw_quantile_rank starts from 1, subtract 1 to match Python indexing
    quantile_index = quantile_rank - 1
    threshold = sorted_scores[quantile_index]

    return float(threshold)
