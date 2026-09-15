import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scores import _aps_class_scores


# Construct LAC prediction sets from class probabilities.
def lac_prediction_sets(
    probabilities: ArrayLike,
    threshold: float,
) -> NDArray[np.bool_]:
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if predicted_probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")

    # Compute a LAC score for each class and return a boolean vector indicating
    # which classes are included in the prediction set.
    class_scores = 1.0 - predicted_probabilities
    prediction_sets = class_scores <= threshold

    return prediction_sets


# Construct APS sets using the same deterministic scores used during calibration.
def aps_prediction_sets(
    probabilities: ArrayLike,
    threshold: float,
) -> NDArray[np.bool_]:
    class_scores = _aps_class_scores(probabilities)

    # Include scores at the threshold, without adding a label that crosses it.
    prediction_sets = class_scores <= threshold

    return prediction_sets
