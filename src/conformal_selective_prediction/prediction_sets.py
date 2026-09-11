import numpy as np
from numpy.typing import ArrayLike, NDArray


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
