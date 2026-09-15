import numpy as np
from numpy.typing import ArrayLike, NDArray


# Calculate LAC scores from the probabilities assigned to the true classes.
def lac_scores(probabilities: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)
    true_labels = np.asarray(labels)

    if predicted_probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")

    number_of_samples = predicted_probabilities.shape[0]

    if true_labels.shape != (number_of_samples,):
        raise ValueError("labels must be one-dimensional with one label per sample")

    labels_below_range = np.any(true_labels < 0)

    if labels_below_range:
        raise ValueError("labels must refer to existing classes")

    # For each sample, select the probability assigned to its true class.
    sample_indices = np.arange(number_of_samples)
    true_class_probabilities = predicted_probabilities[sample_indices, true_labels]

    # The LAC score is one minus the probability of the true class.
    # Higher scores therefore correspond to lower confidence in the true label.
    scores = 1.0 - true_class_probabilities

    return scores


# Calculate deterministic APS scores for every class, including its own probability.
def _aps_class_scores(probabilities: ArrayLike) -> NDArray[np.float64]:
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if predicted_probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")

    # Rank from highest probability to lowest, keeping class order for ties.
    negative_probabilities = -predicted_probabilities
    ranked_class_indices = np.argsort(
        negative_probabilities,
        axis=1,
        kind="stable",
    )
    ranked_probabilities = np.take_along_axis(
        predicted_probabilities,
        ranked_class_indices,
        axis=1,
    )
    cumulative_probabilities = np.cumsum(ranked_probabilities, axis=1)

    # Restore the original class columns before selecting labels or building sets.
    class_scores = np.empty_like(predicted_probabilities)
    np.put_along_axis(
        class_scores,
        ranked_class_indices,
        cumulative_probabilities,
        axis=1,
    )

    return class_scores


# Calculate non-randomized APS scores for the true classes.
def aps_scores(probabilities: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
    class_scores = _aps_class_scores(probabilities)
    true_labels = np.asarray(labels)
    number_of_samples = class_scores.shape[0]

    if true_labels.shape != (number_of_samples,):
        raise ValueError("labels must be one-dimensional with one label per sample")

    labels_below_range = np.any(true_labels < 0)

    if labels_below_range:
        raise ValueError("labels must refer to existing classes")

    sample_indices = np.arange(number_of_samples)
    scores = class_scores[sample_indices, true_labels]

    return scores
