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


# Calculate ranked SOCOP scores from the lower convex hull of set costs.
# Algorithm: https://arxiv.org/abs/2509.24095v2
def _socop_ranked_scores(
    ranked_probabilities: NDArray[np.float64],
    regularization: float,
) -> NDArray[np.float64]:
    number_of_classes = ranked_probabilities.size
    cumulative_probabilities = np.zeros(number_of_classes + 1)
    cumulative_probabilities[1:] = np.cumsum(ranked_probabilities)

    # Every label has a size cost; moving beyond a singleton adds another cost.
    set_sizes = np.arange(number_of_classes + 1, dtype=np.float64)
    set_costs = regularization * set_sizes
    set_costs[2:] += 1.0

    hull_vertices: list[int] = []

    for set_size in range(number_of_classes + 1):
        # Remove intermediate sizes that lie on or above the next hull segment.
        while len(hull_vertices) >= 2:
            first = hull_vertices[-2]
            second = hull_vertices[-1]

            previous_mass = cumulative_probabilities[second] - cumulative_probabilities[first]
            next_mass = cumulative_probabilities[set_size] - cumulative_probabilities[second]
            previous_cost = set_costs[second] - set_costs[first]
            next_cost = set_costs[set_size] - set_costs[second]

            cross_product = previous_mass * next_cost - previous_cost * next_mass
            if cross_product > 0.0:
                break

            hull_vertices.pop()

        hull_vertices.append(set_size)

    ranked_scores = np.empty(number_of_classes)

    # Labels on the same hull segment enter together and receive the same score.
    for start, end in zip(hull_vertices, hull_vertices[1:]):
        probability_gain = cumulative_probabilities[end] - cumulative_probabilities[start]
        cost_increase = set_costs[end] - set_costs[start]
        entry_score = cost_increase / probability_gain
        ranked_scores[start:end] = entry_score

    return ranked_scores


# Calculate full SOCOP scores for normalized, strictly positive probabilities.
def _socop_class_scores(
    probabilities: ArrayLike,
    regularization: float,
) -> NDArray[np.float64]:
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if predicted_probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")

    if not np.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("regularization must be finite and greater than zero")

    finite_probabilities = np.all(np.isfinite(predicted_probabilities))
    positive_probabilities = np.all(predicted_probabilities > 0.0)
    if not finite_probabilities or not positive_probabilities:
        raise ValueError("SOCOP probabilities must be finite and strictly positive")

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
    class_scores = np.empty_like(predicted_probabilities)

    for sample_index, probability_row in enumerate(ranked_probabilities):
        ranked_scores = _socop_ranked_scores(probability_row, regularization)
        class_indices = ranked_class_indices[sample_index]
        class_scores[sample_index, class_indices] = ranked_scores

    return class_scores


# Calculate SOCOP scores for true classes using a fixed positive regularization.
def socop_scores(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    regularization: float,
) -> NDArray[np.float64]:
    class_scores = _socop_class_scores(probabilities, regularization)
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
