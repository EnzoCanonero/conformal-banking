from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .calibration import conformal_quantile
from .prediction_sets import aps_prediction_sets, lac_prediction_sets, socop_prediction_sets
from .scores import aps_scores, lac_scores, socop_scores
from .selection import singleton_mask


ScoreMethod = Literal["lac", "aps", "socop"]


# Predictions are class-column indices, or None for empty and multi-label sets.
@dataclass(frozen=True)
class DecisionBatch:
    prediction_sets: NDArray[np.bool_]
    accepted: NDArray[np.bool_]
    predictions: list[int | None]


# Keep the method and class order fixed after calibration.
@dataclass(frozen=True)
class CalibratedPolicy:
    method: ScoreMethod
    alpha: float
    threshold: float
    class_names: tuple[str, ...]
    regularization: float | None = None

    # Apply the policy using the same scoring procedure and class-column order.
    def predict(self, probabilities: ArrayLike) -> DecisionBatch:
        number_of_classes = len(self.class_names)
        predicted_probabilities = _validate_probabilities(probabilities, number_of_classes)

        if self.method == "lac":
            prediction_sets = lac_prediction_sets(predicted_probabilities, self.threshold)
        elif self.method == "aps":
            prediction_sets = aps_prediction_sets(predicted_probabilities, self.threshold)
        elif self.method == "socop":
            if self.regularization is None:
                raise ValueError("SOCOP requires regularization")
            prediction_sets = socop_prediction_sets(
                predicted_probabilities,
                self.threshold,
                regularization=self.regularization,
            )
        else:
            raise ValueError("method must be 'lac', 'aps' or 'socop'")

        accepted = singleton_mask(prediction_sets)
        number_of_samples = predicted_probabilities.shape[0]
        predictions: list[int | None] = [None] * number_of_samples

        # Read only singleton sets so deferred requests never receive a class.
        accepted_indices = np.flatnonzero(accepted)
        for sample_index in accepted_indices:
            included_classes = np.flatnonzero(prediction_sets[sample_index])
            predictions[sample_index] = int(included_classes[0])

        return DecisionBatch(
            prediction_sets=prediction_sets,
            accepted=accepted,
            predictions=predictions,
        )


# Check normalized class weights without rescaling them or applying softmax.
def _validate_probabilities(
    probabilities: ArrayLike,
    number_of_classes: int,
) -> NDArray[np.float64]:
    predicted_probabilities = np.asarray(probabilities, dtype=np.float64)

    if predicted_probabilities.ndim != 2:
        raise ValueError("probabilities must be two-dimensional")
    if number_of_classes == 0:
        raise ValueError("at least one class is required")
    if predicted_probabilities.shape[1] != number_of_classes:
        raise ValueError("probability columns must match class_names")

    finite_values = np.all(np.isfinite(predicted_probabilities))
    nonnegative_values = np.all(predicted_probabilities >= 0.0)
    if not finite_values or not nonnegative_values:
        raise ValueError("probabilities must be finite and nonnegative")

    row_totals = np.sum(predicted_probabilities, axis=1)
    if not np.allclose(row_totals, 1.0):
        raise ValueError("each probability row must sum to one")

    return predicted_probabilities


# Calibrate held-out normalized weights; labels are integer class-column indices.
# Fix the scoring procedure and any SOCOP tuning before using these examples.
def calibrate(
    probabilities: ArrayLike,
    labels: ArrayLike,
    *,
    class_names: Sequence[str],
    method: ScoreMethod = "lac",
    alpha: float = 0.10,
    regularization: float | None = None,
) -> CalibratedPolicy:
    if isinstance(class_names, str):
        raise ValueError("class_names must be a sequence of names, not a single string")

    ordered_class_names = tuple(class_names)
    if not all(isinstance(name, str) for name in ordered_class_names):
        raise ValueError("class_names must contain strings")
    if len(set(ordered_class_names)) != len(ordered_class_names):
        raise ValueError("class_names must be unique")

    number_of_classes = len(ordered_class_names)
    calibration_probabilities = _validate_probabilities(probabilities, number_of_classes)
    true_labels = np.asarray(labels)
    number_of_samples = calibration_probabilities.shape[0]

    if true_labels.shape != (number_of_samples,):
        raise ValueError("labels must contain one class index per calibration sample")
    if not np.issubdtype(true_labels.dtype, np.integer):
        raise ValueError("labels must be integer class-column indices")

    labels_below_range = np.any(true_labels < 0)
    labels_above_range = np.any(true_labels >= number_of_classes)
    if labels_below_range or labels_above_range:
        raise ValueError("labels must refer to existing classes")

    if method != "socop" and regularization is not None:
        raise ValueError("regularization is only used by SOCOP")

    if method == "lac":
        scores = lac_scores(calibration_probabilities, true_labels)
    elif method == "aps":
        scores = aps_scores(calibration_probabilities, true_labels)
    elif method == "socop":
        if regularization is None:
            raise ValueError("SOCOP requires regularization")
        scores = socop_scores(
            calibration_probabilities,
            true_labels,
            regularization=regularization,
        )
    else:
        raise ValueError("method must be 'lac', 'aps' or 'socop'")

    threshold = conformal_quantile(scores, alpha)

    return CalibratedPolicy(
        method=method,
        alpha=alpha,
        threshold=threshold,
        class_names=ordered_class_names,
        regularization=regularization,
    )
