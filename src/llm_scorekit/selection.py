import numpy as np
from numpy.typing import ArrayLike, NDArray


# Identify prediction sets eligible for singleton automation.
def singleton_mask(prediction_sets: ArrayLike) -> NDArray[np.bool_]:
    included_classes = np.asarray(prediction_sets, dtype=np.bool_)

    if included_classes.ndim != 2:
        raise ValueError("prediction_sets must be two-dimensional")

    set_sizes = np.sum(included_classes, axis=1)
    singleton_samples = set_sizes == 1

    return singleton_samples
