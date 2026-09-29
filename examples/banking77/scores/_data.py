from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from conformal_selective_prediction.data import TextDataset


@dataclass
class PartitionPredictions:
    probabilities: NDArray[np.float64]
    labels: NDArray[np.int64]
    sample_ids: NDArray[np.str_]


@dataclass
class PreparedData:
    train_ids: NDArray[np.str_]
    class_names: list[str]
    tuning_a: PartitionPredictions
    tuning_b: PartitionPredictions
    calibration: PartitionPredictions
    test: PartitionPredictions


# Select original records without changing text, label or ID alignment.
def _select_records(data: TextDataset, indices: NDArray[np.int64]) -> TextDataset:
    return TextDataset(
        texts=data.texts[indices],
        labels=data.labels[indices],
        sample_ids=data.sample_ids[indices],
    )


# Reserve final calibration before dividing the tuning data into two halves.
def split_calibration_pool(
    calibration_pool: TextDataset,
    random_seed: int,
) -> tuple[TextDataset, TextDataset, TextDataset]:
    sample_indices = np.arange(calibration_pool.labels.size)
    tuning_indices, calibration_indices = train_test_split(
        sample_indices,
        train_size=0.5,
        stratify=calibration_pool.labels,
        random_state=random_seed,
    )

    tuning_labels = calibration_pool.labels[tuning_indices]
    tuning_a_indices, tuning_b_indices = train_test_split(
        tuning_indices,
        test_size=0.5,
        stratify=tuning_labels,
        random_state=random_seed,
    )

    tuning_a = _select_records(calibration_pool, tuning_a_indices)
    tuning_b = _select_records(calibration_pool, tuning_b_indices)
    calibration = _select_records(calibration_pool, calibration_indices)

    return tuning_a, tuning_b, calibration


# Predict without fitting preprocessing or the classifier on held-out requests.
def predict_partition(model: Pipeline, data: TextDataset) -> PartitionPredictions:
    probabilities = model.predict_proba(data.texts)

    return PartitionPredictions(
        probabilities=probabilities,
        labels=data.labels,
        sample_ids=data.sample_ids,
    )
