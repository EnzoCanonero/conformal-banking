from collections.abc import Mapping

import numpy as np

from conformal_selective_prediction.data import Banking77Data, TextDataset

from ..banking77_scores._data import PreparedData


# Reassemble the official records in a cache order independent of any random split.
def combine_records(data: Banking77Data) -> TextDataset:
    partitions = (data.train, data.calibration, data.test)
    texts = np.concatenate([partition.texts for partition in partitions])
    labels = np.concatenate([partition.labels for partition in partitions])
    sample_ids = np.concatenate([partition.sample_ids for partition in partitions])
    record_order = np.argsort(sample_ids)

    return TextDataset(
        texts=texts[record_order],
        labels=labels[record_order],
        sample_ids=sample_ids[record_order],
    )


# Check saved splits against the raw records before reusing their IDs and labels.
def verify_reference(
    reference: PreparedData,
    data: Banking77Data,
    records: TextDataset,
    expected_counts: Mapping[str, int],
) -> None:
    if reference.class_names != data.class_names:
        raise ValueError("TF-IDF class order does not match the dataset")
    if not np.array_equal(reference.test.sample_ids, data.test.sample_ids):
        raise ValueError("TF-IDF test IDs must match the unchanged official test set")

    partitions = {
        "tuning_a": reference.tuning_a,
        "tuning_b": reference.tuning_b,
        "calibration": reference.calibration,
        "test": reference.test,
    }
    sample_counts = {"train": reference.train_ids.size}
    partition_ids = [reference.train_ids]
    for name, partition in partitions.items():
        sample_counts[name] = partition.sample_ids.size
        partition_ids.append(partition.sample_ids)

    if sample_counts != expected_counts:
        raise ValueError("TF-IDF partition sizes do not match their manifest")

    # Every official record must occur exactly once across the five partitions.
    combined_ids = np.concatenate(partition_ids)
    sorted_ids = np.sort(combined_ids)
    if not np.array_equal(sorted_ids, records.sample_ids):
        raise ValueError("TF-IDF partitions must be disjoint and cover the dataset")

    for name, partition in partitions.items():
        record_indices = np.searchsorted(records.sample_ids, partition.sample_ids)
        expected_labels = records.labels[record_indices]
        if not np.array_equal(partition.labels, expected_labels):
            raise ValueError(f"TF-IDF {name} labels do not match the dataset")
