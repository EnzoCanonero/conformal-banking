import json
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from sklearn.model_selection import train_test_split

from llm_scorekit.data import load_banking77
from examples.banking77.scores._data import PartitionPredictions, split_calibration_pool


class Request(TypedDict):
    sample_id: str
    partition: str
    text: str
    label: int


# Sample within the existing baseline partitions, keeping the official test separate.
def prepare_requests(
    data_directory: Path,
    random_seed: int,
    sample_sizes: dict[str, int],
) -> tuple[list[Request], list[str]]:
    data = load_banking77(data_directory, random_seed=random_seed)
    tuning_a, tuning_b, calibration = split_calibration_pool(
        data.calibration, random_seed
    )
    partitions = {
        "tuning_a": tuning_a,
        "tuning_b": tuning_b,
        "calibration": calibration,
        "test": data.test,
    }
    requests: list[Request] = []

    for name, partition in partitions.items():
        indices = np.arange(partition.labels.size)
        if sample_sizes[name] == partition.labels.size:
            selected_indices = indices
        else:
            selected_indices, _ = train_test_split(
                indices,
                train_size=sample_sizes[name],
                stratify=partition.labels,
                random_state=random_seed,
            )

        for index in selected_indices:
            request = Request(
                sample_id=str(partition.sample_ids[index]),
                partition=name,
                text=str(partition.texts[index]),
                label=int(partition.labels[index]),
            )
            requests.append(request)

    return requests, data.class_names


# Normalize completed label scores and restore their saved partition and intent order.
def load_predictions(
    cache_path: Path,
) -> tuple[dict[str, PartitionPredictions], dict[str, Any]]:
    with np.load(cache_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        log_scores = archive["log_scores"]

    requests = metadata["requests"]
    class_count = len(metadata["class_names"])
    if log_scores.shape != (len(requests), class_count):
        raise ValueError("evaluation needs a complete cache with one score per intent")
    if not np.all(np.isfinite(log_scores)):
        raise ValueError("cached log scores must be finite")

    labels = np.asarray([row["label"] for row in requests], dtype=np.int64)
    sample_ids = np.asarray([row["sample_id"] for row in requests], dtype=np.str_)
    partition_names = np.asarray([row["partition"] for row in requests], dtype=np.str_)
    if np.unique(sample_ids).size != len(requests):
        raise ValueError("requests must have unique IDs across partitions")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("cached labels must match the intent columns")

    # Keep the notebook's summed scoring rule; no temperature or length adjustment.
    row_maxima = log_scores.max(axis=1, keepdims=True)
    shifted_scores = log_scores - row_maxima
    unnormalized_weights = np.exp(shifted_scores)
    row_totals = unnormalized_weights.sum(axis=1, keepdims=True)
    weights = unnormalized_weights / row_totals
    partitions = {}

    for name in ("tuning_a", "tuning_b", "calibration", "test"):
        selected = partition_names == name
        if selected.sum() != metadata["sample_sizes"][name] or not np.any(selected):
            raise ValueError(f"cached {name} requests do not match the recorded size")
        partitions[name] = PartitionPredictions(
            probabilities=weights[selected],
            labels=labels[selected],
            sample_ids=sample_ids[selected],
        )

    return partitions, metadata
