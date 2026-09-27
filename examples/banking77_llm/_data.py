from pathlib import Path
from typing import TypedDict

import numpy as np
from sklearn.model_selection import train_test_split

from conformal_selective_prediction.data import load_banking77
from examples.banking77_scores._data import split_calibration_pool


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
