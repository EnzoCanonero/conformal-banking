import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


@dataclass
class TextDataset:
    texts: NDArray[np.str_]
    labels: NDArray[np.int64]
    sample_ids: NDArray[np.str_]


@dataclass
class Banking77Data:
    train: TextDataset
    calibration: TextDataset
    test: TextDataset
    class_names: list[str]


# Read an official CSV using the shared label mapping and original row IDs.
def _read_banking77_csv(
    file_path: Path,
    label_to_index: dict[str, int],
) -> TextDataset:
    texts: list[str] = []
    labels: list[int] = []
    sample_ids: list[str] = []

    with file_path.open(encoding="utf-8", newline="") as data_file:
        rows = csv.DictReader(data_file)

        for row_index, row in enumerate(rows):
            text = row["text"]
            label = label_to_index[row["category"]]
            sample_id = f"{file_path.stem}:{row_index}"

            texts.append(text)
            labels.append(label)
            sample_ids.append(sample_id)

    return TextDataset(
        texts=np.asarray(texts, dtype=np.str_),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
    )


# Load local BANKING77 files and split only the official training data 75/25.
def load_banking77(
    data_directory: str | Path,
    random_seed: int = 42,
) -> Banking77Data:
    data_directory = Path(data_directory)
    categories_path = data_directory / "categories.json"

    with categories_path.open(encoding="utf-8") as categories_file:
        class_names: list[str] = json.load(categories_file)

    label_to_index = {}
    for class_index, class_name in enumerate(class_names):
        label_to_index[class_name] = class_index

    official_train = _read_banking77_csv(
        data_directory / "train.csv",
        label_to_index,
    )
    official_test = _read_banking77_csv(
        data_directory / "test.csv",
        label_to_index,
    )

    sample_indices = np.arange(official_train.labels.size)
    train_indices, calibration_indices = train_test_split(
        sample_indices,
        test_size=0.25,
        stratify=official_train.labels,
        random_state=random_seed,
    )

    train = TextDataset(
        texts=official_train.texts[train_indices],
        labels=official_train.labels[train_indices],
        sample_ids=official_train.sample_ids[train_indices],
    )
    calibration = TextDataset(
        texts=official_train.texts[calibration_indices],
        labels=official_train.labels[calibration_indices],
        sample_ids=official_train.sample_ids[calibration_indices],
    )

    return Banking77Data(
        train=train,
        calibration=calibration,
        test=official_test,
        class_names=class_names,
    )
