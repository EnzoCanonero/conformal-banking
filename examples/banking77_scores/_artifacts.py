import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._data import PartitionPredictions, PreparedData


# Save arrays without raw requests, fitted models or Python objects.
def save_prepared_seed(
    output_directory: Path,
    random_seed: int,
    preparation_id: str,
    data: PreparedData,
) -> None:
    arrays: dict[str, ArrayLike] = {
        "preparation_id": np.asarray(preparation_id),
        "train_ids": data.train_ids,
        "class_names": np.asarray(data.class_names, dtype=np.str_),
    }
    partitions = {
        "tuning_a": data.tuning_a,
        "tuning_b": data.tuning_b,
        "calibration": data.calibration,
        "test": data.test,
    }

    for name, predictions in partitions.items():
        arrays[f"{name}_probabilities"] = predictions.probabilities
        arrays[f"{name}_labels"] = predictions.labels
        arrays[f"{name}_ids"] = predictions.sample_ids

    output_path = output_directory / f"seed_{random_seed}.npz"
    np.savez_compressed(output_path, allow_pickle=False, **arrays)


# Load one seed only when it belongs to the requested preparation.
def load_prepared_seed(
    input_directory: Path,
    random_seed: int,
    preparation_id: str,
) -> PreparedData:
    input_path = input_directory / f"seed_{random_seed}.npz"

    with np.load(input_path, allow_pickle=False) as archive:
        saved_preparation_id = str(archive["preparation_id"].item())
        if saved_preparation_id != preparation_id:
            raise ValueError("seed archive and manifest belong to different preparations")

        partitions = {}
        for name in ("tuning_a", "tuning_b", "calibration", "test"):
            partitions[name] = PartitionPredictions(
                probabilities=archive[f"{name}_probabilities"],
                labels=archive[f"{name}_labels"],
                sample_ids=archive[f"{name}_ids"],
            )

        return PreparedData(
            train_ids=archive["train_ids"],
            class_names=archive["class_names"].tolist(),
            tuning_a=partitions["tuning_a"],
            tuning_b=partitions["tuning_b"],
            calibration=partitions["calibration"],
            test=partitions["test"],
        )


# Store the shared configuration and provenance as readable JSON.
def save_manifest(output_directory: Path, manifest: Mapping[str, object]) -> None:
    output_path = output_directory / "manifest.json"

    with output_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
        manifest_file.write("\n")


# Read the configuration that all score experiments will share.
def load_manifest(input_directory: Path) -> dict[str, Any]:
    input_path = input_directory / "manifest.json"

    with input_path.open(encoding="utf-8") as manifest_file:
        manifest: dict[str, Any] = json.load(manifest_file)

    return manifest


# Save one table with its column names and unrounded numerical results.
def save_csv(file_path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    field_names = list(rows[0])

    with file_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)
