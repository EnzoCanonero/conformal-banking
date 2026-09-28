from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from examples.banking77_scores._artifacts import load_manifest, load_prepared_seed
from examples.banking77_scores._data import PartitionPredictions


# Align a classifier's saved predictions with the LLM's exact records and columns.
def load_classifier(
    prepared_directory: Path,
    llm_partitions: dict[str, PartitionPredictions],
    llm_metadata: dict[str, Any],
) -> tuple[dict[str, PartitionPredictions], dict[str, Any]]:
    manifest = load_manifest(prepared_directory)
    seed = llm_metadata["random_seed"]
    data = load_prepared_seed(prepared_directory, seed, manifest["preparation_id"])
    if data.class_names != llm_metadata["class_names"]:
        raise ValueError("classifier and LLM intent columns must have the same order")

    partitions = {}
    for name, llm_data in llm_partitions.items():
        classifier_data: PartitionPredictions = getattr(data, name)
        classifier_ids = classifier_data.sample_ids.tolist()
        llm_ids = llm_data.sample_ids.tolist()
        if len(set(classifier_ids)) != len(classifier_ids):
            raise ValueError("classifier request IDs must be unique")
        if set(classifier_ids) != set(llm_ids):
            raise ValueError(f"classifier and LLM {name} IDs must match exactly")
        if np.intersect1d(data.train_ids, llm_data.sample_ids).size:
            raise ValueError("comparison requests must be outside classifier training")

        # Reused LLM scores come first in its cache, so row order can differ.
        positions = {sample_id: index for index, sample_id in enumerate(classifier_ids)}
        ordered_indices = [positions[sample_id] for sample_id in llm_ids]
        labels = classifier_data.labels[ordered_indices]
        if not np.array_equal(labels, llm_data.labels):
            raise ValueError(f"classifier and LLM {name} labels must match")
        partitions[name] = PartitionPredictions(
            probabilities=classifier_data.probabilities[ordered_indices],
            labels=labels,
            sample_ids=classifier_data.sample_ids[ordered_indices],
        )

    archive_path = prepared_directory / f"seed_{seed}.npz"
    source = {
        "archive": str(archive_path),
        "sha256": sha256(archive_path.read_bytes()).hexdigest(),
        "preparation_id": manifest["preparation_id"],
        "manifest_sha256": sha256(
            (prepared_directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "model_settings": manifest["model_settings"],
        "training_count": int(data.train_ids.size),
    }
    return partitions, source
