import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from conformal_selective_prediction.data import TextDataset, load_banking77

from ..banking77_scores._artifacts import (
    load_manifest,
    load_prepared_seed,
    save_manifest,
    save_prepared_seed,
)
from ..banking77_scores._data import PartitionPredictions, PreparedData
from ..banking77_scores.prepare import DATA_DIRECTORY, RANDOM_SEEDS, _dataset_hashes
from ._data import combine_records, verify_reference
from ._embeddings import load_or_encode_embeddings


TFIDF_DIRECTORY = Path("outputs/banking77/score_comparison/prepared")
OUTPUT_DIRECTORY = Path("outputs/banking77/representation_comparison/encoder")


# Fit only on the saved training IDs, then predict the four held-out partitions.
def prepare_seed(
    reference: PreparedData,
    records: TextDataset,
    embeddings: NDArray[np.float32],
    classifier_settings: dict[str, Any],
) -> PreparedData:
    training_indices = np.searchsorted(records.sample_ids, reference.train_ids)
    training_embeddings = embeddings[training_indices]
    training_labels = records.labels[training_indices]
    classifier = LogisticRegression(**classifier_settings)
    classifier.fit(training_embeddings, training_labels)

    expected_classes = np.arange(len(reference.class_names))
    if not np.array_equal(classifier.classes_, expected_classes):
        raise ValueError("classifier classes must match the dataset label indices")

    partitions = {
        "tuning_a": reference.tuning_a,
        "tuning_b": reference.tuning_b,
        "calibration": reference.calibration,
        "test": reference.test,
    }
    predictions = {}
    for name, partition in partitions.items():
        record_indices = np.searchsorted(records.sample_ids, partition.sample_ids)
        partition_embeddings = embeddings[record_indices]
        probabilities = classifier.predict_proba(partition_embeddings)
        predictions[name] = PartitionPredictions(
            probabilities=np.asarray(probabilities, dtype=np.float64),
            labels=partition.labels,
            sample_ids=partition.sample_ids,
        )

    return PreparedData(
        train_ids=reference.train_ids,
        class_names=reference.class_names,
        tuning_a=predictions["tuning_a"],
        tuning_b=predictions["tuning_b"],
        calibration=predictions["calibration"],
        test=predictions["test"],
    )


# Verify the TF-IDF inputs before encoding or training anything new.
def load_reference(
    data_directory: Path,
    tfidf_directory: Path,
) -> tuple[dict[str, Any], TextDataset, dict[int, PreparedData]]:
    manifest = load_manifest(tfidf_directory)
    if manifest["dataset_sha256"] != _dataset_hashes(data_directory):
        raise ValueError("local data files do not match the TF-IDF preparation")
    if manifest["random_seeds"] != list(RANDOM_SEEDS):
        raise ValueError("the comparison requires the same five TF-IDF seeds")
    if manifest["package_versions"]["scikit_learn"] != version("scikit-learn"):
        raise ValueError("use the same scikit-learn version as the TF-IDF preparation")

    data = load_banking77(data_directory)
    records = combine_records(data)
    if manifest["class_names"] != data.class_names:
        raise ValueError("TF-IDF manifest class order does not match the dataset")

    references = {}
    for random_seed in manifest["random_seeds"]:
        reference = load_prepared_seed(
            tfidf_directory, random_seed, manifest["preparation_id"]
        )
        expected_counts = manifest["sample_counts"][str(random_seed)]
        verify_reference(reference, data, records, expected_counts)
        references[random_seed] = reference

    return manifest, records, references


# Prepare paired probabilities only; do not select scores or evaluate test results.
def prepare_study(
    data_directory: Path,
    tfidf_directory: Path,
    output_directory: Path,
) -> None:
    reference_manifest, records, references = load_reference(
        data_directory, tfidf_directory
    )
    source_manifest_path = tfidf_directory / "manifest.json"
    source_manifest_hash = sha256(source_manifest_path.read_bytes()).hexdigest()
    cache_path = output_directory / "embeddings.npz"

    print("Preparing shared encoder embeddings")
    embeddings = load_or_encode_embeddings(
        records.texts.tolist(), records.sample_ids.tolist(), cache_path
    )
    with np.load(cache_path, allow_pickle=False) as archive:
        embedding_metadata = json.loads(str(archive["metadata"].item()))

    prepared_directory = output_directory / "prepared"
    prepared_directory.mkdir(parents=True, exist_ok=True)
    preparation_id = uuid4().hex
    classifier_settings = reference_manifest["model_settings"]["logistic_regression"]

    for random_seed, reference in references.items():
        prepared_data = prepare_seed(reference, records, embeddings, classifier_settings)
        save_prepared_seed(
            prepared_directory, random_seed, preparation_id, prepared_data
        )
        print(f"Prepared encoder seed {random_seed}")

    # Retain the shared protocol, but not the old naive or fixed-SOCOP settings.
    manifest = reference_manifest.copy()
    manifest.pop("confidence_thresholds")
    manifest.pop("socop_reference_regularization")
    manifest.update(
        {
            "preparation_id": preparation_id,
            "data_directory": str(data_directory),
            "representation": "frozen_encoder",
            "model_settings": {
                "encoder": embedding_metadata["encoder"],
                "logistic_regression": classifier_settings,
            },
            "embedding_cache": {
                "path": str(cache_path),
                "sha256": sha256(cache_path.read_bytes()).hexdigest(),
                "metadata": embedding_metadata,
            },
            "tfidf_reference": {
                "prepared_directory": str(tfidf_directory),
                "preparation_id": reference_manifest["preparation_id"],
                "manifest_sha256": source_manifest_hash,
            },
            "package_versions": {
                "numpy": version("numpy"),
                "scikit_learn": version("scikit-learn"),
            },
        }
    )

    # Write the manifest last so incomplete runs cannot appear as a new preparation.
    save_manifest(prepared_directory, manifest)
    print(f"Encoder predictions saved to {prepared_directory}")


def main() -> None:
    prepare_study(DATA_DIRECTORY, TFIDF_DIRECTORY, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()
