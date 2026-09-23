from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..banking77_scores._artifacts import load_manifest, save_manifest
from ..banking77_scores._comparison_data import (
    _check_metric_grid,
    _check_prepared_ids,
    _check_socop_choices,
    _load_metrics,
)


# Compare actual record IDs and labels, not the two models' preparation UUIDs.
def _check_paired_records(
    encoder_directory: Path,
    tfidf_directory: Path,
    seeds: list[int],
) -> None:
    paired_arrays = (
        "train_ids",
        "tuning_a_ids", "tuning_a_labels",
        "tuning_b_ids", "tuning_b_labels",
        "calibration_ids", "calibration_labels",
        "test_ids", "test_labels",
    )
    for seed in seeds:
        with (
            np.load(encoder_directory / f"seed_{seed}.npz", allow_pickle=False) as encoder,
            np.load(tfidf_directory / f"seed_{seed}.npz", allow_pickle=False) as tfidf,
        ):
            for name in paired_arrays:
                if not np.array_equal(encoder[name], tfidf[name]):
                    raise ValueError(f"encoder and TF-IDF {name} differ for seed {seed}")


# Validate and identify existing TF-IDF results without recomputing any policy.
def validate_reference(prepared_directory: Path, method: str) -> dict[str, Any]:
    encoder_manifest = load_manifest(prepared_directory)
    source = encoder_manifest["tfidf_reference"]
    tfidf_directory = Path(source["prepared_directory"])
    source_manifest_path = tfidf_directory / "manifest.json"
    source_hash = sha256(source_manifest_path.read_bytes()).hexdigest()
    tfidf_manifest = load_manifest(tfidf_directory)
    if source_hash != source["manifest_sha256"]:
        raise ValueError("TF-IDF preparation changed; prepare the paired study again")
    if tfidf_manifest["preparation_id"] != source["preparation_id"]:
        raise ValueError("TF-IDF preparation identity does not match the recorded source")

    shared_fields = (
        "dataset_sha256", "class_names", "random_seeds", "sample_counts",
        "split_protocol", "alphas", "confidence_level", "interval_method",
        "socop_regularizations", "socop_tuning",
    )
    for field in shared_fields:
        if encoder_manifest[field] != tfidf_manifest[field]:
            raise ValueError(f"representation protocols differ: {field}")
    encoder_classifier = encoder_manifest["model_settings"]["logistic_regression"]
    tfidf_classifier = tfidf_manifest["model_settings"]["logistic_regression"]
    if encoder_classifier != tfidf_classifier:
        raise ValueError("representations must use the same classifier settings")

    _check_prepared_ids(prepared_directory, encoder_manifest)
    _check_prepared_ids(tfidf_directory, tfidf_manifest)
    _check_paired_records(
        prepared_directory, tfidf_directory, encoder_manifest["random_seeds"]
    )

    result_directory = tfidf_directory.parent / method
    result_manifest = load_manifest(result_directory)
    for field, value in tfidf_manifest.items():
        if result_manifest.get(field) != value:
            raise ValueError(f"TF-IDF {method} results use a different preparation: {field}")
    if result_manifest["evaluation_numpy_version"] != version("numpy"):
        raise ValueError("use the same NumPy version as the TF-IDF evaluation")

    selected_method = method
    filenames = ["manifest.json", "metrics.csv", "class_coverage.csv", "set_sizes.csv"]
    if method == "socop":
        selected_method = "socop_tuned"
        filenames.extend(["tuning_candidates.csv", "tuning_choices.csv"])

    rows = _load_metrics(result_directory / "metrics.csv")
    selected_rows = [row for row in rows if row["method"] == selected_method]
    _check_metric_grid(
        selected_rows, (selected_method,), "alpha", encoder_manifest["alphas"],
        encoder_manifest,
    )
    if method == "socop":
        _check_socop_choices(result_directory, selected_rows, tfidf_manifest)

    # Record exactly which existing result files the later comparison may reuse.
    hashes = {}
    for filename in filenames:
        file_contents = (result_directory / filename).read_bytes()
        hashes[filename] = sha256(file_contents).hexdigest()

    return {
        "directory": str(result_directory),
        "method": selected_method,
        "sha256": hashes,
    }


# Attach the read-only reference to the newly computed encoder results.
def save_reference(output_directory: Path, reference: dict[str, Any]) -> None:
    manifest = load_manifest(output_directory)
    manifest["tfidf_results"] = reference
    save_manifest(output_directory, manifest)
