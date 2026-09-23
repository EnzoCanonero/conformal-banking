from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from ..banking77_scores._artifacts import load_manifest
from ..banking77_scores._comparison_data import (
    _check_metric_grid,
    _check_socop_choices,
    _load_metrics,
    summarize_metrics,
)
from ._reference import validate_reference


# Read only paired LAC and tuned SOCOP results, preserving each preparation's identity.
def load_results(
    encoder_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, str | float]]]:
    prepared_directory = encoder_directory / "prepared"
    prepared_manifest = load_manifest(prepared_directory)
    result_sources = {}
    comparison_rows = []

    for directory_name, method in (("lac", "lac"), ("socop", "socop_tuned")):
        # Reuse the record, protocol and source-hash checks from evaluation.
        reference = validate_reference(prepared_directory, directory_name)
        result_directory = encoder_directory / directory_name
        result_manifest = load_manifest(result_directory)
        for field, value in prepared_manifest.items():
            if result_manifest.get(field) != value:
                raise ValueError(f"encoder {directory_name} preparation differs: {field}")
        if result_manifest["tfidf_results"] != reference:
            raise ValueError("TF-IDF results changed since the encoder evaluation")

        tfidf_directory = Path(reference["directory"])
        tfidf_manifest = load_manifest(tfidf_directory)
        if result_manifest["evaluation_numpy_version"] != tfidf_manifest["evaluation_numpy_version"]:
            raise ValueError("representations used different evaluation NumPy versions")

        encoder_rows = _load_metrics(result_directory / "metrics.csv")
        _check_metric_grid(
            encoder_rows, (method,), "alpha", prepared_manifest["alphas"], prepared_manifest
        )
        if directory_name == "socop":
            _check_socop_choices(result_directory, encoder_rows, prepared_manifest)

        tfidf_rows = _load_metrics(tfidf_directory / "metrics.csv")
        tfidf_rows = [row for row in tfidf_rows if row["method"] == method]
        for representation, rows in (("tfidf", tfidf_rows), ("encoder", encoder_rows)):
            for row in rows:
                row["representation"] = representation
                sample_count = float(row["sample_count"])
                automated_count = float(row["automated_count"])
                deferred_count = sample_count - automated_count
                set_size_total = float(row["average_set_size"]) * sample_count
                deferred_size_total = set_size_total - automated_count
                deferred_set_size = float("nan")
                if deferred_count > 0:
                    deferred_set_size = deferred_size_total / deferred_count
                row["deferred_set_size"] = deferred_set_size
                comparison_rows.append(row)

        # Keep source files in place; the comparison records exactly what it reads.
        filenames = ["manifest.json", "metrics.csv"]
        if directory_name == "socop":
            filenames.append("tuning_choices.csv")
        hashes = {}
        for filename in filenames:
            contents = (result_directory / filename).read_bytes()
            hashes[filename] = sha256(contents).hexdigest()
        result_sources[directory_name] = {
            "tfidf": reference,
            "encoder": {"directory": str(result_directory), "sha256": hashes},
        }

    manifest = {
        "encoder_preparation": prepared_manifest,
        "result_sources": result_sources,
    }
    return manifest, comparison_rows


# Extend the existing per-split summary with accuracy and deferred-set size.
def summarize_results(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    summary_rows = []
    for representation in ("tfidf", "encoder"):
        representation_rows = [row for row in rows if row["representation"] == representation]
        summaries = summarize_metrics(representation_rows, "alpha")
        for summary in summaries:
            summary["representation"] = representation
            matching_rows = [
                row for row in representation_rows
                if row["method"] == summary["method"] and row["alpha"] == summary["alpha"]
            ]
            for metric in ("accuracy", "deferred_set_size"):
                values = [float(row[metric]) for row in matching_rows]
                summary[metric] = float(np.mean(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
            summary_rows.append(summary)
    return summary_rows
