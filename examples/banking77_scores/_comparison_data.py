import csv
from pathlib import Path
from typing import Any

import numpy as np

from ._artifacts import load_manifest


# Read saved metric tables without recomputing scores or evaluating policies.
def _load_metrics(file_path: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []

    with file_path.open(encoding="utf-8", newline="") as input_file:
        for saved_row in csv.DictReader(input_file):
            row: dict[str, str | float] = {"method": saved_row["method"]}
            for name, value in saved_row.items():
                if name != "method":
                    row[name] = float(value)
            rows.append(row)

    return rows


# Check the shared record identities and class order, without reading probabilities.
def _check_prepared_ids(prepared_directory: Path, manifest: dict[str, Any]) -> None:
    reference_test_ids = None
    partitions = ("train", "tuning_a", "tuning_b", "calibration", "test")

    for seed in manifest["random_seeds"]:
        archive_path = prepared_directory / f"seed_{seed}.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            preparation_id = str(archive["preparation_id"].item())
            class_names = archive["class_names"].tolist()
            if preparation_id != manifest["preparation_id"]:
                raise ValueError("prepared archive and manifest identities differ")
            if class_names != manifest["class_names"]:
                raise ValueError("prepared class order differs from the manifest")

            seen_ids: set[str] = set()
            for partition in partitions:
                sample_ids = archive[f"{partition}_ids"].tolist()
                unique_ids = set(sample_ids)
                expected_count = manifest["sample_counts"][str(seed)][partition]
                if len(sample_ids) != expected_count or len(unique_ids) != expected_count:
                    raise ValueError("prepared record counts differ from the manifest")
                if seen_ids.intersection(unique_ids):
                    raise ValueError("prepared partitions share record IDs")
                seen_ids.update(unique_ids)

            test_ids = archive["test_ids"].tolist()
            if reference_test_ids is not None and test_ids != reference_test_ids:
                raise ValueError("test IDs or their order differ between seeds")
            reference_test_ids = test_ids


# Require each saved method to contain the full seed and parameter grid exactly once.
def _check_metric_grid(
    rows: list[dict[str, str | float]],
    methods: tuple[str, ...],
    parameter: str,
    values: list[float],
    manifest: dict[str, Any],
) -> None:
    expected_keys = {
        (method, seed, value)
        for method in methods
        for seed in manifest["random_seeds"]
        for value in values
    }
    actual_keys = {(row["method"], row["seed"], row[parameter]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError("saved results do not match the method, seed and parameter grid")

    for row in rows:
        seed = int(row["seed"])
        counts = manifest["sample_counts"][str(seed)]
        if row["sample_count"] != counts["test"]:
            raise ValueError("saved test counts differ from the preparation")
        if parameter == "alpha" and row["calibration_count"] != counts["calibration"]:
            raise ValueError("saved calibration counts differ from the preparation")


# Keep the fixed reference and the recorded per-target tuning choices distinct.
def _check_socop_choices(
    result_directory: Path,
    rows: list[dict[str, str | float]],
    manifest: dict[str, Any],
) -> None:
    with (result_directory / "tuning_choices.csv").open(encoding="utf-8") as input_file:
        choice_rows = list(csv.DictReader(input_file))
    choices = {
        (int(row["seed"]), float(row["alpha"])): float(row["regularization"])
        for row in choice_rows
    }
    expected_keys = {
        (seed, alpha) for seed in manifest["random_seeds"] for alpha in manifest["alphas"]
    }
    if set(choices) != expected_keys or len(choice_rows) != len(expected_keys):
        raise ValueError("SOCOP choices do not match the seed and alpha grid")

    for row in rows:
        regularization = row["regularization"]
        if row["method"] == "socop_fixed":
            expected_regularization = manifest["socop_reference_regularization"]
        else:
            key = (int(row["seed"]), float(row["alpha"]))
            expected_regularization = choices[key]
            if expected_regularization not in manifest["socop_regularizations"]:
                raise ValueError("selected SOCOP regularization is outside the saved grid")
        if regularization != expected_regularization:
            raise ValueError("SOCOP results differ from the fixed or selected regularization")


# Load comparable score results and retain only one copy of the shared naive baseline.
def load_results(
    study_directory: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, str | float]],
    list[dict[str, str | float]],
]:
    prepared_directory = study_directory / "prepared"
    manifest = load_manifest(prepared_directory)
    _check_prepared_ids(prepared_directory, manifest)
    method_groups = {
        "lac": ("lac",),
        "aps": ("aps",),
        "socop": ("socop_fixed", "socop_tuned"),
    }
    reference_naive = (study_directory / "lac" / "naive_metrics.csv").read_bytes()
    reference_version = load_manifest(study_directory / "lac")["evaluation_numpy_version"]
    conformal_rows = []

    for directory_name, methods in method_groups.items():
        result_directory = study_directory / directory_name
        result_manifest = load_manifest(result_directory)
        for key, value in manifest.items():
            if result_manifest.get(key) != value:
                raise ValueError(f"{directory_name} preparation metadata differs: {key}")
        if result_manifest["evaluation_numpy_version"] != reference_version:
            raise ValueError("score experiments used different NumPy versions")

        rows = _load_metrics(result_directory / "metrics.csv")
        _check_metric_grid(rows, methods, "alpha", manifest["alphas"], manifest)
        if directory_name == "socop":
            _check_socop_choices(result_directory, rows, manifest)
        conformal_rows.extend(rows)

        naive_content = (result_directory / "naive_metrics.csv").read_bytes()
        if naive_content != reference_naive:
            raise ValueError("independently saved naive results differ between methods")

    naive_rows = _load_metrics(study_directory / "lac" / "naive_metrics.csv")
    _check_metric_grid(
        naive_rows, ("naive",), "confidence_threshold", manifest["confidence_thresholds"], manifest
    )
    return manifest, conformal_rows, naive_rows


# Average per-split metrics; ranges describe split variation, not confidence intervals.
def summarize_metrics(
    rows: list[dict[str, str | float]],
    parameter: str,
) -> list[dict[str, str | float]]:
    metrics = ["automation_rate", "automated_error_rate", "abstention_rate"]
    if parameter == "alpha":
        metrics = ["coverage", "average_set_size", "empty_set_rate"] + metrics

    groups: dict[tuple[str, float], list[dict[str, str | float]]] = {}
    for row in rows:
        key = (str(row["method"]), float(row[parameter]))
        groups.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str | float]] = []
    for (method, setting), group in groups.items():
        summary: dict[str, str | float] = {
            "method": method,
            parameter: setting,
            "run_count": len(group),
        }
        if parameter == "alpha":
            summary["target_coverage"] = 1.0 - setting

        for metric in metrics:
            values = [float(row[metric]) for row in group]
            # Undefined automated errors remain NaN, rather than becoming zero.
            summary[metric] = float(np.mean(values))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(summary)

    return summary_rows
