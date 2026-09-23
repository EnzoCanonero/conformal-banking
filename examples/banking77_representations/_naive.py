from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..banking77_scores._artifacts import load_prepared_seed
from ..banking77_scores._evaluation import naive_metrics


# Apply fixed confidence cutoffs to cached encoder predictions without refitting.
def evaluate_naive(
    prepared_directory: Path,
    manifest: dict[str, Any],
    confidence_thresholds: Sequence[float],
) -> tuple[list[dict[str, str | float]], dict[str, str]]:
    rows: list[dict[str, str | float]] = []
    archive_hashes = {}
    for seed in manifest["random_seeds"]:
        data = load_prepared_seed(prepared_directory, seed, manifest["preparation_id"])
        measured_metrics = naive_metrics(
            data.test, confidence_thresholds, manifest["confidence_level"]
        )
        for metrics in measured_metrics:
            row: dict[str, str | float] = {
                "representation": "encoder",
                "method": "naive",
                "seed": seed,
            }
            row.update(metrics)
            rows.append(row)

        archive_path = prepared_directory / f"seed_{seed}.npz"
        archive_hashes[archive_path.name] = sha256(archive_path.read_bytes()).hexdigest()

    return rows, archive_hashes
