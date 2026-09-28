from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from examples.banking77_scores._artifacts import save_csv, save_manifest
from examples.banking77_scores._data import PartitionPredictions
from examples.banking77_scores._evaluation import prediction_set_metrics


ALPHAS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
CONFIDENCE_THRESHOLDS = (
    0.0,
    0.5,
    0.8,
    0.9,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
    0.999999,
)
SOCOP_REGULARIZATIONS = (0.001, 0.003, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
CONFIDENCE_LEVEL = 0.95


# Reuse coverage and routing metrics, adding the average set size left for review.
def evaluate_sets(
    prediction_sets: NDArray[np.bool_],
    test_data: PartitionPredictions,
) -> dict[str, float]:
    metrics = prediction_set_metrics(prediction_sets, test_data, CONFIDENCE_LEVEL)
    set_sizes = prediction_sets.sum(axis=1)
    deferred_sizes = set_sizes[set_sizes != 1]
    deferred_mean = float("nan")
    if deferred_sizes.size:
        deferred_mean = float(deferred_sizes.mean())
    metrics["deferred_set_size"] = deferred_mean

    return metrics


# Save small result tables and identify the exact score cache used by each method.
def save_evaluation(
    output_directory: Path,
    cache_path: Path,
    metadata: dict[str, Any],
    settings: Mapping[str, object],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        save_csv(output_directory / f"{name}.csv", rows)

    manifest = {
        "source_cache": str(cache_path),
        "source_sha256": sha256(cache_path.read_bytes()).hexdigest(),
        "model_id": metadata["model_id"],
        "model_revision": metadata["model_revision"],
        "seed": metadata["random_seed"],
        "sample_sizes": metadata["sample_sizes"],
        "scoring": metadata["scoring"],
        "normalization": "softmax of summed scores, temperature 1",
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "Wilson, pointwise on the evaluated test requests",
        "numpy_version": np.__version__,
        "settings": dict(settings),
    }
    save_manifest(output_directory, manifest)
    print(f"Results saved to {output_directory}")
