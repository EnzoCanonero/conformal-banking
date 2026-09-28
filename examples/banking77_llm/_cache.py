import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


# Reuse the completed prefix only when inputs and scoring settings are unchanged.
def load_scores(
    cache_path: Path,
    metadata: dict[str, Any],
) -> tuple[list[NDArray[np.float64]], NDArray[np.int64], float]:
    if not cache_path.exists():
        return [], np.empty(0, dtype=np.int64), 0.0

    with np.load(cache_path, allow_pickle=False) as archive:
        saved_metadata = json.loads(str(archive["metadata"].item()))
        if saved_metadata != metadata:
            raise ValueError("cache inputs or settings changed; use a new output path")

        scores = archive["log_scores"]
        token_counts = archive["token_counts"]
        scoring_seconds = float(archive["scoring_seconds"].item())

    class_count = len(metadata["class_names"])
    request_count = len(metadata["requests"])
    if scores.ndim != 2 or scores.shape[1] != class_count:
        raise ValueError("cached score columns must match the intent order")
    if scores.shape[0] > request_count or token_counts.shape != (class_count,):
        raise ValueError("cached scores or token counts do not match the inputs")

    return list(scores), token_counts, scoring_seconds


# Resume the target cache, or reuse matching requests from an earlier preparation.
def restore_scores(
    cache_path: Path,
    metadata: dict[str, Any],
    reuse_cache_path: Path | None,
) -> tuple[dict[str, Any], list[NDArray[np.float64]], NDArray[np.int64], float]:
    source_path = cache_path if cache_path.exists() else reuse_cache_path
    if source_path is None or not source_path.exists():
        return metadata, [], np.empty(0, dtype=np.int64), 0.0

    with np.load(source_path, allow_pickle=False) as archive:
        saved_metadata = json.loads(str(archive["metadata"].item()))

    # Only the sample plan may change when extending an existing preparation.
    plan_fields = ("requests", "sample_sizes")
    current_settings = {
        key: value for key, value in metadata.items() if key not in plan_fields
    }
    saved_settings = {
        key: value for key, value in saved_metadata.items() if key not in plan_fields
    }
    if current_settings != saved_settings:
        raise ValueError("cached model, prompt or scoring settings do not match")

    scores, token_counts, scoring_seconds = load_scores(source_path, saved_metadata)
    if not np.all(np.isfinite(scores)):
        raise ValueError("cached scores must be finite before reuse")

    requested_by_id = {row["sample_id"]: row for row in metadata["requests"]}
    completed_requests = saved_metadata["requests"][: len(scores)]
    completed_ids = {row["sample_id"] for row in completed_requests}
    if len(completed_ids) != len(completed_requests):
        raise ValueError("cached requests must have unique IDs")
    for request in completed_requests:
        if requested_by_id.get(request["sample_id"]) != request:
            raise ValueError("cached request text, label or partition does not match")

    # Keep completed rows first: checkpoint row i still belongs to request i.
    pending_requests = []
    for request in metadata["requests"]:
        if request["sample_id"] not in completed_ids:
            pending_requests.append(request)
    metadata = metadata.copy()
    metadata["requests"] = completed_requests + pending_requests

    if source_path == cache_path and metadata != saved_metadata:
        raise ValueError("target cache plan changed; use a new output path")

    return metadata, scores, token_counts, scoring_seconds


# Replace the archive only after the new checkpoint has been completely written.
def save_scores(
    cache_path: Path,
    metadata: dict[str, Any],
    score_rows: list[NDArray[np.float64]],
    token_counts: NDArray[np.int64],
    scoring_seconds: float,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    serialized_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    np.savez_compressed(
        temporary_path,
        metadata=np.asarray(serialized_metadata),
        log_scores=np.asarray(score_rows, dtype=np.float64),
        token_counts=token_counts,
        scoring_seconds=np.asarray(scoring_seconds),
    )
    temporary_path.replace(cache_path)
