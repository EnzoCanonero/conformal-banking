import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ._cache import restore_scores, save_scores
from ._data import prepare_requests


DATA_DIRECTORY = Path("data/raw/banking77")
STUDY_DIRECTORY = Path("outputs/banking77/llm/full")
CACHE_PATH = STUDY_DIRECTORY / "prepared/scores.npz"
REUSE_CACHE_PATH = Path("outputs/banking77/llm/prepared/scores.npz")
MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MODEL_REVISION = "50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b"
RANDOM_SEED = 42
SAMPLE_SIZES = {"tuning_a": 625, "tuning_b": 625, "calibration": 1251, "test": 3080}
ESTIMATED_SECONDS_PER_REQUEST = 8.42


# Prepare fixed inputs, then resume scoring without fitting or evaluating any policy.
def prepare_study(
    data_directory: Path = DATA_DIRECTORY,
    cache_path: Path = CACHE_PATH,
    *,
    reuse_cache_path: Path | None = REUSE_CACHE_PATH,
    dry_run: bool = False,
) -> None:
    requests, class_names = prepare_requests(data_directory, RANDOM_SEED, SAMPLE_SIZES)
    instructions = (
        "Classify the banking customer request.\n"
        "Choose one of these intents and return only its name:\n"
        + "\n".join(class_names)
    )
    metadata: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "instructions": instructions,
        "scoring": "sum of token log-probabilities, including EOS",
        "add_generation_prompt": True,
        "random_seed": RANDOM_SEED,
        "sample_sizes": SAMPLE_SIZES,
        "class_names": class_names,
        "requests": requests,
    }
    metadata, score_rows, token_counts, scoring_seconds = restore_scores(
        cache_path, metadata, reuse_cache_path
    )
    requests = metadata["requests"]
    completed_count = len(score_rows)
    pending_requests = requests[completed_count:]

    seconds_per_request = ESTIMATED_SECONDS_PER_REQUEST
    if completed_count:
        seconds_per_request = scoring_seconds / completed_count
    estimated_hours = len(pending_requests) * seconds_per_request / 3600

    print(f"Requests: {SAMPLE_SIZES}")
    print(f"Cached or reusable: {completed_count}/{len(requests)}")
    print(f"New requests to score: {len(pending_requests)}")
    print(f"Estimated remaining scoring time: {estimated_hours:.1f} hours")
    print(f"Output cache: {cache_path}")
    if dry_run:
        return

    # Preserve inherited scores before model loading; the source stays untouched.
    if score_rows and not cache_path.exists():
        save_scores(cache_path, metadata, score_rows, token_counts, scoring_seconds)
    if not pending_requests:
        return

    # The optional MLX dependencies are not imported for previews or complete caches.
    from ._scoring import score_requests

    pending_texts = [request["text"] for request in pending_requests]
    scored_requests = score_requests(
        pending_texts, class_names, MODEL_ID, MODEL_REVISION, instructions
    )
    for scores, token_counts, elapsed_seconds in scored_requests:
        if scores.shape != (len(class_names),) or not np.all(np.isfinite(scores)):
            raise ValueError("expected one finite score per intent")

        score_rows.append(scores)
        scoring_seconds += elapsed_seconds
        save_scores(cache_path, metadata, score_rows, token_counts, scoring_seconds)

    print(f"Saved {len(score_rows)} requests to {cache_path}")
    print(f"Scoring time: {scoring_seconds / 3600:.2f} hours")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache BANKING77 LLM label scores.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview inputs and runtime without MLX."
    )
    arguments = parser.parse_args()
    prepare_study(dry_run=arguments.dry_run)


if __name__ == "__main__":
    main()
