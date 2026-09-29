from collections.abc import Iterator, Sequence
from copy import deepcopy
from time import perf_counter

import mlx.core as mx  # type: ignore[import-not-found]
import mlx.nn as nn  # type: ignore[import-not-found]
import numpy as np
from mlx_lm import load  # type: ignore[import-not-found]
from mlx_lm.models.cache import make_prompt_cache  # type: ignore[import-not-found]
from numpy.typing import NDArray


# Score fixed candidate replies, reusing the prompt prefix for each intent.
def score_candidates(
    model: nn.Module,
    prompt_ids: list[int],
    candidate_ids: list[list[int]],
) -> NDArray[np.float64]:
    prompt_cache = make_prompt_cache(model)
    model(mx.array([prompt_ids[:-1]]), cache=prompt_cache)
    mx.eval([layer.state for layer in prompt_cache])
    label_scores = []

    for label_ids in candidate_ids:
        # The last prompt token predicts the first candidate token.
        input_ids = prompt_ids[-1:] + label_ids[:-1]
        model_input = mx.array([input_ids])
        candidate_cache = deepcopy(prompt_cache)
        logits = model(model_input, cache=candidate_cache)

        label_logits = logits[0].astype(mx.float32)
        log_probabilities = nn.log_softmax(label_logits, axis=-1)
        positions = mx.arange(len(label_ids))
        target_ids = mx.array(label_ids)
        token_log_probabilities = log_probabilities[positions, target_ids]

        label_score = mx.sum(token_log_probabilities).item()
        label_scores.append(label_score)

    return np.asarray(label_scores, dtype=np.float64)


# Load Qwen once and yield one completed request at a time for checkpointing.
def score_requests(
    texts: Sequence[str],
    class_names: Sequence[str],
    model_id: str,
    revision: str,
    instructions: str,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.int64], float]]:
    mx.set_cache_limit(512 * 1024**2)
    loaded_model = load(model_id, revision=revision)
    model = loaded_model[0]
    tokenizer = loaded_model[1]
    candidate_ids = []

    for name in class_names:
        label_ids = tokenizer.encode(name, add_special_tokens=False)
        candidate_ids.append(label_ids + [tokenizer.eos_token_id])

    token_counts = np.asarray([len(ids) for ids in candidate_ids], dtype=np.int64)

    for text in texts:
        start_time = perf_counter()
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        scores = score_candidates(model, prompt_ids, candidate_ids)
        scoring_seconds = perf_counter() - start_time

        yield scores, token_counts, scoring_seconds
