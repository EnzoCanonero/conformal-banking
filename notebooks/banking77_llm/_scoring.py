from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import make_prompt_cache
from numpy.typing import NDArray


# Score candidate replies, reusing the request prefix to avoid repeated work.
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
        # The last prompt token predicts the first label token.
        input_ids = prompt_ids[-1:] + label_ids[:-1]
        model_input = mx.array([input_ids])
        candidate_cache = deepcopy(prompt_cache)
        logits = model(model_input, cache=candidate_cache)

        label_logits = logits[0]
        label_logits = label_logits.astype(mx.float32)
        log_probabilities = nn.log_softmax(label_logits, axis=-1)
        positions = mx.arange(len(label_ids))
        target_ids = mx.array(label_ids)
        token_log_probabilities = log_probabilities[positions, target_ids]

        label_score = mx.sum(token_log_probabilities).item()
        label_scores.append(label_score)

    return np.asarray(label_scores, dtype=np.float64)
