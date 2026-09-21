from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    max_sequence_length: int = 256
    batch_size: int = 32
    device: str = "cpu"
    normalize_embeddings: bool = True
    pooling: str = "mean"
    dimension: int = 384
    dtype: str = "float32"


ENCODER_CONFIG = EncoderConfig()
MODEL_DIRECTORY = Path("artifacts/text_encoder")


# Encode requests in their original order without fitting on them or their labels.
def encode_texts(
    texts: Sequence[str],
    model_directory: Path = MODEL_DIRECTORY,
) -> NDArray[np.float32]:
    # The optional dependency is needed only when computing new embeddings.
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    config = ENCODER_CONFIG
    model = SentenceTransformer(
        config.model_name,
        revision=config.revision,
        device=config.device,
        cache_folder=str(model_directory),
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )
    model.max_seq_length = config.max_sequence_length
    model.float()
    model.eval()
    model.requires_grad_(False)

    embeddings = model.encode(
        list(texts),
        batch_size=config.batch_size,
        normalize_embeddings=config.normalize_embeddings,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return np.asarray(embeddings, dtype=np.float32)
