import json
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from conformal_selective_prediction.models.encoder import ENCODER_CONFIG, encode_texts


# Identify the ordered input records without storing their raw text in the cache.
def _input_hash(texts: Sequence[str], sample_ids: Sequence[str]) -> str:
    records = list(zip(sample_ids, texts))
    serialized_records = json.dumps(records, ensure_ascii=False)
    encoded_records = serialized_records.encode("utf-8")
    return sha256(encoded_records).hexdigest()


# Reuse matching embeddings, or compute them once without any labels or splits.
def load_or_encode_embeddings(
    texts: Sequence[str],
    sample_ids: Sequence[str],
    cache_path: Path,
) -> NDArray[np.float32]:
    if len(texts) == 0 or len(texts) != len(sample_ids):
        raise ValueError("provide non-empty texts with one sample ID per text")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique")

    input_hash = _input_hash(texts, sample_ids)
    encoder_settings = asdict(ENCODER_CONFIG)

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["input_sha256"] != input_hash:
                raise ValueError("embedding cache does not match the ordered input records")
            if metadata["encoder"] != encoder_settings:
                raise ValueError("embedding cache uses a different encoder configuration")

            embeddings: NDArray[np.float32] = archive["embeddings"]
            return embeddings

    start_time = perf_counter()
    embeddings = encode_texts(texts)
    encoding_seconds = perf_counter() - start_time
    packages = ("numpy", "sentence-transformers", "transformers", "torch", "tokenizers")
    package_versions = {}
    for package in packages:
        package_versions[package] = version(package)

    metadata = {
        "input_sha256": input_hash,
        "encoder": encoder_settings,
        "package_versions": package_versions,
        "encoding_seconds": encoding_seconds,
    }
    serialized_metadata = json.dumps(metadata, sort_keys=True)

    # Keep vectors, row IDs and provenance together in one non-pickle archive.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        metadata=np.asarray(serialized_metadata),
    )

    return embeddings
