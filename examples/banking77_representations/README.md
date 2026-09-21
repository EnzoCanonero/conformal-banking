# Frozen encoder preparation

This is the first part of the TF-IDF versus frozen-encoder study. It computes
and caches text vectors only: classifier training and LAC/SOCOP evaluation
will follow separately.

- **Encoder:** [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md),
  fixed to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. This compact
  English sentence encoder provides a practical first comparison without
  searching across models. Its license is Apache-2.0.
- **Representation:** 384 values per request, using the model's mean pooling
  and unit-length normalization. Inputs are truncated at 256 tokenizer tokens;
  no task-specific prompt or additional text cleaning is applied.
- **Inference:** CPU, float32 and batches of 32. Weights remain frozen, and
  evaluation mode disables training-time dropout. Encoding uses neither labels
  nor statistics fitted across the input requests, so the same vectors can be
  reused across training/calibration splits.
- **Provenance:** BANKING77 is not listed among the model card's training
  datasets. This does not establish that its requests never appeared in
  upstream training; we do not claim a contamination-free benchmark.

Install the optional dependency from the repository root:

```bash
python -m pip install -e ".[example,encoder]"
```

A small cache example, without loading BANKING77 or training a classifier:

```python
from pathlib import Path

from examples.banking77_representations._embeddings import load_or_encode_embeddings

texts = [
    "My card has not arrived.",
    "How do I change my PIN?",
    "I need to cancel a bank transfer.",
]
sample_ids = ["example:0", "example:1", "example:2"]
cache_path = Path("outputs/banking77/representation_comparison/smoke/embeddings.npz")
embeddings = load_or_encode_embeddings(texts, sample_ids, cache_path)
print(embeddings.shape)
```

The first call downloads the pinned model into `artifacts/text_encoder/` and
saves the vectors. Subsequent calls with the same ordered records and encoder
settings reuse the cache without loading the model. Downloads and outputs are
ignored by Git. Inference is local; the requests are not sent to a hosted model.

Each NPZ contains the vectors, original row IDs and JSON metadata recording the
encoder settings, library versions and SHA-256 of the ordered IDs and texts.
Raw texts and labels are not saved. A mismatched input or configuration raises
an error rather than overwriting an existing cache; use a different path when
preparing different inputs. Library versions describe the environment that
created the vectors; loading a cache does not recompute them after an upgrade.
