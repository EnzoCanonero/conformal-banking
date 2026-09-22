# Frozen encoder preparation

This prepares the TF-IDF versus frozen-encoder study. It caches text vectors,
then trains logistic regression on them using the same requests and settings
as the existing TF-IDF models. The saved probabilities will support the later
LAC/SOCOP comparison; this step does not evaluate routing or compare performance.

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

## Prepare BANKING77 predictions

Keep the local BANKING77 files and the completed TF-IDF preparation available:
`outputs/banking77/score_comparison/prepared/` must contain its manifest and
five seed archives. Then run from the repository root:

```bash
python -m examples.banking77_representations.prepare
```

- **A paired comparison:** the saved TF-IDF sample IDs determine which requests
  train each encoder-based classifier and which belong to tuning, calibration
  and test. The preparation checks the dataset, labels and separation of these
  partitions before fitting. It preserves all 3,080 official test requests and
  the same five seeds, with 7,502 training requests, two tuning halves of 625
  and 1,251 final-calibration requests per seed.
- **Only the representation changes:** each classifier uses the TF-IDF
  manifest's logistic-regression settings: `C=1`, `l1_ratio=0`, `solver="lbfgs"`
  and `max_iter=1000`. The encoder remains frozen. The existing TF-IDF models
  are not retrained, and their artifacts are neither copied nor overwritten.
- **Embeddings are computed once:** a stable ordering of the original sample
  IDs lets all five splits share one cache. Probabilities are then saved for
  each tuning half, final calibration and test, preserving the original class
  order. Elapsed encoding-call time is recorded for the later report; it includes
  model loading and, if needed, downloading, not just inference.

New files are saved under
`outputs/banking77/representation_comparison/encoder/`:

```text
embeddings.npz
prepared/
├── manifest.json
└── seed_*.npz
```

The encoder preparation has its own identity and records the source TF-IDF
directory, preparation ID and manifest hash. Keep that source unchanged for
the paired study: regenerating it invalidates the recorded pairing. The two
representations deliberately have different preparation IDs; they are paired
through their data and split IDs, not by treating them as the same model run.
