# BANKING77 LLM scoring

This example turns the notebook's Qwen scoring procedure into a small, resumable
preparation step. It scores all 77 intents for each request and saves the raw
results. It does **not** tune SOCOP, calibrate prediction sets or evaluate routing
yet; those steps will reuse this cache without loading the LLM again.

## Fixed setup

- **Model:** `mlx-community/Qwen3-4B-Instruct-2507-4bit`, revision
  `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Inference runs locally with MLX on
  Apple silicon. No model training or generated answers are involved.
- **Prompt:** the same instruction as the notebooks, listing every intent in
  the dataset's existing order. Each request is a user message. The tokenizer's
  chat template supplies the assistant-response prefix.
- **Score:** sum the log-probabilities of the candidate's tokens, including its
  end-of-response token. Longer names can be penalised by this sum. We also
  save token counts so mean scoring can be explored later without new inference;
  any such choice must use tuning data, not final calibration or test results.
- **Interpretation:** these are likelihoods of particular label strings.
  Softmax can later turn them into relative candidate weights, but those weights
  are not calibrated probabilities of a correct classification.

The short scoring loop follows the notebook helper. It lives here independently
so this example does not depend on notebook code; the current notebooks remain
untouched.

## Data protocol

We use seed **42** and the existing baseline splitting function. Stratified
subsamples preserve the intent proportions within each source partition:

| Partition | Requests | Later purpose |
|:----------|---------:|:--------------|
| Tuning A | 200 | SOCOP tuning, paired with B |
| Tuning B | 200 | SOCOP tuning, paired with A |
| Final calibration | 500 | Set thresholds after scoring and tuning choices are fixed |
| Official test subset | 500 | Compare routing rules on the same requests |

The first three partitions come from the official training data, outside the
7,502 records used to fit the seed-42 classifiers. The original tuning halves
and final-calibration partition remain separate. Original row IDs and label
order are retained, allowing later alignment with saved TF-IDF and encoder
predictions. Those baselines will need recalibration and, for SOCOP, retuning
on these smaller partitions; their historical aggregate results are not a
matched comparison.

This is one fixed-split study, not a repeat of the earlier five-run benchmark.
The exploratory notebooks sampled from the same training calibration pool, so
these are not all previously unseen development examples. Keep the scoring
configuration fixed for this run. The official test subset is separate from
the notebook samples, although the test set has already been studied with the
other classifiers. The training and test intent mixtures still differ, so
this preparation does not establish the exchangeability needed by the coverage
guarantee.

## Run from the repository root

Use the existing `llm-scorekit` environment. If setting up another Apple-silicon
environment, install `python -m pip install -e ".[example,llm]"`. Keep the local
files described in [the data guide](../../data/README.md) available.

Preview the sample sizes, cache status and estimated time without importing MLX,
loading the model or writing outputs:

```bash
python -m examples.banking77_llm.prepare --dry-run
```

Once the notebook has finished, run scoring:

```bash
python -m examples.banking77_llm.prepare
```

The default **1,400 requests** take approximately **3 hours 11 minutes**, scaling
the previously observed 10.9 minutes for 80 requests. Budget around **3–4 hours**
on the M1 with 16 GB, with no other model running. This is an extrapolation, not
a new benchmark; model loading, checkpoint I/O, request lengths and machine load
can change wall time. Model files use the existing Hugging Face cache, with a
download only if the pinned revision is missing.

## Saved scores and resuming

The output is `outputs/banking77/llm/prepared/scores.npz`, ignored by Git:

- `metadata`: JSON containing the model, prompt, scoring rule, class order and
  the complete ordered request list, with text, label, original ID and partition.
- `log_scores`: one row per completed request and one column per intent. During
  a partial run these rows correspond to the **first completed requests** in
  the metadata list, not the entire plan.
- `token_counts`: candidate lengths including EOS, in the same intent order.
- `scoring_seconds`: accumulated request-scoring time, excluding model loading
  and checkpoint writes.

Each completed request is checkpointed through a temporary archive, then replaces
the previous checkpoint. Run the same command after an interruption to resume;
at most the in-progress request needs rescoring. A complete cache skips model
loading. Do not run two preparation processes against the same output path.

Defaults are collected at the top of `prepare.py`. Changing inputs, the prompt
or recorded settings rejects an existing cache rather than silently mixing
results; choose a new `CACHE_PATH` for a different preparation. Load archives
with `np.load(path, allow_pickle=False)`. No model weights or Python objects are
stored in them.
