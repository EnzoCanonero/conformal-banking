# BANKING77 LLM scoring and routing

This example turns the notebook's Qwen scoring procedure into a small, resumable
preparation step. It scores all 77 intents for each request and saves the raw
results. Separate scripts then evaluate LAC, naive confidence and tuned SOCOP
using that cache, without loading the LLM again. A comparison script evaluates
the same rules on saved TF-IDF and frozen-encoder predictions. The results
report is the next step.

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

We use seed **42** and the complete held-out partitions from the existing
TF-IDF and encoder studies:

| Partition | Requests | Later purpose |
|:----------|---------:|:--------------|
| Tuning A | 625 | SOCOP tuning, paired with B |
| Tuning B | 625 | SOCOP tuning, paired with A |
| Final calibration | 1,251 | Set thresholds after scoring and tuning choices are fixed |
| Official test | 3,080 | Compare routing rules on the same requests |
| **Total** | **5,581** | |

The first three partitions come from the official training data, outside the
7,502 records used to fit the seed-42 classifiers. The original tuning halves
and final-calibration partition remain separate. Original row IDs and label
order are retained, allowing later alignment with saved TF-IDF and encoder
predictions without retraining those classifiers. A matched comparison should
use their seed-42 predictions and the same routing settings, not their historical
five-run averages.

This is one fixed-split study, not a repeat of the earlier five-run benchmark.
The exploratory notebooks sampled from the same training calibration pool, so
these are not all previously unseen development examples. The first LLM study
also evaluated 500 of the official test requests, and the other classifiers
have used the full test set. This expansion is therefore not a fresh blind
evaluation. Model, prompt, scoring rule and routing grids remain unchanged.
The training and test intent mixtures still differ, so using more examples
does not establish the exchangeability needed by the coverage guarantee.

## Run from the repository root

Use the existing `llm-scorekit` environment. If setting up another Apple-silicon
environment, install `python -m pip install -e ".[example,llm]"`. Keep the local
files described in [the data guide](../../data/README.md) available.

Preview the sample sizes, cache status and estimated time without importing MLX,
loading the model or writing outputs:

```bash
python -m examples.banking77_llm.prepare --dry-run
```

Run scoring, keeping the Mac connected to power. `caffeinate` prevents idle
sleep while the command runs; keep the laptop lid open:

```bash
caffeinate -i python -m examples.banking77_llm.prepare
```

The completed **1,400-request** run is reused, leaving **4,181 new requests**.
Its measured rate, about **8.42 seconds per request** on the M1 with 16 GB,
projects **9 hours 47 minutes** of additional scoring. Budget **10–12 hours**
with no other model running: checkpoint writes, request lengths and machine
load affect wall time. Without the original cache, scoring all 5,581 requests
would take roughly 13 hours at that rate. Model files use the existing Hugging
Face cache, with a download only if the pinned revision is missing.

## Saved scores and resuming

The new output is `outputs/banking77/llm/full/prepared/scores.npz`, ignored by
Git. The original `outputs/banking77/llm/prepared/scores.npz` and its routing
results are left untouched.

On the first run, matching completed scores from the original cache are copied
into the new checkpoint before loading the model. They remain at the start of
the request list; the remaining requests follow. Rows are therefore not grouped
contiguously by partition: the saved partition field identifies their role.

The archive contains:

- `metadata`: JSON containing the model, prompt, scoring rule, class order and
  the complete ordered request list, with text, label, original ID and partition.
- `log_scores`: one row per completed request and one column per intent. During
  a partial run these rows correspond to the **first completed requests** in
  the metadata list, not the entire plan.
- `token_counts`: candidate lengths including EOS, in the same intent order.
- `scoring_seconds`: accumulated request-scoring time, including the reused
  work but excluding model loading and checkpoint writes.

Each completed request is checkpointed through a temporary archive, then replaces
the previous checkpoint. Run the same command after an interruption to resume;
at most the in-progress request needs rescoring. Once the new checkpoint exists,
resuming uses it directly and no longer needs the original cache. A complete
cache skips model loading. Do not run two preparation processes against the same
output path.

Defaults are collected at the top of `prepare.py`. Changing inputs, the prompt
or recorded settings rejects incompatible cached scores rather than silently
mixing results. For a deliberately different scoring setup, use a new output
path and call `prepare_study(..., reuse_cache_path=None)` to start fresh.
Load archives with `np.load(path, allow_pickle=False)`. No model weights or
Python objects are stored in them.

## Evaluate routing from the cache

Once all **5,581 requests** are scored, run the methods independently:

```bash
python -m examples.banking77_llm.lac
python -m examples.banking77_llm.naive
python -m examples.banking77_llm.socop
```

These commands need only the completed cache and the `example` dependencies;
they do not import MLX, read the raw dataset or call the model. Summed scores
are normalized with softmax, exactly as in the notebook. There is no temperature
fitting or switch to mean scoring.

- **LAC** calibrates on the 1,251 reserved examples. Coverage targets are
  99%, 95%, 90%, 80%, 70%, 60% and 50%.
- **Naive** evaluates fixed weight cutoffs: `0`, `0.5`, `0.8`, `0.9`, `0.95`,
  `0.99`, `0.999`, `0.9999`, `0.99999` and `0.999999`. The finer spacing near one
  reflects the concentrated weights seen in the notebook, not a cutoff search
  on the test requests.
- **SOCOP** uses the same coverage targets and the existing lambda grid:
  `0.001`, `0.003`, `0.01`, `0.05`, `0.10`, `0.25`, `0.50`, `1.00`. For each
  target, it calibrates on tuning A and counts singletons on B, then swaps the
  roles, using 625 examples in each half. It chooses the lambda with the most
  combined singletons, breaking ties by smaller total set size and then smaller
  lambda. Final thresholds come from the separate 1,251 calibration examples,
  after all lambdas are selected.

The grids are retained from the first 500-test-request study, without changes
for this expansion. Test results describe the operating choices; they do not
select a deployment policy. Only tuned SOCOP is included here, without
repeating fixed-lambda or APS studies.

Results are written under `outputs/banking77/llm/full/`:

- `lac/`: `metrics.csv`, `set_sizes.csv`, `manifest.json`.
- `naive/`: `metrics.csv`, `manifest.json`.
- `socop/`: the same files as LAC, plus `tuning_candidates.csv` and
  `tuning_choices.csv`.

Metrics include classification accuracy, automation, automated-case error and
its 95% Wilson interval. Conformal rows also include coverage and its interval,
average set size, empty-set rate and mean set size among deferred requests
(including empty sets). Error is `NaN` when nothing is automated, not zero.
Intervals are pointwise estimates for this fixed evaluation, not guarantees
about every configuration or future calibration split.

The manifests identify the source score archive by its SHA-256 and record the
evaluation settings. Rerunning a method replaces only its result files, never
the scores or historical classifier results.

SOCOP sums each probability segment directly when constructing its scores.
For these concentrated LLM weights, subtracting cumulative masses near one
can erase a positive segment and cause division by zero. Direct summation
avoids that loss without clipping weights or changing the score definition.

## Compare with TF-IDF and the frozen encoder

With the full LLM cache and the existing seed-42 classifier archives available,
run:

```bash
python -m examples.banking77_llm.compare
```

This needs only the `example` dependencies. It does not train classifiers, load
the encoder or call the LLM. The separate routing commands above do not need
to run first: the comparison recalculates metrics directly from saved scores.

- **Match the data before comparing results.** The script checks intent order,
  labels and identical record IDs in all four held-out partitions, then aligns
  classifier rows with the LLM cache. No evaluation record may belong to
  classifier training. It uses the full 3,080-request test set and the same
  625 / 625 / 1,251 tuning and calibration budgets for every model.
- **Apply the same evaluation procedure, not the same numerical thresholds.**
  LAC is recalibrated for each model. SOCOP lambda is selected separately for
  each model and coverage target, using the shared grid and tuning halves;
  final calibration remains separate. Naive uses the union of the already
  declared classifier and LLM cutoff grids, including the LLM's cutoffs near
  one. Equal cutoffs need not give equal automation across models.
- **Compare complete systems, not equal training budgets.** Both logistic
  regressions were trained on the same 7,502 labelled requests. Qwen uses a
  fixed prompt and no task-specific training. This is one matched seed, not
  the earlier five-run average. Prior test inspection and the official split
  limitations still apply; these results do not select a deployment policy.

Outputs go to `outputs/banking77/llm/full/comparison/`, leaving previous studies
and score caches untouched:

- `accuracy.csv`: unrestricted classification accuracy for each model.
- `conformal_metrics.csv`: LAC and tuned SOCOP coverage, automation, error,
  set sizes and selected lambdas. Lambda is `NaN` for LAC.
- `naive_metrics.csv`: automation and error at the shared confidence cutoffs.
- `tuning_candidates.csv`: SOCOP's A/B tuning results for each model and target.
- `manifest.json`: source hashes, model settings and the common evaluation grid.
- `figures/`: separate LAC and SOCOP model-comparison plots with matching axes,
  plus a Qwen-only plot comparing LAC, SOCOP and naive confidence.

The horizontal axis measures automation; the vertical axis measures errors
among automated requests. Points follow the evaluated parameter order, so
curves can turn back when prediction sets become empty. Lines connect tested
settings; they are not fitted frontiers or confidence bands. Zero-automation
settings have undefined error and are omitted from the plots.
