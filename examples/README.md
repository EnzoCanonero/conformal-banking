# Runnable studies

These scripts reproduce the studies discussed in the [reports](../docs/README.md).
The folders follow the same progression: synthetic validation, then BANKING77
baselines, scores, representations and LLMs. Exploratory walkthroughs remain in
[notebooks](../notebooks/); reusable decision-layer functions live in
[src](../src/conformal_selective_prediction/).

## Where to start

| Study | Entry points | Results and instructions |
|:------|:-------------|:-------------------------|
| Synthetic validation | [`synthetic/multiclass.py`](synthetic/multiclass.py) | [Report](../docs/synthetic/validation.md) |
| TF-IDF baseline | [`single_run.py`](banking77/baseline/single_run.py), [`validation.py`](banking77/baseline/validation.py) | [Report and commands](../docs/banking77/baseline/tfidf_lac.md) |
| LAC, APS and SOCOP | [`banking77/scores/`](banking77/scores/) | [Comparison and commands](../docs/banking77/scores/lac_aps_socop.md) |
| Frozen text encoder | [`banking77/representations/`](banking77/representations/) | [Running guide](banking77/representations/README.md), [report](../docs/banking77/representations/tfidf_vs_encoder.md) |
| Local Qwen | [`banking77/llm/`](banking77/llm/) | [Running guide](banking77/llm/README.md), [report](../docs/banking77/llm/qwen.md) |

## Running an example

Follow the root [installation instructions](../README.md#run-locally), then run
commands from the repository root. The synthetic example needs no downloaded data:

```bash
python -m examples.synthetic.multiclass
```

For BANKING77, first follow the [data guide](../data/README.md). A single baseline
run is:

```bash
python -m examples.banking77.baseline.single_run
```

The comparison studies separate preparation from evaluation: `prepare.py`
creates or resumes cached predictions, individual method scripts evaluate
routing, and `compare.py` combines results. Use each study's guide or report
for its command sequence and optional dependencies. Qwen scoring can take
hours; use its documented `--dry-run` preview before starting inference.

Generated artifacts remain under the existing `outputs/` paths, outside Git.
The folder reorganisation does not require regenerating cached predictions.
Rerunning evaluation scripts can replace their same-named output files.
