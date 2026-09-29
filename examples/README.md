# API quickstart and study progression

The API quickstart shows how to use the library with your own model outputs.
The studies show how the methods work and how the project developed. Both use
the same underlying functions in [src](../src/conformal_selective_prediction/).

## API — use the decision layer

[`api/quickstart.py`](api/quickstart.py) uses the API wrappers to calibrate a
policy, save/load it and evaluate new predictions. The [usage guide](../docs/usage.md)
explains the inputs and outputs. The example needs only the NumPy-based core,
with no model or dataset downloads. From the repository root:

```bash
python -m pip install -e .
python -m examples.api.quickstart
```

It writes `outputs/quickstart/policy.json`, replacing that demo file on reruns.
The reported numbers illustrate the API, not statistical performance.

The [API mini notebook](../notebooks/api/01_quickstart.ipynb) follows the same
workflow interactively, one cell at a time.

## Studies — follow the project's progression

These examples keep the detailed score, calibration, prediction-set and routing
steps explicit to demonstrate the methods. They do not use the policy wrappers;
the API coordinates those same functions for reuse. The table follows the
development order, from controlled data to real requests and LLMs. Exploratory
walkthroughs remain in [notebooks](../notebooks/).

| Study | Entry points | Results and instructions |
|:------|:-------------|:-------------------------|
| 1. Synthetic validation | [`synthetic/multiclass.py`](synthetic/multiclass.py) | [Report](../docs/synthetic/validation.md) |
| 2. TF-IDF baseline | [`single_run.py`](banking77/baseline/single_run.py), [`validation.py`](banking77/baseline/validation.py) | [Report and commands](../docs/banking77/baseline/tfidf_lac.md) |
| 3. LAC, APS and SOCOP | [`banking77/scores/`](banking77/scores/) | [Comparison and commands](../docs/banking77/scores/lac_aps_socop.md) |
| 4. Frozen text encoder | [`banking77/representations/`](banking77/representations/) | [Running guide](banking77/representations/README.md), [report](../docs/banking77/representations/tfidf_vs_encoder.md) |
| 5. Local Qwen | [`banking77/llm/`](banking77/llm/) | [Running guide](banking77/llm/README.md), [report](../docs/banking77/llm/qwen.md) |

### Run a study

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

Study artifacts are saved under `outputs/`, outside Git. Rerunning evaluation
scripts can replace their same-named output files.
