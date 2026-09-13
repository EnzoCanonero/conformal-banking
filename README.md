# Conformal Selective Prediction

[![Tests](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml)

## When should a classifier act, and when should it defer?

A classifier normally picks one label, even when several answers are plausible.
This project studies when to act on that prediction and when to defer to a human,
starting with banking-support intent routing.

It adds **conformal prediction sets** to a classifier to represent ambiguity,
then measures how much work can be automated and how often those automated
decisions are wrong. The decision layer is reusable across classifiers;
comparisons with other text representations and LLMs are planned.

### How it works

1. **Train a classifier** to produce probabilities over the possible labels.
2. **Calibrate prediction sets** using a separate held-out partition and a chosen
   coverage target. A set can contain one, several, or no labels.
3. **Act or defer:** automate a one-label set; send multi-label and empty sets
   to human review. Evaluate both set coverage and automated-case error on test data.

Coverage measures whether the set contains the true label. The conformal
marginal-coverage guarantee requires exchangeability between calibration and
future examples; it does **not** guarantee low error among automated decisions
or nominal coverage for every intent.

## What's in this repository

- **A reusable conformal core:** LAC scoring, finite-sample calibration,
  prediction sets, singleton selection, and coverage and automation metrics.
- **Runnable experiments:** a synthetic multiclass example and a BANKING77
  TF-IDF/logistic-regression baseline, including repeated-split evaluation and
  a naive confidence-threshold comparison.
- **Reports and checks:** documented results and figures, focused tests,
  and CI running pytest, Ruff and mypy.

The scope is fixed-label classification, without agent construction, RAG or
model fine-tuning. The table below distinguishes completed work from planned work.

## Milestones

| Milestone | Status | Main outcome or next question |
|:----------|:-------|:------------------------------|
| Synthetic foundation | Complete | End-to-end workflow and repeated IID coverage checks. [Report](docs/synthetic_validation.md). |
| BANKING77 baseline | Complete | LAC versus naive thresholding across five splits. [Report](docs/banking77_validation.md). |
| Representation and score comparisons | Planned | Compare a frozen pretrained text encoder with TF-IDF, and APS, an alternative conformal score, with LAC. |
| LLM classification | Planned | Score the same intent labels with a local LLM and compare automation versus error. |
| Distribution-shift experiments | Planned | Measure what changes when deployment inputs or prompts differ from calibration. |

### BANKING77: the first real-data result

The baseline routes banking-support requests among 77 intents. TF-IDF converts
request text into word-based numerical features, and logistic regression predicts
the intent probabilities. LAC automates singleton prediction sets. The
naive policy automates the model's top label when its probability meets a
fixed cutoff, without conformal calibration.

The table shows averages across five training/calibration splits, all evaluated
on the same official test set. Automation is the fraction of all requests
handled automatically; error is measured only among those handled requests.

| Policy and reference setting | Automation rate | Error among automated cases |
|:-----------------------------|----------------:|----------------------------:|
| LAC, 90% target coverage (`alpha=0.1`) | 53.49% | 5.51% |
| Naive confidence cutoff (`tau=0.5`) | 24.49% | 1.33% |

At these settings, LAC handles more requests with more automated-case error.
These are different operating points, not a matched-automation comparison or a
ranking of the methods. LAC's mean set coverage is 90.08% at the 90% target,
but some intents have substantially lower coverage.

![BANKING77 automation versus error among automated predictions, with five traces per policy](docs/figures/banking77/automation_vs_error.png)

The figure extends the comparison over a fixed grid of settings. Moving right
means automating more requests; moving down means fewer errors among automated
cases. Blue circles show LAC and orange squares show naive thresholding. Each
line is one split, and each point is a different policy setting. The overlapping
traces are not confidence bands. LAC can turn back as shrinking singleton sets
become empty and are deferred.

The [BANKING77 report](docs/banking77_validation.md) explains all four figures,
the uncertainty across splits, intent-level weaknesses, and the limits of the
coverage claim. These results establish a baseline, not a deployment policy.

## Repository structure

```text
src/conformal_selective_prediction/   # Conformal core, data loader and model baseline
examples/
├── synthetic_multiclass.py           # Self-contained example
├── banking77_baseline.py             # Single-split walkthrough
└── banking77_validation.py           # Repeated-split evaluation
tests/                               # Core mathematical and policy tests
docs/                                # Reports and versioned figures
data/README.md                       # Dataset setup and limitations
.github/workflows/ci.yml             # Automated checks
pyproject.toml                       # Package and tool configuration
```

Downloaded data and generated experiment outputs are ignored by Git. The figures
under `docs/figures/` are report snapshots and are not overwritten by experiment runs.

## Installation and running

Use Python 3.12 or later. From the repository root, on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[example,dev]"
```

This installs the examples and development tools. For the NumPy-only library,
use `python -m pip install -e .`. Dependencies are defined in
[pyproject.toml](pyproject.toml).

Start with the synthetic example, which needs no downloaded data:

```bash
python examples/synthetic_multiclass.py
```

For BANKING77, first follow the [data setup instructions](data/README.md).
Then run the single-split walkthrough or the full validation:

```bash
python examples/banking77_baseline.py
python examples/banking77_validation.py
```

Validation writes CSVs and figures to `outputs/banking77/`, replacing the same-named
files on subsequent runs. Experiment settings and reproduction details are in the
[synthetic report](docs/synthetic_validation.md) and
[BANKING77 report](docs/banking77_validation.md#reproduce-the-report).

## License

Code is available under the [MIT License](LICENSE). BANKING77 has a separate
dataset license, documented in [data/README.md](data/README.md#source).
