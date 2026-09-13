# Conformal Selective Prediction

[![Tests](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml)

This repository develops a conformal selective-prediction layer for reliable
automation. Instead of forcing a model to act on every input, it produces
prediction sets with coverage guarantees and defers uncertain cases to human
review.

The framework will first be validated on BANKING77 for automated banking-support
routing. It will then be applied to LLM outputs and stress-tested under
distribution shift to measure the trade-off between automation rate and
operational risk.

## Current status

The reusable multiclass split-conformal core now includes LAC scores,
finite-sample calibration, prediction-set construction, and basic set-quality
metrics. Its singleton policy automates one-class prediction sets and defers all
other cases. A synthetic IID multiclass example exercises the complete workflow
and measures both marginal coverage and the automation-versus-error trade-off.

BANKING77 data loading now preserves the official test set and creates a
reproducible, stratified training/calibration split with a shared intent mapping.
The TF-IDF/logistic-regression baseline fits both preprocessing and the classifier
on the training partition. A single-split BANKING77 experiment now compares LAC
singleton selection with naive confidence thresholding on the same predictions.
A repeated-split validation script evaluates fixed policy grids and saves
uncertainty intervals, class-level diagnostics and four comparison figures.

## Installation

Create and activate a virtual environment, then install the package in editable
mode:

```bash
python -m pip install -e .
```

For development tools:

```bash
python -m pip install -e ".[dev]"
```

The examples use scikit-learn for modelling and Matplotlib for validation
figures, kept separate from the NumPy-only conformal core:

```bash
python -m pip install -e ".[example]"
```

## BANKING77 baseline

Follow the [data setup instructions](data/README.md) to download the pinned
official files, load them and create the 75/25 training/calibration split.

The baseline uses word unigrams and bigrams with multinomial logistic regression.
It starts with L2 regularization, `C=1.0`, the `lbfgs` solver and a maximum of
1,000 iterations, without hyperparameter tuning.

```python
from conformal_selective_prediction.data import load_banking77
from conformal_selective_prediction.models import fit_tfidf_classifier

data = load_banking77("data/raw/banking77", random_seed=42)
model = fit_tfidf_classifier(data.train)

calibration_probabilities = model.predict_proba(data.calibration.texts)
test_probabilities = model.predict_proba(data.test.texts)
```

Prediction transforms held-out texts without refitting the vectorizer or
classifier. Probability column `j` corresponds to `model.classes_[j]`. With all
77 intents present in training, these are the indices `0` through `76` from
`data.class_names`; no second label encoding is introduced.

### End-to-end experiment

After downloading the data, run from the repository root:

```bash
python examples/banking77_baseline.py
```

The experiment uses seed `42`, LAC miscoverage level `alpha=0.1` and a naive
confidence threshold of `0.5`, fixed before test evaluation. These values are
defined at the top of the script. The calibration partition is used only to
estimate the LAC score threshold; neither held-out partition is used for model
tuning.

LAC automates only singleton prediction sets. The naive policy automates the
most probable intent when its probability is at least `0.5`. Both policies use
the same classifier and test probabilities. The script reports classifier
accuracy, LAC coverage and average set size, and each policy's automation rate
and automated-case error. An error of `nan` means no cases were automated, not
zero errors.

The first run with these settings gives classifier accuracy `0.841`:

| Policy | Set coverage | Average set size | Automation rate | Automated-case error |
|:-------|-------------:|-----------------:|----------------:|---------------------:|
| LAC singleton | 0.895 | 1.524 | 0.547 | 0.054 |
| Naive threshold | — | — | 0.240 | 0.015 |

This is a comparison at two different operating points, not a matched-automation
comparison or a policy ranking. The LAC target applies to prediction-set coverage,
not error among automated cases; the naive threshold has no conformal coverage
guarantee. A single split does not establish stability or satisfy the full
Phase 1 acceptance gate. See the [data limitations](data/README.md#split-limitations)
when interpreting the results.

### Repeated-split validation

Run from the repository root after installing the `example` extra and downloading
the data:

```bash
python examples/banking77_validation.py
```

The script uses seeds `7, 21, 42, 84, 123`. For each split it fits the model once,
then reuses calibration scores and test probabilities over two fixed grids:

- LAC `alpha`: `0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50`.
- Naive confidence threshold: `0.0` through `1.0` in steps of `0.1`.

The grids are defined at the top of the script, not selected from test results.
No "best" threshold is chosen. Outputs go to the Git-ignored `outputs/banking77/`
directory; rerunning replaces the same named files.

| File | Contents |
|:-----|:---------|
| `config.json` | Seeds, grids, interval settings and library versions |
| `lac_metrics.csv` | Per-split coverage, set size, empty-set rate and selection metrics |
| `naive_metrics.csv` | Per-split naive selection metrics |
| `class_coverage.csv` | Coverage and counts for each intent, split and alpha |
| `set_sizes.csv` | Set-size counts and fractions, including empty sets |
| `nominal_vs_empirical_coverage.png` | Coverage curves with per-split intervals |
| `coverage_vs_set_size.png` | Coverage versus average prediction-set size |
| `automation_vs_error.png` | LAC and naive automation-versus-error curves |
| `set_size_distribution.png` | Mean per-split set-size fractions for alpha `0.01, 0.05, 0.10, 0.30` |

Coverage and automated-case error include pointwise 95%
[Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
Coverage uses all test cases; automated-case error uses only the selected cases.
The CSVs retain numerator and denominator counts. With no automated cases, the
error and its interval are `nan`, and that operating point is omitted from the
risk plot. Zero observed errors with a positive denominator still gives a
nonzero upper interval bound.

Every seed reuses the same official test set. Curves across seeds describe
sensitivity to the training/calibration split, not independent new test samples.
Counts are never pooled across seeds to construct intervals. Class-level
intervals use only 40 examples per intent and are not simultaneous guarantees
across all intents or grid points. The fixed, class-balanced test set also makes
the global binomial intervals approximate diagnostics, not a proof of
exchangeability or conformal validity on this benchmark.

Increasing `alpha` shrinks LAC sets, so coverage and average set size cannot
increase on a fixed split. Singleton automation need not be monotone: a set can
move from multiple labels to one label and then become empty. Automated-case
error remains an empirical measurement, not an `alpha`-level guarantee.

## Synthetic IID example

Run the complete training, calibration, and evaluation workflow with:

```bash
python examples/synthetic_multiclass.py
```

Using the fixed seed and a 4,000/2,000/2,000 train/calibration/test split, the
example produces:

| Target coverage | Empirical coverage | Average set size | Automation rate | Automated-case error |
|----------------:|-------------------:|-----------------:|----------------:|---------------------:|
| 0.900           | 0.908              | 1.155            | 0.845           | 0.101                |

A single finite test set can fall slightly above or below the coverage target
because the conformal guarantee is marginal rather than a deterministic lower
bound for every realized test set. The automated-case error is measured only
among singleton predictions and is not controlled by that marginal guarantee.

## Statistical scope

The intended split-conformal guarantee is finite-sample **marginal coverage**.
It relies on exchangeability of the calibration examples and future examples.
The guarantee does not automatically imply conditional coverage for every
subgroup, or a bound on the error rate among cases selected for automation.

## License

This project is available under the [MIT License](LICENSE).
