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
on the training partition. The next milestone will connect its probabilities to
the conformal decision layer and compare LAC with naive thresholding.

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

The synthetic example and BANKING77 workflow use scikit-learn, which is
kept separate from the NumPy-only conformal core:

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

The BANKING77 conformal evaluation experiment is not implemented yet. The
calibration and test partitions must not be used to tune the model.

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
