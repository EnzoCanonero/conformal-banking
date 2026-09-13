# Synthetic multiclass validation

The synthetic milestone checks the conformal workflow before applying it to real
banking requests. It combines a small end-to-end classifier example with repeated
IID simulations of the core score, calibration and prediction-set calculations.

## End-to-end example

[synthetic_multiclass.py](../examples/synthetic_multiclass.py) generates 8,000
examples with four classes and 20 features, using seed `42`. It randomly partitions
them into 4,000 training, 2,000 calibration and 2,000 test examples.

A standard scaler and logistic regression are fitted only on the training data.
True-label LAC scores on the calibration data determine the threshold for
`alpha=0.1`. The script then builds test prediction sets and automates only
singletons, deferring empty and multi-label sets.

The reference run produces these rounded results:

| Metric | Result |
|:-------|-------:|
| Target set coverage | 90.0% |
| Empirical set coverage | 90.8% |
| Average prediction-set size | 1.155 |
| Automation rate | 84.5% |
| Error among automated cases | 10.1% |

Coverage measures true-label inclusion across all test examples. Automation is
the fraction with singleton sets, and automated-case error uses only that selected
subset. A set containing the true label can still have several labels and be
deferred, so coverage and automation measure different things.

This one run illustrates the workflow. Coverage in a finite test sample can fall
above or below its target; a single result does not establish the marginal
guarantee. The selected-case error is also an empirical measurement, not a bound
implied by `alpha`.

## Repeated-simulation check

[test_conformal.py](../tests/test_conformal.py) checks the core LAC pipeline over
30 independently seeded simulations. Each draws three-class probability vectors
and samples labels from those probabilities, then separates 1,000 calibration
and 5,000 test examples. The test checks that mean coverage is within one
percentage point of the 90% target.

This is a regression check under a controlled IID construction, separate from
the trained four-class example above. Other focused tests cover quantile ranks,
true-label scoring, threshold inclusion, set metrics and uncertainty intervals.
[test_selection.py](../tests/test_selection.py) covers singleton selection and
automated-case error. These checks protect the implementation; they do not
establish exchangeability for a real dataset.

## Reproduce

From the repository root, after setting up the environment:

```bash
python -m pip install -e ".[example,dev]"
python examples/synthetic_multiclass.py
python -m pytest tests/test_conformal.py tests/test_selection.py
```

The example prints its metrics and downloads no data. The next completed
milestone is the [BANKING77 baseline](banking77_validation.md), which evaluates
the same decision layer on text classification and reports its limitations.
