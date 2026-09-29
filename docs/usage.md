# API usage — bring your own class scores

The decision layer takes scores over a fixed set of classes and decides whether
to accept a single answer or defer. It does not train or call a model: you supply
the same kind of class weights for calibration and new requests. BANKING77 is
one application, not a dependency of this workflow.

## Run the quickstart

From the repository root, with Python 3.12 or later:

```bash
python -m pip install -e .
python -m examples.api.quickstart
```

The distribution name used by pip is `llm-scorekit`; Python imports use
`llm_scorekit`. If the repository is already installed, rerun the editable
installation above after this namespace change.

The core requires only NumPy. This example needs no model, downloaded data or
optional dependencies. It uses ten hand-written calibration rows and five
evaluation rows for `billing`, `delivery` and `returns`, with an 80% coverage
target. The rows illustrate correct and incorrect automatic decisions, as well
as empty and multi-label sets that require review. They are a usage demonstration,
not a statistical study.

The [full example](../examples/api/quickstart.py) saves its policy to
`outputs/quickstart/policy.json`, reloads it and evaluates the decisions.
Rerunning it replaces that demo file.

The [API mini notebook](../notebooks/api/01_quickstart.ipynb) walks through the
same workflow cell by cell.

## Prepare your inputs

- **One row per request, one column per class.** Supply a two-dimensional array
  of finite, nonnegative weights whose rows sum to one. SOCOP additionally
  requires every weight to be strictly positive. These weights need not be
  perfectly calibrated estimates of class probabilities.
- **Keep the labels and columns aligned.** `class_names` is an ordered sequence
  of unique names. Calibration labels are integer column indices: for
  `("billing", "delivery", "returns")`, label `1` means `delivery`.
- **Normalize before using the policy.** Raw logits and log-probabilities are
  not accepted inputs, and the policy does not apply softmax or normalize rows.
  Fix your conversion procedure before calibration and reuse it for prediction.
- **Reserve separate data.** Train the model and choose its scoring configuration
  on other examples. Use held-out labelled examples to calibrate, and a separate
  evaluation sample to measure performance. Any SOCOP parameter tuning must also
  finish before final calibration; do not select settings on the evaluation data.

## Calibrate, save and predict

The snippet below assumes you already have `calibration_probabilities`,
`calibration_labels`, `evaluation_probabilities` and `evaluation_labels` in that format.
The complete runnable example supplies all four arrays.

```python
from pathlib import Path

from llm_scorekit import calibrate, load_policy, save_policy

class_names = ("billing", "delivery", "returns")
policy = calibrate(
    calibration_probabilities,
    calibration_labels,
    class_names=class_names,
    alpha=0.20,
)

policy_path = Path("outputs/quickstart/policy.json")
policy_path.parent.mkdir(parents=True, exist_ok=True)
save_policy(policy, policy_path)

restored_policy = load_policy(policy_path)
decisions = restored_policy.predict(evaluation_probabilities)

for prediction in decisions.predictions:
    if prediction is None:
        print("defer")
    else:
        print(restored_policy.class_names[prediction])
```

LAC is the default method. To change it, pass `method="aps"` for deterministic,
non-randomized APS, or `method="socop", regularization=0.25` for SOCOP.
SOCOP's positive `regularization` value is its lambda parameter: choose it before
final calibration. The policy does not tune it automatically, and `0.25` here is
an example, not a recommended value for every dataset.

## Read and evaluate the decisions

- **`prediction_sets`** is a boolean matrix with the same class columns as the
  input. Each `True` means that the class remains in the request's set.
- **`accepted`** is a boolean vector selecting exactly the one-label sets.
  Both empty and multi-label sets are deferred.
- **`predictions`** is a list of class-column indices or `None` for deferred
  requests. Look up a class name only when the index is not `None`; deferral is
  not an instruction to silently substitute the model's top prediction.

Use the existing metrics on a separate labelled evaluation sample:

```python
from llm_scorekit import (
    automated_error_rate,
    automation_rate,
    average_set_size,
    empirical_coverage,
)

coverage = empirical_coverage(decisions.prediction_sets, evaluation_labels)
mean_size = average_set_size(decisions.prediction_sets)
automation = automation_rate(decisions.accepted)
error = automated_error_rate(
    decisions.predictions, evaluation_labels, decisions.accepted
)
```

Coverage measures whether the correct class remains in the set; automated-case
error measures mistakes only among accepted requests. If none are accepted,
that error is undefined and the function returns `NaN`, not zero.

## Reuse the policy with care

- **The JSON stores the policy, not the scoring system.** It records the method,
  alpha, calibrated threshold, class names, optional regularization and file
  format version. It does not save model weights, prompts, preprocessing or the
  calibration examples. Keep those separately and reuse the same scoring setup.
- **Class order is part of the contract.** Prediction checks the number of
  columns but cannot detect a permutation. Supply columns in the saved order;
  changing the model, prompt, preprocessing or class set requires recalibration.
- **Coverage is not an automatic-error guarantee.** Under exchangeability—for
  example, independent calibration and future requests from the same unchanged
  population—the target is marginal coverage of at least `1 - alpha`. It is not
  a promise for every batch, class or accepted subset, and it does not bound
  error among automatic decisions. Changes in incoming requests can invalidate
  the calibration assumptions even when the model stays fixed.

The [studies and results](README.md) show how these distinctions affect real
routing trade-offs. Use those reports for evidence and limitations, and this
guide for applying the same decision layer to your own scores.
