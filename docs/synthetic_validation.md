# Synthetic multiclass validation

This study compares LAC, APS and SOCOP on generated data before taking the
additional scores into the banking experiment. One experiment shows how the
three methods change automatic decisions; repeated simulations separately check
whether coverage behaves as expected.

## One classifier, three shortlist rules

The [synthetic example](../examples/synthetic_multiclass.py) compares the methods
without changing the classifier:

- **The data and model stay fixed.** Seed `42` generates 8,000 examples with four
  classes and 20 numerical features: 4,000 examples for training, 2,000 for
  calibration and 2,000 for evaluation. Feature scaling and logistic regression
  use only the training examples. All three methods receive the same model
  probabilities, so differences come from the shortlist rules, not model training.

- **LAC and APS use the probabilities differently.** LAC considers each class
  probability on its own. APS orders classes from most likely to least likely
  and adds their probabilities down the list; each class gets the total reached
  at its position.
  We use non-randomized APS, with tied probabilities following class order.

- **SOCOP explicitly discourages multiple-answer sets.** Unlike simply trying
  to shorten every list, it puts extra emphasis on keeping a single answer,
  while still penalizing long lists. We use the full
  [SOCOP score](https://arxiv.org/html/2509.24095v2), keeping `lambda=0.25` as
  the central choice. Two additional settings, `0.05` and `1.0`, illustrate how
  this parameter changes the balance: lower values put more emphasis on avoiding
  multiple answers, while higher values put more emphasis on average set size.
  They were chosen before running the additional comparisons, not selected as
  winners from the test results. Lambda is not a coverage target or an error limit.

- **Calibration and routing follow the same protocol.** Each configuration,
  including each SOCOP lambda, learns its own threshold from the same calibration
  examples for a 90% coverage target (`alpha=0.1`). We automate only when the
  resulting shortlist contains one
  answer. Several answers, or no answers, mean review; we do not force a label
  into an empty set. Each SOCOP configuration uses the same lambda for calibration
  and prediction.

## What changes when we switch the score?

These are single-run results, all using the same 90% coverage target. The first
percentage counts sets containing the correct answer; the last counts wrong
decisions only among examples handled automatically.

| Rule and setting | Coverage | Average set size | Automation rate | Automated-case error rate |
|:-----------------|---------:|-----------------:|----------------:|--------------------------:|
| LAC | 90.80% | 1.16 | 84.50% | 10.12% |
| APS | 92.35% | 1.88 | 28.75% | 9.04% |
| SOCOP, `lambda=0.05` | 90.90% | 1.29 | 86.00% | 10.52% |
| **SOCOP, `lambda=0.25` — reference** | 91.00% | 1.22 | 85.45% | 10.36% |
| SOCOP, `lambda=1.0` | 91.20% | 1.18 | 84.60% | 10.05% |

- **The central SOCOP choice is close to LAC.**
  It handles 0.95 percentage points more examples automatically, while the error
  rate among those handled rises from 10.12% to 10.36%. Its average set is also
  slightly larger. More single-answer sets can coexist with a larger average
  set size, so neither metric alone identifies the better routing rule.

- **Changing lambda shows different priorities, with modest effects here.**
  Lowering it to `0.05` increases automation to 86.00%, but also increases average
  set size and automated-case error. Raising it to `1.0` reduces all three in
  this run. The central value remains our reference, not a claimed optimum, and
  these results do not imply that automation or error must always move this way.

- **APS retains more answers, but automates much less.** Only 28.75% of examples
  receive a single answer, compared with more than 84% for LAC and SOCOP. Its
  coverage is higher and its automated-case error is lower, but it handles a
  much smaller, different subset. This does not show that APS would make fewer
  mistakes at the same automation rate.

These results do not establish a general winner or an optimal SOCOP setting.
The methods share a coverage target, but their measured coverage differs, and a
90% target for retaining the correct answer does not mean that at most 10% of
automatic decisions will be wrong.

## Repeated-simulation check

The [simulation test](../tests/test_conformal.py) runs 30 independently seeded
simulations. Each generates probabilities for three classes and draws the correct
answers from them, with no classifier training. All three methods receive the same
1,000 calibration and 5,000 test examples in each simulation.

- **LAC retains its original check:** mean coverage must stay within one
  percentage point of the 90% target.

- **APS and SOCOP are checked at both 80% and 90% targets.** Mean coverage must
  not fall more than one percentage point below either target. These deterministic
  scores can retain extra answers and exceed the target; higher coverage is not
  treated as a failure. The SOCOP simulation checks use only the central
  regularization of `0.25`; the additional values are compared in the trained
  example above.

- **Requesting higher coverage must not discard an answer.** For every simulated
  example, every answer in a method's 80% set must remain in its 90% set. This
  check applies separately to APS and SOCOP. Unchanged sets are allowed; they
  do not have to grow for every request.

All checks pass. The one-percentage-point allowance is a regression-test tolerance,
not a confidence interval. These controlled simulations do not establish the same
sampling conditions for BANKING77.

## Reproduce

From the repository root, use Python 3.12 or later:

```bash
python -m pip install -e ".[example,dev]"
python examples/synthetic_multiclass.py
python -m pytest tests/test_conformal.py tests/test_selection.py
```

The example prints the coverage target, the central SOCOP regularization and all
five configurations' metrics, and downloads no data. The results above were
reproduced with NumPy `2.4.6` and scikit-learn `1.9.0`.

Next, we will compare APS and SOCOP with the [BANKING77 baseline](banking77_validation.md),
keeping its classifier and data splits fixed. The relative performance observed
here may not carry over to real requests.
