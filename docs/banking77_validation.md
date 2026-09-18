# BANKING77: the LAC baseline

At the 90% LAC target, the TF-IDF/logistic-regression baseline reaches 90.08%
mean coverage across five splits. Singleton selection automates 53.49% of
requests, with 5.51% error among automated cases. Aggregate coverage is stable,
but intent-level differences prevent describing the policy as uniformly reliable.

This is the historical baseline, with **2,501 calibration examples per run**.
The later [score comparison](banking77/score_comparison/comparison.md) reserves
some of those records for SOCOP tuning and recalibrates all methods on 1,251
examples. Its fresh LAC results are therefore slightly different; the numbers
and figures in this report retain the original experiment.

## Experiment setup

The [validation script](../examples/banking77_validation.py) uses the official
[BANKING77 files](../data/README.md), pinned to revision
`57ec275d8078af65b7731c2a98be812d844a6d6b`:

- Split the official training data into 7,502 model-training and 2,501 calibration
  examples, stratified by intent.
- Preserve all 3,080 official test examples, with 40 examples for each of 77
  intents. Every run evaluates the same test records.
- Use seeds `7, 21, 42, 84, 123` to vary training and calibration partitions.
- Fit word TF-IDF unigrams and bigrams followed by logistic regression. Both
  preprocessing and classifier fitting use only the training partition; the
  [script](../examples/banking77_validation.py) records the fixed model settings.
- Fit once per seed and reuse predictions across policy settings, without
  choosing a deployment threshold from test performance.

The LAC grid is `alpha = 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50`. The naive
confidence grid runs from `0.0` through `1.0` in steps of `0.1`. Both were fixed
before evaluation.

### What the two policies do

- **LAC** uses calibration labels to set a class-probability cutoff. Exactly
  one surviving intent means automatic routing; zero or several means review.
  A 90% coverage target concerns retaining the true intent in the set, not
  correctness among automated decisions.
- **Naive thresholding** routes the top answer when its probability reaches
  `tau`, without checking the second-best intent. A cutoff of `0.8` does not
  guarantee an automated-case error below 20%.

## Results at the reference settings

The reference settings, `alpha=0.1` and `tau=0.5`, come from the
[single-split example](../examples/banking77_baseline.py), not a search for the
best test results. Classifier accuracy averages 84.01% (83.83–84.29%).

Tables report arithmetic means across five splits, with minimum–maximum ranges.
Ranges describe split variation, **not confidence intervals**; error rates are
averaged per split rather than pooled across repeated test observations.

| Metric | LAC, alpha = 0.1 | Naive, tau = 0.5 |
|:-------|----------------:|----------------:|
| Prediction-set coverage | 90.08% (89.51–90.62%) | Not applicable |
| Average prediction-set size | 1.568 (1.520–1.614) | Not applicable |
| Automation rate | 53.49% (52.31–55.19%) | 24.49% (24.03–24.81%) |
| Error among automated cases | 5.51% (5.04–5.77%) | 1.33% (1.06–1.71%) |

LAC handles more requests here, but with more error among those handled. These
are different operating points, not a matched-automation comparison or a
deployment recommendation.

## What changes as the rule becomes stricter?

### 1. Nominal versus empirical coverage

![LAC nominal versus empirical coverage for five splits, with pointwise 95% Wilson intervals](figures/banking77/nominal_vs_empirical_coverage.png)

The horizontal axis is target coverage, `1 - alpha`; the vertical axis is the
fraction of requests whose set contains the true intent. Each line is one split;
the diagonal marks agreement and the bars are pointwise 95% Wilson intervals.
Coverage follows the target across the grid. All five intervals contain 90%
at that target, supporting measured agreement without proving that calibration
and test requests follow the same distribution.

### 2. Coverage versus average set size

![Empirical coverage versus average LAC prediction-set size for five splits](figures/banking77/coverage_vs_set_size.png)

Mean set size is on the horizontal axis and coverage on the vertical axis,
with one point per alpha on each split's line. Mean size rises from 1.568 at
the 90% target to 2.455 at 95% and 8.510 at 99%. More labels help retain the correct intent,
but multi-label sets require review. Smaller average sets are not sufficient
either: empty sets reduce the average while also requiring review.

### 3. Automation versus error

![Automation rate versus error on automated cases for LAC and naive thresholding, highlighting LAC alpha 0.3 in red across five splits](figures/banking77/automation_vs_error.png)

Automation is horizontal; error among automated requests is vertical. For example,
`(0.60, 0.05)` means 600 automated requests per 1,000, including 30 mistakes.
Blue paths are LAC; orange paths are naive; each method has five split traces,
not confidence bands. Red circles mark LAC `alpha=0.3`.

Paths follow tested parameter order, not an interpolated optimal frontier.
Raising the naive cutoff reduces automation. LAC can turn back: shrinking a
multi-label set can produce a singleton, then an empty set. At the strict 99%
target, LAC automates only 6.03% with no observed errors, based on just 170–207
selected requests per run. Zero observed error is not zero risk. Settings with
no automated requests have undefined error and are omitted.

#### Where LAC is favorable

The clearest local advantage is around **65–68% automation**, comparing the
actual tested LAC `alpha=0.3` and naive `tau=0.2` settings:

| Evaluated setting | Automation rate | Error among automated cases |
|:------------------|----------------:|----------------------------:|
| LAC, alpha = 0.3 (red) | 68.08% (67.73–68.44%) | 5.18% (4.93–5.51%) |
| Naive, tau = 0.2 | 65.06% (64.64–65.29%) | 6.46% (6.36–6.58%) |

LAC handles **3.02 percentage points more requests with 1.28 percentage points
less automated-case error**. Both improvements hold in every split, but do not
establish a general winner or statistical significance.

The 70% target retains almost all the automation of the 80% target (68.08%
versus 68.25%), with lower error (5.18% versus 5.91%). A stricter naive cutoff
can instead reduce error at the cost of automation, as the reference table shows.

The red setting targets 70% coverage and achieves 70.86%: it is **not eligible
if 90% set coverage is required**. Despite its "best trade off" label, the
highlight was chosen after test inspection. Deployment selection needs an error
budget, review capacity and separate validation, not a winner chosen from this plot.

### 4. Prediction-set-size distribution

![Mean per-split LAC set-size fractions for alpha 0.01, 0.05, 0.1 and 0.3, including empty sets](figures/banking77/set_size_distribution.png)

The horizontal axis counts labels; the vertical axis is the mean fraction of
sets at each size. Curves show `alpha=0.01, 0.05, 0.1, 0.3`, not individual
splits. Only size one is automated. At `alpha=0.1`, 53.49% are singletons and
4.72% empty; at `alpha=0.3`, these become 68.08% and 25.31%. The growing empty
bin explains why shrinking sets eventually reduces automation.

## Intent-level diagnostics

Aggregate coverage hides substantial differences between intents. The three
lowest mean intent coverages at `alpha=0.1` are:

| Intent | Mean coverage | Split range |
|:-------|--------------:|------------:|
| `contactless_not_working` | 53.00% | 50.00–60.00% |
| `card_acceptance` | 60.00% | 55.00–67.50% |
| `card_swallowed` | 72.50% | 70.00–77.50% |

These post-hoc diagnostics reveal an important weakness: nominal marginal
coverage does not protect every intent. Each intent has only 40 unique test
examples, not 200 independent observations across seeds. Full results are in
`class_coverage.csv`; these findings should not be used to tune on the test set.

## Uncertainty and limitations

Pointwise 95% [Wilson intervals](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm)
use 3,080 cases for coverage, selected cases for automated error, and 40 for each
intent. They are not simultaneous guarantees across settings. Repeated test
observations are never pooled across seeds.

The [official split limitations](../data/README.md#split-limitations) matter:
training intent counts range from 35 to 187, but test counts are exactly 40.
Calibration inherits the unequal training mixture, so exchangeability with
test data is not established and global intervals are approximate diagnostics.
Six texts also overlap across official splits after lowercasing and trimming,
though none overlap exactly. Original records remain unchanged.

## Conclusion

LAC provides a useful, reproducible automation trade-off, including a local
advantage over naive `tau=0.2`. It neither guarantees automated-case error nor
resolves weak intent coverage. The later score comparison changes how prediction
sets are built; a frozen-encoder comparison remains ahead of the LLM phase. This baseline
alone does not complete the broader acceptance gate.

## Reproduce the report

Numerical results come from commit `3ed6f5f`, using NumPy `2.4.6`, scikit-learn
`1.9.0` and Matplotlib `3.11.1`; the red highlight was added later without changing
the data. With Python 3.12 and the [pinned data](../data/README.md#download), run
from the repository root:

```bash
python -m pip install -e ".[example]"
python examples/banking77_validation.py
```

The shorter `python examples/banking77_baseline.py` walkthrough uses seed 42,
already included here. Validation replaces generated files in
`outputs/banking77/`, but not the four reviewed snapshots in
`docs/figures/banking77/`. Update snapshots and reported numbers together only
after reviewing a new run.
