# BANKING77: selective routing with Qwen

[All reports](../../README.md)

**Qwen reaches the requested prediction-set coverage, but high coverage comes
with little automation and broad sets for review.** Its top intent is correct
on 61.04% of test requests. LAC and SOCOP can retain alternatives when that top
answer is wrong, but do not show a clear automation/error advantage over naive
confidence in the regions compared. That does not make the methods equivalent:
conformal's distinct statistical benefit is calibrated prediction-set coverage,
not a promise of better automatic decisions.

This report studies Qwen on its own. The separate
[classifier comparison](qwen_vs_classifiers.md) evaluates the same routing rules with
TF-IDF and a frozen text encoder.

## Model, scores and evaluation

- **One local LLM, with no task-specific training.** We use
  `Qwen3-4B-Instruct-2507`, quantized to 4 bits, through MLX on an M1 with 16 GB.
  A fixed prompt lists BANKING77's 77 intent names. Instead of generating a
  reply, the model scores each name as a possible answer. We sum its token
  log-probabilities, including the end-of-response token, then normalize across
  candidates. These weights express relative preference, not verified chances
  of being correct. The pinned revision and prompt are in the
  [running guide](../../../examples/banking77_llm/README.md#fixed-setup).
- **Separate tuning, calibration and test requests.** Seed 42 gives two tuning
  halves of 625 and 1,251 final-calibration requests from the official training
  data. Evaluation uses all 3,080 official test requests. This is one fixed
  split, not an average across repeated runs.
- **Three decision rules, using the same saved scores.** LAC keeps intents
  above a calibrated weight cutoff. SOCOP favors one-label sets, with lambda
  controlling the cost of larger sets; lambda is selected for each target on
  tuning A/B, before final calibration. Both automate only one-label sets.
  Naive confidence instead accepts the top answer above a fixed cutoff. The
  evaluation grids are fixed; test errors do not select a deployment policy.

**Coverage** is the fraction of requests whose set contains the correct intent,
including requests sent to review. **Automated-case error** is the fraction of
automatic decisions that are wrong. A 95% coverage target does not promise
that automatic decisions have at most 5% error.

## How much automation, at what error rate?

![Qwen routing: LAC, tuned SOCOP and naive confidence](figures/qwen_automation_vs_error.png)

The horizontal axis shows automation; the vertical axis shows errors among
automated requests. Blue is LAC, orange is tuned SOCOP and green is naive
confidence. Points are evaluated settings on the same requests; connecting
lines are not an estimated optimum or a confidence band.

- **LAC reduces the error rate by leaving most requests for review.** At the 90%
  target it automates 10.03% with 3.88% error. SOCOP at the same target raises
  automation to 26.23%, but error also rises to 10.77%. More singletons do not
  necessarily mean more reliable automatic decisions.
- **SOCOP's 95% and 99% settings serve different goals.**

  - At **95%**, it automates 13.99% with 5.57% error: 24 mistakes among 431
    decisions. This retains more automation than the stricter setting, but
    still sends roughly six out of seven requests to review.
  - At **99%**, automation falls to 2.18%, with one mistake among 67 decisions
    (1.49%). Its 95% Wilson error interval is 0.26%–7.98%: the small observed
    error is not evidence of an error ceiling below 2%. LAC at this target
    automates nothing, so its automated error is undefined, not zero.

- **Naive confidence follows a similar trade-off where the grids overlap.**
  At cutoff `0.999`, it automates 71.30% with 26.64% error; SOCOP's 70% coverage
  target gives 73.05% and 27.33%. Neither point improves both measures. The
  tested naive grid has no nonzero-automation point below 46.43%, so the plot
  does not establish equivalence in the low-automation region.
- **The turnback is a change in which requests are deferred.** Relaxing
  coverage initially creates more singletons, then more empty sets. At the
  50% target, LAC and SOCOP both leave 28.44% of requests with an empty set;
  automation falls and their aggregate routing metrics coincide. A smaller
  set is not always an automated answer.

## Coverage follows the requested target

![Qwen prediction-set coverage versus target for LAC and tuned SOCOP](figures/qwen_coverage_vs_target.png)

The horizontal axis shows the requested coverage and the vertical axis the
coverage observed on the test requests. The diagonal marks equality; vertical
bars are pointwise, approximate 95% Wilson intervals. Blue is LAC and orange is
tuned SOCOP, as in the automation plot.

- **Both methods meet or exceed every tested target.** Their coverage curves
  remain close, despite their different automation rates. The intervals describe
  this fixed evaluation, not variation from new calibration splits; they do not
  establish the assumptions of the conformal guarantee.
- **High coverage is not high classification accuracy.** Calibration retains
  the true answer by admitting alternatives with very low weights. For LAC,
  that requires a low inclusion cutoff and broad sets; the top answer does
  not improve.

## Higher coverage means broader sets for review

![Qwen average prediction-set size versus observed coverage for LAC and tuned SOCOP](figures/qwen_set_size_vs_coverage.png)

Here the horizontal axis is observed coverage and the vertical axis is average
set size across all test requests, including singletons and empty sets. Naive
confidence is absent from both plots because it does not construct calibrated
prediction sets.

- **The cost of high coverage rises sharply.** At 99%, retaining roughly 49
  of 77 intents barely narrows the reviewer's choices. At the other extreme,
  the mean falls below one at the 50% target because 28.44% of sets are empty,
  not because every request receives a single answer.
- **SOCOP trades compactness for more singletons.** Tuning selects the smallest
  tested lambda, `0.001`, at every target; extra labels are relatively cheap
  once a set expands beyond one. At 95%, deferred sets average 27.31 intents,
  versus LAC's 22.13. This is a grid endpoint, not a demonstrated global optimum.

The table separates the overall mean from the mean for deferred requests.
Deferred means include empty sets, although none occur at the settings below.

| Method and target | Coverage (95% Wilson interval) | Mean set size | Mean deferred-set size |
|:--|--:|--:|--:|
| LAC, 90% | 91.56% (90.52%–92.49%) | 12.33 | 13.60 |
| SOCOP, 90% | 91.75% (90.73%–92.67%) | 14.68 | 19.54 |
| LAC, 95% | 95.42% (94.63%–96.11%) | 21.52 | 22.13 |
| SOCOP, 95% | 95.52% (94.73%–96.20%) | 23.63 | 27.31 |
| LAC, 99% | 99.16% (98.77%–99.42%) | 49.02 | 49.02 |
| SOCOP, 99% | 99.09% (98.69%–99.37%) | 48.52 | 49.58 |

## Conclusions and limits

- **Naive confidence remains competitive on automation and error.** Where
  the tested grids overlap, neither LAC nor SOCOP establishes a clear routing
  advantage over naive confidence. The naive grid has no nonzero-automation
  point below 46.43%, so this finding does not extend to the low-automation
  region.
- **SOCOP buys more automation than LAC at the cost of more errors.** At the
  95% coverage target, SOCOP automates 13.99% with 5.57% error, versus LAC's
  2.92% with 1.11% error. Neither dominates both measures. High coverage still
  leaves most requests for review, with broad sets: the calibration layer does
  not repair the underlying classifier.
- **Conformal adds a coverage promise that the fixed naive cutoff does not.**
  Separate labelled examples calibrate which intents to retain, without
  requiring the model's weights to be accurate probabilities. The resulting
  prediction sets have a finite-sample marginal-coverage guarantee when
  calibration and future requests are exchangeable, as with independent draws
  from the same unchanged population. This is the [statistical benefit of conformal prediction](https://arxiv.org/abs/2107.07511),
  even when its routing curve resembles naive confidence. The promise includes
  deferred requests; it does not bound errors among automatic decisions or
  ensure coverage for every intent.
- **Concentrated weights are not dependable probabilities of correctness.**
  Even naive cutoff `0.999999` leaves 16.92% observed error among accepted
  requests. Label wording, unequal token lengths and the scoring rule may
  matter, but we have not isolated their effects or established model size or
  quantization as the cause. Improving these scores requires a separate
  development study, not a more favorable test-selected setting.

The [classifier comparison](qwen_vs_classifiers.md) puts these results alongside TF-IDF
and the frozen encoder. The next planned experiment changes the prompt while
keeping the original calibration threshold, then recalibrates on separate
examples to examine whether coverage persists.

## Runtime and reproduction

Scoring all **5,581 requests** took **12 hours 47 minutes** of recorded model
scoring on the M1 with 16 GB. The expansion reused 1,400 requests and added 4,181
in **9 hours 31 minutes**. These times exclude model loading and checkpoint
writes, so they are not end-to-end benchmarks. With the complete score cache,
reproduce the Qwen metrics without inference:

```bash
python -m examples.banking77_llm.lac
python -m examples.banking77_llm.naive
python -m examples.banking77_llm.socop
```

The [running guide](../../../examples/banking77_llm/README.md) covers preparation
and outputs under `outputs/banking77/llm/full/`. The figures are snapshots from
the shared comparison script, which also includes the classifier study's
existing naive cutoffs. Refresh them together with the reported numbers.
Notebook and earlier pilot results are not pooled into this report.
