# Conformal Banking: From Classifiers to LLMs

[![Tests](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/EnzoCanonero/conformal-selective-prediction/actions/workflows/ci.yml)

## Conformal prediction for automation: from classifiers to LLMs

When should a model handle a customer request automatically, and when should a
person take over? This project investigates that question using **conformal
prediction**, a way to keep several possible answers instead of always forcing
the model to choose one.

We start with conventional classifiers trained to recognise banking-support
categories, then move towards **large language models (LLMs)** on the same task.
We compare how many requests each approach handles automatically and how often
those decisions are wrong, testing whether conformal prediction helps us decide
which requests to hand to a person.

### A shared workflow

Consider a customer writing, “The ATM kept my card.” The classifier and the
planned LLM will both follow these steps:

- **Use known answers to decide how strict the shortlist should be.** A model
  reporting “80% confidence” is not necessarily right 80% of the time, so we
  examine its predictions on a separate collection of requests whose correct
  answers are already known. Those examples show how strongly the model favours
  the correct categories and let us set a rule for which answers to retain on
  new requests. This is calibration. A longer shortlist can protect the correct
  answer while leaving us unable to choose a single category automatically.

- **Build a shortlist for each new request.** Rather than immediately accepting
  the model's favourite support category, conformal prediction keeps the
  categories that meet the rule learned during calibration. This shortlist of
  possible answers is called a prediction set.

- **Route the request or ask a person to review it.** We route automatically
  only when the shortlist contains one category. If several answers remain
  plausible, or none makes the list, a person reviews the request instead.

### What does the coverage guarantee give us?

- **Coverage asks whether the correct answer is still on the list.** For the
  ATM request, a list containing both “card retained by an ATM” and “failed cash
  withdrawal” counts as covered: the right answer is present, even though we
  still need a person to choose between the two. This is why coverage and the
  accuracy of automatic routing are different things.

- **A 90% target is a statistical promise, not a fixed result for every run.**
  The guarantee is that the correct answer is retained in at least nine out of
  ten cases on average, assuming calibration and future requests are drawn
  independently from the same unchanged population. That average includes
  repeating calibration with new examples. The particular examples used can
  make one run more or less cautious than another, so a run can fall below 90%
  even when it receives many requests.

- **Keeping the right answer available does not guarantee correct automation.**
  The promise is not about getting nine out of ten automatic decisions right,
  and some support categories can fare worse than others. We therefore measure
  automatic-routing errors and results for individual categories separately,
  rather than treating coverage as a general certificate of reliability.

- **The protection can change when the requests change.** If incoming requests
  shift from familiar card questions to unfamiliar fraud complaints, for
  example, the earlier calibration may no longer provide the same protection.

## BANKING77: one task, a shared benchmark

BANKING77 contains about 13,000 short English banking-support requests. Each
has one of 77 **intents**: the kind of help the customer needs, such as finding a
missing card or checking a pending transfer. The easily confused categories make
human review relevant, and the dataset is small enough for repeated experiments.

Both classifiers and the planned LLM will choose from **the same 77 categories**.
The LLM's job is classification, not writing customer replies or taking banking
actions. We compare them on the same official test requests, using separate
examples to calibrate each model. The [data guide](data/README.md) explains the
source, setup and limitations.

## Milestones

| Milestone | Status | Purpose |
|:----------|:-------|:--------|
| Synthetic validation | Complete | Check the method on generated data under controlled conditions. [Report](docs/synthetic_validation.md). |
| BANKING77 baseline | Complete | Establish what a simple word-based classifier can automate. [Report](docs/banking77_validation.md). |
| Representation and score comparisons | Next | Use knowledge learned from other text to help the classifier interpret requests; compare two ways of building the shortlist. |
| LLM comparison | Planned | Compare the same routing rule across classifiers and an LLM. |
| Changing conditions | Planned | Test whether changing instructions or incoming requests requires updating calibration. |

## First results on BANKING77

The first experiment uses a classifier that learns to recognise support
categories from the words in a request. Alongside the conformal shortlist rule,
we test a simpler **confidence rule**: accept the model's top answer when its
confidence score is above a chosen cutoff. Both rules use the same model.

At a **90% coverage target**, the conformal shortlist contains the correct answer
for **90.08% of test requests**. It routes **53.49% automatically**, of which
**5.51% are wrong**. These results are averages from five runs that change which
examples are used for training and calibration, while keeping the test requests
the same.

### How much automation, at what error rate?

We tested several configurations of both methods, varying **the coverage target
for the conformal method (LAC)** and **the confidence cutoff for the naive rule**.
Every configuration was evaluated in all five runs described above. Within each
run, both methods used the same classifier and test requests, so the comparison
concerns the routing rules rather than different models.

- **Our preferred high-automation trade-off is LAC with a 70% coverage target
  (`alpha=0.3`).** It retains almost all the automation achieved at the 80%
  coverage target, but with fewer mistakes among the requests handled
  automatically. This makes it the configuration we highlight from the tested
  high-automation choices.

- **The closest tested naive configuration uses a 20% confidence cutoff
  (`tau=0.2`).** Its average automation rate is the nearest to that LAC setting,
  making it the relevant comparison for asking which rule makes fewer mistakes
  while handling a similar amount of work.

The table shows the average results across the five runs:

| Rule and chosen setting | Requests handled automatically | Wrong answers among those handled |
|:------------------------|-------------------------------:|----------------------------------:|
| LAC, 70% coverage target (`alpha=0.3`) | 68.08% | 5.18% |
| Naive, 20% confidence cutoff (`tau=0.2`) | 65.06% | 6.46% |

At these settings, LAC handles **3.02 percentage points more requests** while
reducing the error rate among those handled by **1.28 percentage points**.
Both improvements hold in each of the five runs.

Other choices are possible. A stricter naive cutoff can achieve lower error by
automating less, while a requirement for 90% set coverage would rule out this
70% LAC configuration. The highlighted choice therefore reflects a favourable
high-automation balance in this experiment, not a setting that is best for every
requirement.

The curves below put this comparison in context by showing **all evaluated
configurations**, including choices that prioritise lower error over greater
automation.

![BANKING77 automation versus automated-case error, highlighting LAC alpha 0.3 in red across five splits](docs/figures/banking77/automation_vs_error.png)

- The **horizontal axis** shows the proportion of requests handled automatically,
  while the **vertical axis** shows the proportion of those decisions that are
  wrong. At the same automation rate, a lower point indicates fewer routing
  errors; at the same error rate, a point further right indicates more automation.

- The **blue curves** represent conformal selection (labelled “LAC”), and the
  **orange curves** represent the confidence rule (“Naive”). Each method has five
  curves, one per run. The points along a curve correspond to different settings
  applied to the same model and test requests within that run.

- The **red points** mark the 70% conformal coverage target used in the table.
  The “best trade off” label refers to the high-automation choice discussed
  above. It was highlighted after examining the results, not selected as a
  deployment policy.

Coverage also varies across support categories, and the calibration and test
data contain different proportions of intents. The
[BANKING77 report](docs/banking77_validation.md) discusses these limitations,
explains the curve shapes and presents the remaining results.

## Repository structure

```text
src/conformal_selective_prediction/   # Reusable methods, data loading and classifier
examples/
├── synthetic_multiclass.py           # Self-contained example
├── banking77_baseline.py             # One BANKING77 run, step by step
└── banking77_validation.py           # Compare results across several runs
tests/                               # Checks for the mathematics and routing rules
docs/                                # Reports and saved figures
data/README.md                       # Dataset setup and limitations
.github/workflows/ci.yml             # Automated checks
pyproject.toml                       # Package and tool configuration
```

Automated checks cover tests, code style and types. Downloaded data and generated
outputs stay local; the report figures are saved in Git.

## Installation and running

Use Python 3.12 or later. From the repository root, on macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[example,dev]"
```

This installs the examples and development tools. For the NumPy-only library,
use `python -m pip install -e .`.

Start with the synthetic example, which needs no downloaded data:

```bash
python examples/synthetic_multiclass.py
```

For BANKING77, first follow the [data setup instructions](data/README.md).
Then run the step-by-step example or the comparison across several runs:

```bash
python examples/banking77_baseline.py
python examples/banking77_validation.py
```

Results are saved to `outputs/banking77/`, replacing same-named files when rerun.
The reports explain how to reproduce them.

## License

Code is available under the [MIT License](LICENSE). BANKING77 has a separate
dataset license, documented in [data/README.md](data/README.md#source).
