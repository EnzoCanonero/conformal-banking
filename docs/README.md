# API usage and study reports

## API — use the decision layer

To apply the decision layer to your own model outputs, start with the
[usage guide](usage.md) and [runnable quickstart](../examples/api/quickstart.py).
They cover calibration, saved policies, routing and evaluation without BANKING77
data or a model backend. The API wraps the same functions used explicitly in
the study scripts.

For a cell-by-cell walkthrough, open the
[API mini notebook](../notebooks/api/01_quickstart.ipynb).

## Studies — follow the project's progression

The [project overview](../README.md) explains the goal: decide when a classifier
or LLM can route a request automatically and when it should defer to a person.
These reports follow the evidence from synthetic validation to BANKING77 and
the local LLM study.

1. **Do the conformal methods behave as expected on controlled data?**
   [Synthetic validation](synthetic/validation.md) checks coverage across
   repeated simulations and compares LAC, APS and SOCOP. Similar coverage does
   not imply similar automation: the methods retain different sets of answers.

2. **Can a simple text classifier support selective routing?**
   [TF-IDF and LAC baseline](banking77/baseline/tfidf_lac.md) establishes the
   first BANKING77 reference. LAC improves on one nearby naive confidence
   setting, but does not dominate the full automation/error trade-off.

3. **Does changing the conformal score improve that trade-off?**
   [LAC, APS and SOCOP](banking77/scores/lac_aps_socop.md) keeps the classifiers
   fixed. Tuned SOCOP offers useful high-coverage choices, while LAC and naive
   confidence remain competitive; deterministic APS is a poor singleton router
   here. The [APS](banking77/scores/aps.md) and
   [SOCOP](banking77/scores/socop.md) reports provide optional method-level detail.

4. **Does a better text representation help more than a different score?**
   [TF-IDF versus a frozen encoder](banking77/representations/tfidf_vs_encoder.md)
   finds substantially better routing trade-offs with the encoder for both LAC
   and SOCOP. Naive confidence remains competitive on automation and error.

5. **Can the same decision layer work with a local LLM?**
   [Selective routing with Qwen](banking77/llm/qwen.md) applies LAC, SOCOP and
   naive confidence to cached intent scores. Observed coverage meets the tested
   targets, but high coverage requires low automation and broad review sets;
   conformal shows no clear routing advantage in the region compared with naive.

6. **How does Qwen compare with the trained classifiers?**
   [Qwen versus TF-IDF and the encoder](banking77/llm/qwen_vs_classifiers.md)
   evaluates the systems on identical records. The frozen encoder gives the
   strongest automation/error trade-off in this setup, not a general verdict
   on all LLMs or training strategies.

## How to interpret the comparisons

- **Similar routing results do not mean equivalent statistical protection.**
  Conformal calibrates prediction sets to retain the correct answer at a chosen
  coverage level. Under exchangeability—for example, independent calibration
  and future requests from the same unchanged population—it provides a
  marginal-coverage guarantee that the fixed naive cutoff used here does not.
  This is not a guarantee on error among automated decisions. The reports
  discuss why BANKING77's official split requires care with these assumptions.

- **Keep each study's protocol with its results.** The historical LAC baseline
  uses more calibration examples than the later score comparison, which
  reserves separate examples for tuning. The score and representation studies
  average five splits; Qwen and its classifier comparison use one shared split.
  Their reported numbers are not interchangeable.

## Reports, execution guides and notebooks

- **Reports explain results and limitations.** Each report links to its
  reproduction commands; figures live alongside the study they describe.
- **Execution guides explain how to run the code.** Start with the
  [data guide](../data/README.md) and [examples index](../examples/README.md), then use the
  [representation guide](../examples/banking77/representations/README.md) or
  [LLM guide](../examples/banking77/llm/README.md) for those workflows.
- **Notebooks are exploratory walkthroughs.** Follow the
  [synthetic introduction](../notebooks/synthetic/01_lac_basics.ipynb),
  [TF-IDF routing example](../notebooks/banking77_tfidf/01_lac_routing.ipynb), or
  the LLM sequence:
  [scoring pilot](../notebooks/banking77_llm/01_scoring_pilot.ipynb) →
  [BANKING77 scoring](../notebooks/banking77_llm/02_banking77_scoring.ipynb) →
  [conformal routing](../notebooks/banking77_llm/03_conformal_routing.ipynb).
  These support understanding and development, rather than replacing the
  consolidated reports above.
