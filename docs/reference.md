# Data splits: why not 7:1:2, and references

Status: notes, 2026-10-11. References are from memory: check titles, venues and details
before citing them in the paper. Implemented: `split.n_cal` in the configs
(`data/splits.py`) and the conformal step in `certified_selection.py` (decisions.md #17).

## Why the conventional ratio does not apply

| | Conventional ML | CEL |
|---|---|---|
| What is fitted | millions of parameters, so most labels go to training | a score with a handful of parameters (pass rates q₀/q₁, thresholds), so it needs few labels |
| What sets the size of each set | fitting quality | fitting needs ~100; the guarantee and the paper's tests need more |
| Calibration set | usually none | sets the error floor: about m / (K·(M_cal + 1)) |
| Eval set | ~20%, enough for a stable average | sized for power: 10 batches of 50 for the paired confirmatory test |

So the split is roughly inverted: the smallest set fits, a larger one calibrates, the
largest evaluates. Proposed per dataset × model cell: **~100 fit : ≥200 calibration : 500
eval** (about 1:2:5).

| Set | Job | Size and reason |
|---|---|---|
| dev / fitting (M_fit) | shape the score: prompts, thresholds, q₀/q₁ estimates (methodology.md §4b), CEL-k allocation | ~100: enough for q̂₀ with a conservative upper bound |
| calibration (M_cal) | measure the score's error rate; untouched by any tuning | ≥200: floor 50/(10·201) ≈ 0.025 at m = 50, K = 10 |
| eval | report results; labels read only by metrics | 500 = 10 batches of 50, for the confirmatory test |

Calibration price at m = 50, K = 10 (exact, Cor. 2):

| M_cal | 10 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| smallest guaranteed error | 0.45 | 0.098 | 0.050 | 0.025 | 0.012 |

## What the calibration set is for

The score ranks cases; the calibration set turns the ranking into a guarantee. For a test
case with score s, the conformal p-value is (1 + #wrong calibration cases scoring ≥ s) /
(M_cal + 1). BH on these p-values at level α delivers a set with E[FDP] ≤ α, for any
score, if calibration and test cases are exchangeable. A weak score costs deliveries,
not errors. The calibration set must be disjoint from the data used to fit the score;
otherwise the check is biased toward the score.

## Label budget

300 labelled cases per cell outside eval (100 now). TriviaQA (9,960 unique questions):
no issue. StrategyQA (1,603 rows): fits with 800 to spare, but uses more of the pool.

## References

1. **Jin & Candès (2023)**, "Selection by Prediction with Conformal p-values", JMLR.
   The procedure Cor. 2 uses (conformal p-values + BH); FDR holds for any calibration
   size, power grows with it.
2. **Gui, Jin & Candès (2024)**, "Conformal Alignment: Knowing When to Trust Foundation
   Models with Guarantees", NeurIPS. Closest setting: choosing which LLM outputs to trust
   with FDR control; splits a labelled reference set into score-training and calibration
   parts and reports how results change with each part's size. Main anchor for the split.
3. **Angelopoulos & Bates**, "A Gentle Introduction to Conformal Prediction and
   Distribution-Free Uncertainty Quantification". Calibration-set size: the guarantee
   holds at any size but varies run to run with small sets; suggests ~1,000 points for
   tight coverage. For FDR selection the relevant floor is 1/(M_cal + 1).
4. **Bates, Candès, Lei, Romano & Sesia (2023)**, "Testing for Outliers with Conformal
   p-values", Annals of Statistics. BH stays valid on conformal p-values that share one
   calibration set (PRDS), which is how Cor. 2 uses them.
5. **Vovk (2012)**, "Conditional Validity of Inductive Conformal Predictors", ACML;
   **Lei et al. (2018)**, "Distribution-Free Predictive Inference for Regression", JASA.
   Standard references for split conformal and the fit/calibration split.
6. **Mohri & Hashimoto (2024)**, "Language Models with Conformal Factuality Guarantees",
   ICML; **Quach et al. (2024)**, "Conformal Language Modeling", ICLR. LLM uses of a
   calibration set of a few hundred labelled examples.

## For the paper

State the sizes per cell with the reason for each (fit ~100 for q̂₀ with an upper bound;
calibration ≥200 for a 0.025 floor; eval 500 for the confirmatory test) and cite
Jin & Candès (2023) and Gui et al. (2024) for the split.
