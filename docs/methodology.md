# CEL methodology: evidence, sufficiency, error

The method in one sentence: **treat every extra LLM call as an experiment that yields
evidence about whether an answer is reliable, measure that evidence on one scale, spend
calls where they buy the most evidence, and stop when the evidence is sufficient.**
This document states the approach, the theory it rests on, and the hypotheses the
experiments must test. Implementation details live in `cel_k_allocation.md`,
`decisions.md` and the code; this is the part that should stay stable while the
pipeline changes.

## 0. Essence

**In one sentence:** every extra LLM call is a small experiment producing evidence about
whether an answer is right; we measure all evidence on one scale, buy the most
informative calls first, stop when the evidence is enough, and can state how often that
stopping rule is wrong, including when the data shift.

| # | Component | Intuitive idea | Theory behind it | Tested by |
|---|---|---|---|---|
| 1 | **Evidence on one scale** (§2) | Each call has a pass/fail outcome (a draw agrees, a negated question flips, an answer survives "are you sure?"). Right answers pass more often than wrong ones, so each outcome shifts our belief. | **Log-likelihood ratio**: λ(pass) = log(q₁/q₀), λ(fail) = log((1−q₁)/(1−q₀)). The most informative way to weigh an observation (Neyman–Pearson); these scores add up across calls, so every signal goes into one running **ledger** per case. | — |
| 2 | **Which calls to buy** (§3) | Some questions reveal more than others: a negated question exposes a model that follows surface cues; asking the same question again cannot. | **KL divergence** = the *expected* evidence per call. With default values a negation probe ≈ 2.5 draws and a distractor ≈ 0.5, so buy in order of evidence per call. | H1, H6 |
| 3 | **How many calls are enough** (§4) | Stop asking once you are sure enough; spend more on doubtful cases, less on clear ones. | **Sequential testing**: certify when the ledger reaches log(1/α); **Wald's formula** predicts the cost (~15 calls with draws only, ~12 with probes first). With a fixed budget, the same numbers give the **draw-equivalence**: what *k* calls are worth in repeated draws. | H2, H5 |
| 4 | **Calibration per dataset** (§4b) | Models fail differently on different kinds of questions, so "how often a wrong answer survives a probe" must be measured, not assumed. | Estimate q₁ (pass rate of correct answers) and q₀ (pass rate of wrong ones) on each dataset's small labelled **dev slice**, with a **conservative upper bound for q₀**, since the guarantee depends only on q₀ and q₁ only affects speed. Spending order, equivalence and cost are all derived from them. | H2 |
| 5 | **Error guarantee** (§5) | Each call is a bet against "this answer is wrong". If it really is wrong the bets are fair, so the money rarely grows 20-fold; certifying at 20× means ≤ 5% error. | The ledger is a **supermartingale** under the wrong-answer hypothesis. **Ville's inequality**: P(certify a wrong answer) ≤ α for *any* adaptive spending order or stopping time. At batch level, **e-values + e-BH** keep the certified set's error share ≤ δ under arbitrary dependence. | H4 |
| 6 | **Robustness to shift** (§5b) | If new data makes wrong answers pass a bit more often, the bets become slightly unfair, and the error grows by a computable amount. | **Rescaled supermartingale**: error ≤ α · Π over calls (1 + d·c), with *d* the size of the shift and *c* each signal's sensitivity; within 2α while d ≤ 0.046 on a 12-call path. Safeguards: theory anchors for q₀, label-free drift monitoring, online recalibration from spot-checks. | H7 |
| 7 | **Two ways to spend** (§6) | *CEL-k*: "about *k* calls per question; rank as well as possible". *Adaptive*: "certify the top-*k* at error ≤ δ; spend what it takes". | The same ledger under two constraints; the next call goes to the contender where it adds the most evidence per call. CEL-k is measured against self-consistency (and self-consistency + verbal); adaptive CEL carries guarantees 5–6. | H1, H3, H4, H5 |

**A method, not a pipeline:** nothing above is a fixed schedule. The order, the depth and
the stopping point follow from the estimated q's and the chosen error level; the old
"3 draws, then 6 more" design is one special case.

## 1. Problem

A batch of *n* questions (e.g. 50), each answered by a black-box LLM. We may spend a
limited number of extra calls, measured as the **expected calls per question**. Goal:
select the *k* answers (e.g. 10) most likely to be correct, with no labels at
deployment. The target is correctness of the answer, Z = 1{answer = truth}.

## 2. Evidence model

**Latent status.** Each case is either *reliable* (H1) or *unreliable* (H0). With the
per-dataset calibration of §4b these become *correct* and *incorrect*. Operationally:

| Status | Draws (fresh paraphrase) agree with the case's answer w.p. | A probe of type *g* passes w.p. |
|---|---|---|
| reliable, H1 | ≥ p₁ | ≥ q₁ᵍ |
| unreliable, H0 | ≤ p₀ | ≤ q₀ᵍ |

**Calls are experiments with a pass/fail outcome.**

| Call | Passes when | Where q₀ comes from |
|---|---|---|
| draw on a fresh paraphrase | it agrees with the case's answer | p₀ = 0.7 (design choice: "stability below 70% is unreliable") |
| negation probe | the answer flips as the negation requires | 0.5: a model following a surface cue passes at chance |
| relabel probe | the answer follows the swapped labels | 0.5, same argument |
| distractor probe | the answer survives irrelevant context | 0.8: weak cues survive distractors too |
| challenge, verbal confidence (thresholded), rationale agreement | answer holds / confidence ≥ τ / explanations agree | no theory: estimated conservatively on the dev slice (§4b) |

**Evidence of one call** is the log-likelihood ratio of its outcome:

  λ(pass) = log(q₁ / q₀),  λ(fail) = log((1 − q₁) / (1 − q₀))

(for draws, p in place of q). A case's **ledger** Λ is the sum of the λ's of every call
spent on it. All signals share one currency (nats), so calls of different kinds can be
compared, traded and added.

## 3. Value of a call

A call's value is how much evidence it yields **on average**, which is a
Kullback–Leibler divergence:

- for a reliable case, the expected gain is KL(q₁ ‖ q₀) per call;
- for an unreliable case, the expected loss is KL(q₀ ‖ q₁) per call.

At the defaults (q₁ = p₁ = 0.95):

| Call | +pass | −fail | expected gain, reliable | expected loss, unreliable | vs a draw |
|---|---|---|---|---|---|
| draw | +0.31 | −1.79 | 0.20 | 0.32 | 1.0 |
| negation / relabel | +0.64 | −2.30 | 0.50 | 0.83 | **2.5** |
| distractor | +0.17 | −1.39 | 0.09 | 0.14 | 0.5 |

This gives the **allocation principle**: buy calls in order of expected evidence per
unit cost. Negation and relabel first, then draws, distractor last. Probes at
temperature 0 are one-shot (asking again returns the same answer), so each probe type
is bought at most once; draws are repeatable.

## 4. Sufficiency: how much evidence is enough

**Per case.** A case is *sufficiently* evidenced when its ledger reaches A = log(1/α)
(certify) or falls to −A′ (eliminate). With α = 0.05, A = 3.0 nats.

**Expected cost to get there** (Wald's approximation): a reliable case needs about
A / (expected gain per call) calls.

- draws only: 3.0 / 0.20 ≈ **15 calls**;
- probes first: negation + relabel supply ~1.0 nats in 2 calls, and ~10 draws the
  remaining 2.0, ≈ **12 calls**; an unreliable case fails at least one of the two
  probes with probability ≥ 0.75, and one failure (−2.30) nearly eliminates it.

**Under a fixed budget** (CEL-k), the same numbers answer the reverse question: with
*k* calls, how much evidence can a case accumulate? The **draw-equivalence** of the *k*
calls chosen by the principle is

  r(k) = (Σ expected gains of the chosen calls) / (expected gain of one draw).

It is **specific to the dataset and model**: it depends on their q's (§4b). With the
default q's, 1 draw + negation + relabel (k = 3) gives r ≈ 5.9 draws, but that number
is illustrative only. Self-consistency draws are also correlated (same wording; design
effect of Lemma 2), so their effective number is below the nominal count.

## 4b. Calibration: the per-dataset procedure

The q's are not universal constants: how often a wrong answer survives a negation probe
depends on the model and the kind of question. So for each dataset × model the method
**estimates them, then derives everything else from them**:

1. **Estimate** (q₀ᵍ, q₁ᵍ) for every signal *g* on that cell's labelled dev slice, with
   H1 = the answer is correct and H0 = it is incorrect:
   q̂₁ᵍ = pass rate among correct answers, and q₀ᵍ = an **upper confidence bound** on
   the pass rate among incorrect ones (conservative, because validity depends on q₀).
   Theory values (§2) serve as the prior when the dev slice is small.
2. **Derive** per signal the evidence functions λᵍ, the expected evidence per call
   (KL), and hence the spending order for this cell.
3. **Predict** the draw-equivalence r̂(k) for each budget *k*, and the calls needed to
   certify (§4).
4. **Freeze** all of it before any evaluation item is scored.

This is the method's deliverable for a new dataset: *given a small labelled dev slice,
it tells you which signals to buy, in what order, how much confidence a budget of k
calls buys, and what that is worth in self-consistency draws.*

Defining H0/H1 by correctness makes the guarantee (§5) speak about correctness
directly: an incorrect answer is certified with probability ≤ α. It rests on one
transfer assumption: incorrect answers in deployment pass each signal no more often
than the dev slice's upper bound q₀.

## 5. Error: what the evidence guarantees

**Betting view.** Start each case with $1 and multiply by e^λ after each call. If the
case is unreliable and the q₀'s are not understated, every call is a fair or losing bet
(E[e^λ] ≤ 1). Certify a case when its money reaches 1/α.

**Proposition (anytime-valid certification).** If, for every call, the pass
probability of an unreliable case (given everything observed so far) is at most its q₀,
then for **any** rule choosing which case gets which call and when to stop:

1. *per case:* P(an unreliable case is certified) ≤ α (Ville's inequality);
2. *per batch:* certifying the cases selected by e-BH at level δ, applied to the final
   ledgers e^Λ, keeps E[unreliable certified / certified] ≤ δ.

*Proof sketch.* Under H0 each call multiplies the ledger by a factor with conditional
mean ≤ 1, so e^Λ is a nonnegative supermartingale in the batch's filtration; Ville's
inequality gives (1), and optional stopping makes each final e^Λ an e-value, so e-BH
(Wang & Ramdas, 2022) gives (2) under arbitrary dependence.

**What it needs, and what it does not.**
- Validity needs only the q₀'s (and p₀) not to be understated. q₁ affects how fast
  reliable cases certify (power), never validity. So every signal enters safely with q₀
  set to a dev-slice upper confidence bound (§4b).
- With H0/H1 calibrated as incorrect/correct (§4b), it is a guarantee about
  **correctness**, under the transfer assumption that deployment's incorrect answers
  pass no more often than the dev bound. With the theory defaults instead, it is about
  stability, and linking stability to correctness is Assumption 1, tested separately.
- It does not need independence between calls: only the conditional bound on q₀.

**Label-free error estimate.** For a delivered set, δ bounds the expected share of
unreliable certified cases. Without certification (small budgets), the ledger still
orders cases; its quality is measured, not guaranteed.

## 5b. Error under distribution shift

The calibrated guarantee assumes deployment looks like the dev slice: incorrect answers
pass each signal no more often than the dev bound q₀. When the data shift, the same
betting argument says **how much the error can grow**, as a function of the shift size.

**Sensitivity of a call to shift.** If an incorrect answer's true pass rate exceeds the
assumed q₀ᵍ by d_g, the call's expected payoff becomes 1 + d_g·c_g, with

  c_g = q₁ᵍ / q₀ᵍ − (1 − q₁ᵍ) / (1 − q₀ᵍ).

At the defaults: draw c = 1.19, negation/relabel c = 1.80, distractor c = 0.94.

**Proposition (robustness to shift).** If incorrect answers pass each signal *g* at most
d_g more often than its assumed q₀ᵍ, then for any adaptive allocation:

1. *per case:* P(an incorrect answer is certified) ≤ α · Π over its calls (1 + d_g c_g);
2. *per batch:* the certified set's expected error share ≤ δ · the largest such product
   over the certified cases' paths (bounded, since each case's calls are capped).

*Proof sketch.* Dividing the ledger e^Λ by the running product of (1 + d_g c_g)
restores a nonnegative supermartingale under H0, so Ville's inequality applies to the
rescaled ledger; the certification event then has probability at most α times the
product along the path. The same rescaling bounds the e-values' means for e-BH.

With no shift the bound is the original α. At the defaults, for a case certified along a
typical 12-call path (2 probes + 10 draws) with α = 5%:

| Shift d (extra pass rate of wrong answers) | Error bound |
|---|---|
| 0 | 5.0% |
| 0.02 | 6.8% |
| 0.05 | 10.6% |
| 0.10 | 21.4% |

The bound stays within 2α while d ≤ 0.046. **Long paths amplify shift**, because the
factor compounds per call: fewer, stronger calls are less exposed in total, one more
reason to prefer probes over many draws.

**Three safeguards.**

1. **Theory anchors.** For negation and relabel, q₀ = 0.5 comes from an argument (a
   cue-following model passes at chance), not from data. Using
   q₀ = max(theory value, dev upper bound) keeps them robust to dataset shift.
2. **Label-free shift monitoring.** Compare each deployment batch's pass-rate profile
   with the dev slice's (e.g. how often unanimous cases still fail a negation probe). A
   drift larger than the tolerated d is a warning before any error is observed.
3. **Online recalibration.** The q₀ used for a call may be updated from human
   spot-checks of delivered answers gathered *before* that call (the paper's deployment
   loop); the
   guarantee stays valid because each q₀ is fixed before its call is made.

## 6. Two ways to spend

| | CEL-k (fixed budget) | Adaptive CEL (target error) |
|---|---|---|
| Constraint | ≈ *k* expected calls per case | certified set's error ≤ δ |
| Rule | spend the batch's *k·n* calls where they add most evidence to the top-*k* frontier | same greedy rule; stop when e-BH certifies *k* cases, or the cap is reached |
| Output | a ranking by ledger | a certified top-*k* (or fewer, flagged) |
| Guarantee | none; quality measured | Proposition, §5 |
| Cost | fixed | adapts per case and per dataset |

**Greedy rule.** Give the next call to the case and call type with the largest expected
evidence per call, among cases that can still change the selected set (contenders near
the certify boundary or the top-*k* cut). Cases below −A′ stop.

## 7. What is set where

| Quantity | Source |
|---|---|
| α, δ | design choices |
| q₀, q₁ for every signal (draws included) | per dataset × model, on that cell's dev slice (§4b); theory values (§2) as the prior |
| spending order, r̂(k), predicted calls to certify | derived from the q's (§3, §4) |

Nothing is fitted on the evaluation items.

## 8. Hypotheses and experiments

| # | Hypothesis | Test | If the data disagree |
|---|---|---|---|
| **H1** value | At equal expected calls, CEL ranks better than self-consistency, most of all at 2–3 calls. | prec@10 and AUROC over batches vs self-consistency at the same expected calls; paired over batches. | Revisit the evidence model (§2); report where probes don't help. |
| **H2** calibration | For each dataset × model, the draw-equivalence r̂(k) predicted from the dev slice (§4b) matches the one measured on eval. | Per cell: measured equivalence (the self-consistency budget with equal prec@10) vs r̂(k); also whether the dev-derived spending order is the best order on eval. | A consistent gap means the evidence model mis-describes that dataset (e.g. signals are correlated); revise §2 for it. |
| **H3** control | The gain survives giving self-consistency the free per-draw verbal confidence. | CEL vs self-consistency + verbal at equal calls. | If the control ties, confidence elicitation is the finding; probes are secondary. |
| **H4** validity | Among certified cases, the realised share that are incorrect is ≤ δ (with no shift). | On labelled eval items: false-certification rate; probe pass rates of incorrect answers vs the assumed q₀. | Raise q₀ (dev upper bounds) until validity holds; report the cost in calls. |
| **H5** sufficiency | Calls to certify match the Wald predictions; adaptive CEL spends less on easy datasets. | Mean calls per certified case vs A / KL; spend per dataset. | Recalibrate q₁ (power), not q₀. |
| **H6** mechanism | Probes catch confidently wrong answers that resampling cannot. | Among unanimous-but-wrong cases (TriviaQA): share caught by negation/relabel; worked examples. | Report honestly; the budget claim may rest on H1 alone. |
| **H7** shift | Under a real shift, the realised false-certification rate stays within the shift-adjusted bound (§5b). | Calibrate on one dataset's dev slice, evaluate on another (e.g. TriviaQA → StrategyQA); measure the shift d per signal from labels; compare realised error with α·Π(1 + d c). Also: does label-free monitoring flag the shift? | If the realised error exceeds the bound, calls are not conditionally bounded as assumed (e.g. correlated failures); tighten the model. |

**First test: the smoke run** (TriviaQA × gpt-4.1, 60 questions). It is too small for
significance but large enough to see whether H1–H3 point the right way, whether the
assumed q₀'s look plausible (H4's second test) and whether cost per certified case is
near the prediction (H5). It is the checkpoint at which this methodology is revised.

## 9. Relation to the current code

- **Exists:** the λ functions (`sprt_posterior.probe_llr`, `SPRT.up/down`), the ledger
  (`EvidenceLedger`), CEL-k with dev-chosen allocations, the matched-budget comparisons
  and the self-consistency + verbal control, the simulator.
- **To build:** the per-cell calibration of (q₀, q₁) on the dev slice (§4b), with the
  derived λ functions, spending order and r̂(k); the KL-based greedy order; the
  shift-adjusted bound, label-free shift monitoring and the cross-dataset test (H7); batch-level adaptive CEL with e-BH
  certification; the draw-equivalence measurement (H2); the validity checks (H4).
- **To retire or demote:** the fixed 3 + 6 schedule becomes one special case of the
  greedy rule; Shapley valuation and the evidence arms become supporting analyses.
