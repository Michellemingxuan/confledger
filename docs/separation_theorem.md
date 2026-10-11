# Separation theorem: why resampling has a floor and probes do not

Status: **draft, proposed 2026-10-08** (decisions.md #16). Not yet in the paper or the code.
It is meant to be the paper's headline theoretical result; Lemma 4 (power separation) becomes
its test-level special case.

## Current form (odds), 2026-10-08

Formal version: `docs/paper/sec_separation.tex`. One assumption (S): in the stable pool,
cases are i.i.d., draws are independent of the type, and probes are independent of the
draws and of each other given the type.

**Theorem (evidence odds).**  odds(wrong | draws, probes) = Σ_{t wrong} (ρ_t/ρ_C) · Π_j λ_{t,j}(b_j),
with λ_{t,j}(b) = P(probe j gives b | type t) / P(probe j gives b | correct). The draws do not
appear: the ledger of log-likelihood ratios is the log-odds of error.

**Corollary 1 (neutral, geometric, floored).** For a case that passes r probes:
(ρ_B/ρ_C)·λ_Bʳ ≤ O_r ≤ O₀·max_t λ_tʳ, and O₀ for r = 0 at any number of draws. So any
draw-only rule has E[FDP] = O₀/(1 + O₀) = ρ.

**Corollary 2 (price of a guarantee).** E[α*_K | t] ≤ m·G(t)/K + m/(K(M_cal + 1)).

**Remark.** r* = ln(a·ln(1/λ_F) / (b·ln λ_B)) / ln(λ_B/λ_F); at the illustration values
r* = 5.2, FDP = 0.081 against 0.20 for any number of draws.

The sections below are the earlier type-by-type form; they remain valid as derivations
(ρ_A is the odds identity at b = all pass).

## Claim in one sentence

Any selection rule that sees only resampled answers, self-consistency included and with any
number of draws, delivers a top-k whose expected share of wrong answers is at least the
share of stable-but-wrong cases; a few label-action probes push the share below that floor,
at a cost of one call per probe.

## Setting

A batch contains a **stable pool** of cases (the cases from which a top-k is drawn when the
pool is large). Case i has a stability p̄ᵢ (Lemma 1), a latent type Tᵢ, a correctness
Zᵢ ∈ {0, 1}, draw data Dᵢ (the agreement pattern of its draws) and probe outcomes
Bᵢ = (Bᵢ₁, …, Bᵢᵣ) from r label-action probes (negation, relabel, …).

Three types share the pool:

| Type | Meaning | Correct? | Passes all r probes with probability |
|---|---|---|---|
| C, concept-computer | computes the proposition | Z = 1 | π_C(r) ≥ (1 − ε)ʳ under independence |
| F, cue-follower | answers from a surface cue that paraphrases keep | Z = 0 | π_F(r) ≤ (½ + δ)ʳ under independence |
| B, stable wrong belief | holds a consistent false fact | Z = 0 | π_B(r) ≤ 1 (no bound: probes can't see it) |

Shares in the pool: ρ_F, ρ_B, ρ_C = 1 − ρ_F − ρ_B. Write ρ = ρ_F + ρ_B for the share of
stable-but-wrong cases.

## Assumptions

- **M1 (independent cases).** (p̄ᵢ, Tᵢ, Zᵢ, Dᵢ, Bᵢ) are i.i.d. across cases.
- **M2 (type does not depend on stability inside the pool).** Tᵢ ⟂ p̄ᵢ within the pool.
- **M3 (draws see stability only).** Dᵢ ⟂ (Tᵢ, Zᵢ) | p̄ᵢ. This is Lemma 4(i): by Lemma 1, the
  draws are i.i.d. Bern(p̄ᵢ) whatever the type.
- **M4 (probes see the type).** Bᵢ ⟂ (Dᵢ, p̄ᵢ) | Tᵢ, with the all-pass probabilities in the
  table. No independence between probes is needed: π_T(r) is the joint pass probability.
  With independent probes it becomes the product form of Lemma 4(ii).
- **M5 (correctness follows the type).** Z = 1 for C, Z = 0 for F and B.

M1–M5 are a stylised model, in the way the linear-Gaussian model of Yu, Wu & Block (2026)
is: they isolate the mechanism, they do not describe language. Each is testable (below).

## Theorem (separation)

Let T be the delivered set, |T| = k, and FDP = (1/k)·#{i ∈ T : Zᵢ = 0}.

**(i) Floor for resampling.** For any rule that picks T from the pool as a function of the
draw data (D₁, …, D_m) and independent randomisation, for any number of draws,

  E[FDP] = ρ.

**(ii) Probes go below the floor.** Let the rule first keep the pool cases that pass all r
probes (A = {i : Bᵢ = all pass}) and then pick T from A by any draw-only rule. On the event
|A| ≥ k,

  E[FDP | |A| ≥ k] = ρ_A := (ρ_F π_F(r) + ρ_B π_B(r)) / (ρ_F π_F(r) + ρ_B π_B(r) + ρ_C π_C(r))
               ≤ (ρ_F (½ + δ)ʳ + ρ_B) / (ρ_F (½ + δ)ʳ + ρ_B + ρ_C (1 − ε)ʳ)   (independent probes).

**(iii)** ρ_A < ρ if and only if ρ_F·[π_C(r) − π_F(r)] > ρ_B·[π_B(r) − π_C(r)]: the probes help
when the cue-followers they remove outweigh the enrichment of stable wrong beliefs (which pass
more often than correct cases). At the illustration values this holds up to r = 27.

Formal statement and proofs (LaTeX, ready for the paper): `docs/paper/sec_separation.tex`.

## Proof

**(i)** Fix i. The event {i ∈ T} is a function of D = (D₁, …, D_m) and of randomisation that is
independent of everything else. By M1 and M3, given D the correctness Zᵢ depends on D only
through Dᵢ, and P(Zᵢ = 0 | Dᵢ) = E[P(Zᵢ = 0 | p̄ᵢ) | Dᵢ]. By M2 and M5,
P(Zᵢ = 0 | p̄ᵢ) = P(Tᵢ ∈ {F, B}) = ρ for every p̄ᵢ in the pool. Hence

  E[1{i ∈ T}(1 − Zᵢ)] = E[1{i ∈ T} · P(Zᵢ = 0 | D)] = ρ · P(i ∈ T).

Summing over i gives E[#wrong in T] = ρ·E|T| = ρk. No independence between the selection
events is used, only linearity of expectation (as in Props. 1 and 2). ∎

**(ii)** By M1, conditioning on the set A conditions each case only on its own event
{Bᵢ = all pass}. By M4, for i ∈ A the type has the posterior
P(Tᵢ = t | Aᵢ) = ρ_t π_t(r) / Σ_s ρ_s π_s(r), and the type stays independent of (Dᵢ, p̄ᵢ). So
inside A the setting of (i) holds with ρ replaced by P(Zᵢ = 0 | Aᵢ) = ρ_A, and the argument of
(i), conditional on A with |A| ≥ k, gives E[FDP | |A| ≥ k] = ρ_A. For the bound: ρ_A has the
form x/(x + c) with x = ρ_F π_F + ρ_B π_B and c = ρ_C π_C; it increases in x and decreases in
c, so π_F at its upper bound, π_B = 1 and π_C at its lower bound give the stated maximum. ∎

## Corollaries

1. **Matched budget.** Self-consistency with n + r draws stays at ρ (part i). CEL with n draws
   and r probes reaches ρ_A < ρ (part ii). At equal calls, the gap is at least ρ − ρ_A.
2. **A finite best number of probes.** When ε > 0 and ρ_B > 0, ρ_A first falls and then rises
   with r: each extra probe removes cue-followers but also rejects some correct cases, while
   the B-type passes. The best r minimises ρ_A.
3. **Pool condition.** (ii) needs at least k probe-passers. With a pool of m_s cases,
   |A| ~ Bin(m_s, ρ_F π_F + ρ_B π_B + ρ_C π_C). Example: m_s = 30, r = 2 (below) gives a
   mean of 24 passers for k = 10.
4. **The blind spot is named.** The B-type (stable wrong belief) passes every probe; no
   label-action battery removes it. This is the limitation stated after Lemma 4, now with
   its size: ρ_A → ρ_B / (ρ_B + ρ_C π_C(r)) as π_F → 0.

## Illustration (parameters chosen for illustration, not estimated)

ρ_F = 0.15, ρ_B = 0.05 (so ρ = 0.20), δ = 0, ε = 0.05, independent probes:

| Probes r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|---|
| E[FDP] bound ρ_A | 0.200 | 0.141 | 0.108 | 0.091 | 0.084 | **0.081** | 0.082 | 0.087 |
| P(pass all) in the pool | 1.000 | 0.885 | 0.809 | 0.755 | 0.711 | 0.674 | 0.640 | 0.581 |

Without the B-type (ρ_F = 0.20, ρ_B = 0): r = 1 gives 0.116 and r = 2 gives 0.065.

## Error budget: an upper bound on error given k calls and M calibration cases

Notation: k = mean calls per case (the budget), M = M_fit + M_cal labelled calibration
cases, m = batch size, K = number of delivered cases.

**Starting point (exact).** With the conformal step (conformal p-values on M_cal + BH at
level α), E[FDP] ≤ α for any score, any budget and any M (Jin & Candès 2023). So k and M do
not decide whether the error is bounded; they decide the smallest α at which K cases can
still be delivered. Call it α*(k, M). It is the error bound CEL can offer at that budget.

**Derivation.** BH delivers at least K cases if p₍K₎ ≤ Kα/m, so α* = m·p₍K₎/K. With
p₍K₎ = (1 + N_K)/(M_cal + 1), where N_K counts the wrong calibration cases scoring at or above
the K-th test score t_K, and E[N_K] ≈ M_cal·ρ·F̄₀(t_K) (F̄₀ = upper tail of the wrong-case score):

  E[α*_K | t_K] ≤ m / (K(M_cal + 1)) + m·G(t_K)/K   (exact; G(t) = P(Z = 0, score ≥ t)),

because BH returns at least K cases once p₍K₎ ≤ Kα/m and the count in p₍K₎ is
Bin(M_cal, G(t_K)) given the test scores; m·G(t_K)/K is the top-K error of the score, up to the
randomness of the threshold. The score's
own error splits further:

  FDP_K(score) ≤ ρ_A(r(k)) + Δ_draw(n(k)) + Δ_fit(M_fit).

The step uses Prop. 1: ranking by the true posterior P(Z = 1 | signals) is the best size-K rule
on those signals, so it does at least as well as the hard-filter rule of part (ii), whose
error is ρ_A plus the draw term. The fitted score adds Δ_fit.

| Term | Meaning | Lever | Type |
|---|---|---|---|
| ρ_A(r(k)) | wrong share left after the r probes that the budget buys (part ii) | k (probes) | exact given M1–M5 |
| Δ_draw(n(k)) | unstable cases that pass the screen: ≤ (u/(1 − u))·[(p_sⁿ + (1 − p_s)ⁿ)/p_hⁿ]·(π_U/π_min), with unstable share u, unstable stability in [½, p_s), stable stability ≥ p_h | k (draws) | exact bound |
| Δ_fit(M_fit) | excess error of the fitted score over the best score on the same signals | M_fit | estimated by cross-fitting; distribution-free bounds are vacuous at M ≈ 200 |
| m / (K(M_cal + 1)) | calibration price: the smallest conformal p-value is 1/(M_cal + 1) | M_cal | exact |

**Calibration price (exact), m = 50, K = 10:**

| M_cal | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| smallest guaranteed error | 0.098 | 0.050 | 0.025 | 0.012 |

**Screen leak (exact bound), p_s = 0.8, p_h = 0.95:**

| unanimous draws n | 1 | 2 | 3 | 5 | 9 |
|---|---|---|---|---|---|
| max P(an unstable case passes) | 1.000 | 0.680 | 0.520 | 0.328 | 0.134 |
| ratio to a stable case's pass rate | 1.05 | 0.75 | 0.61 | 0.42 | 0.21 |

The ratio is what enters Δ_draw: draws separate stable from unstable cases only slowly, which
is one reason the budget also buys probes.

**Consequences.**
1. The budget k trades ρ_A (probes) against Δ_draw (draws). The greedy knapsack (Props. K,
   K′) is the rule that splits k between the two.
2. M splits into two parts with different roles: M_cal lowers the floor at an exact price of
   1/M_cal; M_fit lowers Δ_fit. For fixed M, choose the split that minimises
   Δ_fit(M_fit) + m/(K(M_cal + 1)), estimated by cross-fitting.
3. A bad fit costs deliveries, not errors: with the conformal step, a weak score raises α* (or
   delivers fewer than K at a fixed α); the guarantee E[FDP] ≤ α still holds.
4. Certified mode: with M used to fit the prior, the bound is Σε under the fitted prior plus
   the probability that the bootstrap depth is too low (about 5% by construction, approximate).
   The Jeffreys depth stays the exact reference.

**Planned output.** A figure of α* against k (x-axis) for several M_cal (lines), with the three
terms stacked, per dataset × model; the qualitative error taxonomy (missed cue-followers,
blind-spot beliefs, screen leaks, parse/paraphrase/probe/gold errors) is kept only as a check
that each term has the predicted size.

## Relation to the rest of the theory

- **Lemma 4** is the special case of one case and one test. This theorem lifts it to the
  delivered set, which is the object the paper evaluates.
- **Props. K, K′ (allocation)** count the same effect per call: a probe with κ > 1 is worth
  more than a draw. The theorem says why the effect does not vanish with more draws.
- **The residual model** (logit P(Z = 1) = a₀ + a₁η + Σβ_g x_g) fits ρ_A empirically: η
  carries the stability part, the β terms carry the probe part that M3 says draws cannot.
- **Type of result.** Exact given M1–M5; the model is stylised. Report it as such, next to
  the exact Prop. 2 and the model-based Prop. 3.

## How to test the assumptions on the data

| Assumption | Test (dev slice or calibration cases, labelled) |
|---|---|
| Part (i), floor | wrong share among unanimous cases ≈ constant as draws grow; the pool-size experiment shows self-consistency levelling off near 1 − ρ |
| M2 | wrong share among stable cases does not trend with the number of agreeing draws |
| M4 | all-pass rate of the probes among wrong vs correct stable cases: estimates of ρ_F π_F + ρ_B and π_C |
| ρ_A | wrong share among stable cases that pass all probes: a direct estimate |
| M3 | a logistic fit of Z on draw features within the stable pool has no signal beyond η |

## Open items

1. The theorem uses a hard filter (deselect on any failed probe). CEL uses a soft score; a
   version for the score s = μ̂ − λσ̂ needs Assumption 1 (Monotonicity) in the score.
2. Real cue-followers may have slightly lower stability than concept-computers (M2 fails
   mildly); then draws see them a little and the floor in (i) is soft. Quantify on the data.
3. The pool is defined by p̄, which is not observed; with finite draws the pool membership is
   itself estimated. A version with the pool defined by unanimity in n draws is the next step.
