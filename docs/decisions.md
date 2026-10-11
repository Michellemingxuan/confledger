# Design decisions and rationale

Condensed from the design conversations. Each entry: decision — why.

## Problem framing
- Black-box API, no labels, limited budget, goal = top-k (k=10) per batch — the
  industry regime existing UQ papers don't address (they assume free sampling or labels).
- Confidence is a *second-order* prediction: score s predicts Z = 1{Yhat=Y}. Only
  ranking-grade confidence is needed (Prop. 1: under monotonicity, top-k by s is optimal);
  no calibration claimed — that would require labels.
- Single named assumption: Monotonicity, P(Z=1|s) nondecreasing. Tested on benchmarks
  (global strata + isotonic gap; local top-strata test where Prop. 1 consumes it).
- Behavioural (sampled) confidence over verbalized: RLHF inflates/clusters stated
  confidence (80–95% pile-up) but preserves rank, so verbal = final tie-breaker only.

## Sampling design
- Fresh paraphrase per draw (Lemma 1, suggested by Michelle): draws exactly i.i.d.
  Bern(pbar); reuse costs design effect 1+(m-1)rho (Lemma 2). Estimand is relative to
  the paraphrase frame Q; semantic drift contaminates the estimand, not independence.
- Truncated SPRT (p0=0.7, p1=0.95, alpha=beta=0.1): 3-draw screen, any disagreement
  eliminates; stage 1 never certifies (0.92 < A=2.20). Stage 2 is the continuation region.
  Rejection cheap, certification expensive (Lemma 3: E[N] 5.4 / 8.8 vs ~12 fixed).
- Jeffreys prior, not empirical Bayes: at equal depth the ranking is prior-invariant; for
  the certificate, EB would double-use data, borrow strength across cases, and drift with
  batches. "Estimation borrows strength; certification must not."
- k=9 is derived (Cor. 1: minimal unanimous depth for eps<=0.05 at p0*=0.8), not chosen.
- Compute-limited deployment: lean stage 2 (2 draws + probe battery) for survivors and
  deepen only the top ~15 contenders. Probes are cheaper evidence per call. (Runner archived
  in docs/archive/deployment.py; not part of the Sec. 4 experiments.)

## Signals and aggregation
- Five signals: agreement lower bound L, invariance battery (specification test vs
  cue-following; resampling has zero power there, Lemma 4), Miller–Madow rationale
  entropy, challenge resistance, verbalized confidence.
- Cascade primary, Borda robustness. Exact Shapley over 32 coalitions audits the order.
- Evidence exchange (Prop. 3, Michelle's objective): log-LR as common currency; a passed
  negation/relabel probe ~ 2.1 draws, distractor ~ 0.56, failed probe ~ -7.5.
  Model-based (naive-Bayes composite), kappa is an upper bound; report kappa and kappa/2,
  test with arms A (12 draws) / B (6 draws + battery) / C (9 draws), fit kappa_hat.

## Experiments
- Datasets = one per error regime: TriviaQA-verification (wrong memorised facts, stable-
  but-wrong), StrategyQA (broken reasoning), MMLU (knowledge boundary; adversarial pick —
  LM-Polygraph found verbalized beats sampling there). 500 eval items (10 batches x 50)
  + 100 dev items per dataset. Three model families; judge from a different provider.
- Baseline ladder at matched budget: random -> verbalized -> self-consistency ->
  semantic entropy -> ours (+ BSDetector-style). Confirmatory test: ours vs matched
  self-consistency, paired Wilcoxon over batches, Holm across cells.
- Headline figure: prec@10 vs calls/case; value = vertical gap at low budget.

## Decisions made during implementation (not yet in the paper)
1. Held-out probe = conversational re-framing (entity renaming changes factual answers).
2. TriviaQA distractors from a separate generator model, checked against gold aliases.
3. Matched budget auto-set to the pipeline's realised calls/case.
4. Shapley holds the stage-1 screen fixed; report random->screen separately.
5. No-SPRT ablation = flat 2 draws + all probes (9 calls incl. paraphrase generation, see #8).
6. Models are OpenAI-only (2026-09-29): real runs go through SafeChain in the private
   env, which serves only OpenAI models. Model under test = gpt-4.1. DEVIATION:
   the judge cannot come from a different provider; it is a different OpenAI model,
   and judge sensitivity (alt judge) carries the burden. The generator must still be
   a different model from every model under test (it sees gold answers). Transport is
   an env switch (LLM_BACKEND=safechain), mirroring AgenticSys_v2; cache keys carry the
   transport, so dev and SafeChain responses never mix. Risk to check in the dev audit:
   SafeChain's InputRedactor may alter prompt text (e.g. names in trivia questions).
7. TriviaQA rc.nocontext lists most questions twice (web + wiki, same question_id);
   the loader de-duplicates by question_id before sampling, otherwise 17 of 600 items
   repeat and can straddle dev/eval.
8. Paraphrase generation counts as 1 call per case (2026-09-29). Every policy that draws on
   paraphrases (CEL, its ablations and budget configs, evidence arms) pays 1 generator call
   before its screen; same-prompt baselines do not. Previously uncounted, which under-stated
   CEL's cost and the matched budget by 1. Paper numbers to update: main.tex:759 no-SPRT
   8 -> 9 calls; :766-768 arms A/B/C 12/9/9 -> 13/10/10 (relative comparison unchanged);
   :874 matched budget (re-read from the first real run); :736 CEL cost per case (+1).
9. Signal effects are measured two ways (2026-09-29). Leave-one-out, on prec@10, AUROC and PRR:
   an ablation per signal (cel_no_L / _flip / _se / _chal / _verb), i.e. CEL's own cascade
   without it, with a paired CI over batches and calls saved. Shapley: on prec@10 as before,
   now with a bootstrap CI over batches (phi is linear in the
   per-batch V). Removing L drops only its tie-break; its draws stay (the SPRT needs them).
   Paper Sec. 3.8: add the leave-one-out view next to Shapley.
10. Verbal confidence per draw is a config switch, collect.verbal_per_draw (default false;
    2026-09-29). On: every sampled draw's prompt asks for a Confidence line; the separate T=0
    verbal call is dropped (35 target calls/question, -1); CEL's verb signal = mean confidence
    of the draws it uses (free); the verbalized baseline reads the first same-prompt draw.
    Risk to check on the dev slice: asking for confidence in the same response can shift the
    answers themselves. Also: every method now charges the draws it actually uses, and the
    protocol refuses a collection smaller than it needs (>= 12 fresh, >= 12 same-prompt,
    >= 9 reuse draws, n_same >= matched budget) instead of mislabelling short replays.
11. Self-consistency deployment baseline + n_same = 16 (2026-09-29). Self-consistency is
    reported as a curve (3/6/9/12), at the matched budget (confirmatory test; unchanged) and
    at `deployment_sc_draws` (default 3) as the cheap deployment baseline: its own row in
    main_ladder.tex, ringed on the headline figure; it pairs naturally with CEL's cheapest
    point (cel_b3: paraphrase-gen + 3 screen draws = 4 calls). Because every CEL call is
    counted (paraphrase-gen, draws, 3 flips, judge, challenge, verbal), CEL's mean spend is
    ~12-13 calls and can reach 16, above the 12 same-prompt draws collected before; n_same
    now defaults to 16 (+4 target calls/question) and collection warns when it is smaller.
    Paper to update: main.tex:750-753 describe self-consistency and semantic entropy as
    "8 draws" at identical spend -- the matched budget is now read from the run (~12-13);
    add the deployment baseline row.
12. Budget-matched CEL-k (2026-09-29), for deployment where adaptive CEL (~12-16 calls/case)
    is too expensive. CEL-k spends exactly k calls/case, like self-consistency with k draws:
    1 draw on the original wording + the k-1 highest-ranked one-call probe units (2nd draw,
    negation, relabel, distractor, challenge, verbal). Units are ranked by exact Shapley value
    (prec@k of Borda over all 64 coalitions) on the DEV collection and frozen before eval;
    prior order (derived kappa) when the dev slice is too small (e.g. the 10-item smoke dev).
    No screen, no certificate: the ranking mode; adaptive CEL stays as the certified mode.
    The challenge now targets the original-wording draw's answer (same[0]) so CEL-k can use
    it; adaptive CEL counts it only when that answer equals its own. Reported for
    budget_matched_k = 2..5 in budget_matched.tex and as a third line on the headline figure.
    Paper: add CEL-k as the deployment mode (Sec. 3.9) and the budget-matched table (Sec. 4).
13. SPRT settings are configurable (2026-09-29): `sprt: {p0, p1, alpha, beta, eps0, p0_star}`,
    defaults = the paper's values (unchanged results). The stage-2 depth is derived, never set
    by hand: unanimous draws to certify = max(SPRT boundary, Cor. 1: eps <= eps0 at p0_star)
    (certification_plan). With the defaults the SPRT alone needs 8 and Cor. 1 needs 9, so 9
    (3 + 6). Widening H0/H1 alone saves nothing (Cor. 1 binds); fewer draws need a larger eps0
    too, e.g. p0 = 0.6 + eps0 = 0.13 certifies at 5/5 (3 + 2): survivors cost 12 instead of 16
    calls, the certificate only rules out stability <= 0.6, and Prop. 2's bound loosens.
    sprt_settings.tex tabulates the trade-off exactly (sensitivity, not a new default).
14. CEL-k becomes budget ALLOCATION, chosen on the dev slice (2026-09-29; supersedes #12's
    greedy per-unit Shapley ranking). An allocation spreads the budget over a screen (k1
    draws, original or paraphrased; any disagreement eliminates) and, for survivors only,
    extra draws, a subset of the flips, the challenge and SE (1 judge call); verbal confidence
    rides on every draw (collect.verbal_per_draw now true in the configs). All ~1,250
    replayable allocations are scored on dev (prec@k, AUROC if dev batches are too small); per
    budget k the best one whose EXPECTED calls/case on dev are <= k is frozen and evaluated
    once on eval. Comparison: CEL-k on expected calls (batch average), self-consistency on
    exactly k draws (user's specification). Control added: self-consistency + verbal
    (same k draws, ties by mean confidence) -- with verbal free for every method, it
    separates "better allocation" from "reads the confidences". Simulated runs show the
    control matters: it matches CEL-k on 2 of 3 datasets. Dev-selection optimism: 100 dev
    items per dataset in the full run; eval stays untouched.
    Full write-up (strategy, cost model, theory, limitations): docs/cel_k_allocation.md.
15. Generator and judge = gpt-5, a reasoning model (2026-09-30; models available: gpt-4.1 and
    gpt-5, gpt-4.1 is under test). A `reasoning_effort` in a model spec switches the backend
    (direct API and SafeChain) to reasoning parameters: no temperature (fixed default), and
    max_completion_tokens = the request's cap + `reasoning_tokens` (default 4000) of headroom
    for hidden reasoning. Generator effort minimal, judge effort low. Consequences: paraphrases
    and distractors are sampled at gpt-5's default temperature instead of 0.7, and the judge's
    clustering is no longer temperature-0; both are cached, so every replay is reproducible.
    The model under test (gpt-4.1) keeps T=0.7 draws and T=0 probes unchanged.
15. Bridge between adaptive CEL and CEL-k (2026-09-29): one evidence ledger under two
    constraints -- adaptive CEL spends until certification, CEL-k maximises evidence under
    an expected-call budget. Sec. 3 theory was written for adaptive CEL only; now:
    - Nesting (exact): adaptive CEL = allocation (3 para draws, +s2, 3 flips, chal, SE)
      with the cascade combiner (methods.cel_as_allocation). allocation_space's max_extra
      now reaches s2 (was 4 < 6), so adaptive CEL is in CEL-k's space and CEL-k at
      k >= C(a_CEL) can choose it. Same calls, answers and m/k (test pins it).
    - Carries over unchanged: Prop. 1; Lemma 1 for para allocations only (orig ones are
      Lemma 2's m > 1 corner); Lemma 3(a) prices the screen exactly; Prop. 2 applies to
      every delivered CEL-k set (eps at 1/2/3/5 unanimous: 0.45/0.31/0.23/0.13 -- the
      certificate exists but is weak, which is why k <= 5 is the ranking mode).
    - Certification budget (Cor. 2): per survivor, exact needs 9 unanimous draws (Cor. 1);
      model-based (Prop. 3) needs 7 calls (4 draws + battery, k_eff 8.8, eps_eff 0.045);
      at kappa/2 it needs 9 (no saving). Reported as least EXPECTED calls on dev.
    - Derived allocation (model-based, label-free): per budget, the para allocation with
      max k_eff under expected calls <= k. Predicts neg/relabel (kappa 2.1) before extra
      draws, distractor (0.56) after, never chal/SE (no derived rate). Compared with the
      dev-chosen allocation, like phi/c vs kappa; the gap prices the unpriced signals and
      the double-counting.
    Code: select_allocations(..., kappas, cert) returns `derived` and `certification_budget`;
    delivered_certificates() = certified share of the delivered top-k (exact / model-based)
    and mean sum eps; run_protocol's `linkage` block; config key linkage_k (budgets for the
    figure only, not in the SC table); experiments/scripts/mode_linkage.py.
    Paper: Sec. 3.9 (Prop. 4 nesting, Cor. 2 certification budget, derived allocation),
    Sec. 4 mode-linkage figure + mapping row. Simulated dry run: the dev-chosen CEL-k stops
    spending at ~2 calls because simulated dev prec@10 saturates (ties -> fewer calls);
    a simulator artefact to recheck on real data, not a finding.
16. PROPOSED, not in code (2026-10-08): headline theorem + method redesign. Design log with
    each idea's propose/validate/challenge/outcome: the "CEL Run Pipeline" artifact.
    - Headline theory: separation theorem (docs/separation_theorem.md). Any draw-only rule,
      self-consistency with any number of draws included, has E[FDP of top-k] = rho, the
      share of stable-but-wrong cases; r label-action probes give rho_A < rho. Exact given a
      stylised model (M1-M5); lifts Lemma 4 from a test to the delivered set; names the
      blind spot (stable wrong belief passes every probe) and gives a finite best r.
    - Redesign (each item to be decided before coding): calibration set of M labelled cases
      outside eval (8 calls/case: 1 generator + 5 fresh-paraphrase draws + negation + relabel);
      logistic fit of Z with an unpenalised stability term eta = psi(a+m) - psi(b+n-m) and a
      ridge penalty on the other signals only; SHAP importance to rank/prune signals and
      v_g = I_g / c_g as the greedy order (Props. K, K': equal-weight knapsack, model-based),
      replacing the dev-chosen enumeration (#14, kept as an offline reference); pessimistic
      score s = mu_hat - lambda*sigma_hat (lambda by 5-fold CV on M_fit) replacing Borda;
      optional conformal selection + BH on M_cal for FDR on correctness; optional GLR stop in
      the certified mode (~25 unanimous draws at delta = 0.1, vs 9).
    - Reverses "Jeffreys prior, not empirical Bayes" for the certificate, at the user's choice:
      a Beta prior per dataset x model fitted on the calibration draws, depth = bootstrap 95th
      percentile (pessimistic); the certificate becomes model-based, the Jeffreys depth (9)
      is reported next to it as the exact reference. Double use is avoided because the prior
      is fitted on calibration cases only, never on the cases it certifies.
    Paper, tests (test_sprt_posterior.py for any prior change) and code follow only after the
    items are confirmed.
Observed in dry run: the screen trades global AUROC/PRR for top-k precision (address in
paper); plain Jaccard understates stability under top ties (add tie-aware variant).

## Naming
- Method: CEL (Confidence Evidence Ledger); package/CLI: confledger. Not "bench": this is
  a method, the benchmark is only how it is evaluated.
