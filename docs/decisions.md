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
17. IMPLEMENTED (2026-10-12): the parts of #16 needed for the two modes of
    docs/problem_setup.md. Simulated runs check the plumbing only.
    - Split: `split.n_cal` adds a calibration set D_cal (default 200; smoke 0), drawn after
      dev and eval from one shuffle; collected as `<ds>__<mk>__cal.jsonl`. D_fit = the dev slice.
    - Score model (`score_model.py`, model-based): logistic fit of Z on D_fit, per CEL-k
      allocation; stability term eta = psi(a+m) - psi(b+n-m) unpenalised, ridge on the other
      signals. sigma = delta-method sd from coefficient covariance + the case's own
      Var[logit p] = psi'(a+m) + psi'(b+n-m). Pessimistic score s = mu - lambda*sigma, lambda
      by 5-fold CV on D_fit (ties -> smaller lambda). Prop. P1: with equal evidence per case
      pessimism cannot change the top-k (tested); it can only help with uneven evidence,
      which the current fixed allocations do not create (frontier allocation: not built).
    - Certified mode (`certified_selection.py`): conformal p-values on D_cal + BH (Jin &
      Candès 2023; exact, any frozen score), for the pessimistic score, the mean score and
      self-consistency agreement (draws-only baseline); and e-BH on ledger e-values with
      D_fit pass rates, q0 = Jeffreys upper bound raised to the theory anchor (model-based).
    - Protocol: `pessimism` and `certified` sections per budget, same allocation and calls
      as CEL-k; reports FDR, power, size, calls per case.
    - Not built from #16: SHAP-ordered greedy allocation, GLR stop, fitted Beta prior for
      the certificate (sprt_posterior.py constants unchanged).
18. IMPLEMENTED (2026-10-12): frontier allocation for CEL-k (`frontier.py`), the batch rule
    of methodology.md §6 and the setting where pessimism (#17, Prop. P2) can act.
    - Each case gets k1 = 1 draw; then one call at a time until the batch has spent B calls
      per case on average: LUCB for top-k identification (Kalyanakrishnan et al. 2012) on
      mu +- z*sigma (z = 1) picks the case; the call with the largest expected move of mu
      (one-step lookahead, pass probability mu*q1 + (1-mu)*q0_hat from D_fit) is bought.
      Menu: next draw, each flip probe, challenge; SE left out (no pass/fail outcome).
    - Score model for partial evidence: each D_fit case contributes 8 random states; cov
      scaled by 8 (states of one case are not independent); lambda by CV with grouped folds.
    - Draw type (paraphrase vs original wording) chosen per budget on the dev batches.
    - Reported against self-consistency with B draws and Borda CEL-k at the same budget, with
      the spread of calls per case. Model-based. Frontier scores depend on the batch, so they
      are not exchangeable with calibration scores: no conformal step on the frontier.
19. IMPLEMENTED (2026-10-12): the rest of #16.
    - Signal values (`signal_value.py`, model-based): score model fitted on D_fit with every
      signal bought; SHAP closed form phi_g = beta_g (x_g - mean); I_g = mean |phi_g| with a
      bootstrap 90% interval; free signals (verbal, survived) pruned when the lower bound is
      below 0.01 (held at zero weight); v_g = I_g / c_g, a draw valued at I_eta / draws.
    - Prop. K allocation: per screen (1-4 draws, paraphrased or not), the label-free dev
      survival sigma gives floor((B - para - k1) / sigma) stage-2 items, bought by value;
      the screen with the largest total value wins. SE only with a screen of >= 2 draws.
      Budget met in expectation on dev, like the enumerated CEL-k it replaces (eval calls
      can run slightly above B). Reported against the enumerated allocation and SC.
    - Fitted prior (`stability_prior.py`, model-based): beta-binomial MLE on 5 fresh draws
      of every holdout case (D_fit + D_cal, labels unread); depth = bootstrap 95% quantile;
      replayed on eval next to Jeffreys (9, exact). sprt_posterior.py constants unchanged.
    - GLR stopping: decision rule and depths (25 / 29 unanimous at delta 0.1 / 0.05; 43 / 46
      with one disagreement) in the theory constants. Not replayable: the collection holds
      12 fresh draws per case.
    - Not built: the lean 4-call battery with text signals (needs new prompts and a new
      collection).
20. IMPLEMENTED (2026-10-12): the text and gray-box signals of the lean battery, as features
    of the score model (no new calls).
    - length (mean words of the draws' rationales) and hedge (share with a hedge word),
      from the cached explanations.
    - neg_drop: the negation probe now carries the confidence line when
      collect.verbal_per_draw is on; signal = draws' mean confidence - confidence under
      negation. Changes the negation prompt, so its cache key (no real collection yet).
    - p_first: P(Yes) at the answer token, normalised over Yes/No in the top-5 logprobs,
      on every draw when collect.logprobs is on (OpenAI and SafeChain; not reasoning
      models). GRAY-BOX: held out of every fit unless scoring.gray_box; report it as a
      separate arm. Cache keys change only for logprob requests.
    - Free signals join the pruning step; neg_drop's importance adds to the negation call's
      value. The 4-call lean battery itself is one allocation of the existing space.
21. Calibration and AUARC (metric only; following Pedapati et al., arXiv:2406.04370, who
    report AUROC, AUARC and ECE). AUARC is added for every method. Calibration (ECE with 10
    equal-width bins, Brier, equal-count reliability bins with Jeffreys intervals) is
    reported only for scores that claim to be P(Z = 1): the score model's mu (fixed and
    frontier allocations), SC agreement m/n and verbal confidence/100, pooled over eval, each
    against its own method's Z. mu - lambda*sigma ranks and is not calibrated. Secondary to
    prec@k; nothing is tuned on these.
22. Answer format stays binary (Yes/No verification) for the main 3x3 grid. Open-ended QA
    is a later generality experiment, not yet implemented: one cell (TriviaQA open-ended,
    gold aliases, gpt-4.1), generate-then-verify -- cluster sampled answers for agreement on
    the modal answer a, then run the label-action probes on "Is a the correct answer to
    [question]?", so Lemma 4's (1/2 + delta)^r bound still applies. Needs an answer-
    equivalence judge (its own dev audit) and graded Z; base accuracy is not controlled and
    SC is expected to be stronger there. Run after the main grid.
23. Baselines subpackage (src/confledger/baselines/, one module per source), replayed from
    the existing collection at the matched budget, each on its own Z:
    - Xiong et al. 2024: Avg-Conf over b self-random draws (their recommended aggregation)
      and over paraphrase draws ('prompting' sampling, + 1 generator call). Reference
      answer = the majority (the paper leaves it unspecified). Needs verbal_per_draw.
    - CISC (Taubenfeld et al. 2025): softmax(c_i/T)-weighted vote over b draws, T on their
      80-point grid tuned on D_fit for vote accuracy (ties -> larger T). Selection score =
      the winner's weight share (our adaptation; CISC defines only the answer). Verbal
      source; answer-token probability as a gray-box variant. WQD reported on eval.
    - Pedapati et al. 2024, "Pedapati-style": semantic-set count + mean ROUGE-L of full
      responses for stochastic decoding and paraphrase draws, logistic regression on D_fit.
    Not implemented (new prompts or a local NLI model): misleading hints, Top-K/Pair-Rank,
    P(True), sentence permutation, entity-frequency, stop-word removal, split-response NLI.
    Paper §4 baselines and metrics updated.
24. Four headline results (docs/result.md, .pdf): R1 budget curve (ranking mode), R2 metrics
    table (certified mode, adaptive CEL at matched budget), R3 calibration curve, R4 mode
    linkage; everything else is appendix.
    - CEL-k ranks by mu - lambda*sigma (score model on D_fit, lambda by CV, 0 allowed);
      Borda is a variant. The allocation is still chosen on dev by Borda prec@10.
    - R1: every baseline at exactly k calls; CISC and Pedapati-style refit at each k.
    - R2b: conformal BH at the matched budget for adaptive CEL (mu - lambda*sigma) and every
      baseline (its primary score), D_cal replayed under each policy: exact, since each
      policy's spend depends only on the case's own draws. SPRT certificate reported
      beside it as exact for stability, not correctness.
    - R3: calibration.py (reliability figure + calibration.tex) at k = 3 and matched budget.
    - R4: dev-chosen and derived CEL-k both ranked by mu - lambda*sigma; conformal
      certified share added beside the SPRT shares.
    - cel() now records the screen's agreement (m, k) for eliminated cases, as cel_alloc does.
25. R2 is the metrics table at the elbow k* of CEL-k's dev curve (Kneedle on dev prec@10
    over budgets, running max; ties -> smaller k; dev only): CEL-k and every baseline at k*
    calls, conformal power at alpha = 0.1, certified CEL at its own spend (the SPRT sets it)
    with its certified share (elbow.tex). The matched-budget ladder and certified.tex stay as
    R2 detail. Formal definition of both modes: docs/method.md; results: docs/result.md.
26. CEL-k's budget is a target, not only a cap (2026-10-21): full use of the budget is the
    goal. Per budget k, the dev-chosen allocation must spend in (k - delta, k] expected calls
    per case on dev (delta = scoring.budget_band, default 1); if none does, the best of those
    spending the most without exceeding k. Ties go to more calls (was: fewer). Reason: on the
    5-item smoke dev slice every one of the 1,760 allocations scored 1.0, so every k chose the
    1-call plan and the budget curve collapsed to one point; at ~90% base accuracy the same
    saturation is likely on the full dev slice. Paper Eq. (devalloc) and method.md updated.
27. Selection size is a share of the batch (2026-10-21): the top 10% of each batch
    (config `top_frac: 0.10`; k = ceil(0.10 * batch_size) = 5 of 50), reported as prec@10%.
    Was a fixed top-10 (20% of a 50-item batch). Cost: 5 cases per batch makes prec move in
    steps of 0.2, so per-batch variance is higher and the confirmatory test loses power.
    Paper, method.md, problem_setup.md and result.md updated; the sketch figures in
    docs/figures still say prec@10.
29. No batches (2026-10-22): the top 10% is taken over the whole evaluated population
    (split.batch_size removed; k = ceil(top_frac * n_eval)). Uncertainty and the
    confirmatory test use a paired bootstrap over items (metrics.prec_rep / paired_compare,
    10,000 resamples shared by every method, two-sided percentile p) in place of the
    per-batch Wilcoxon test. Dev plan choice uses the same top share of the dev slice.
    Selection stability is re-drawn from cached draws (seeds 1 vs 2), so no batches are
    re-collected (--reseed-batches defaults to 0). Paper text still to update.
30. Semantic-entropy judge is a local NLI model (2026-10-22): microsoft/deberta-large-mnli,
    rationales grouped by bidirectional entailment with the question prefixed (Kuhn et al.
    2023; utility/nli.py, config judge.provider = local_nli). No API calls, independent of
    every model under test; replaces the gpt-5 judge and its same-provider deviation (#6).
    No judge-swap comparison with gpt-5 (alt_judge removed from default.yaml). Existing
    collections are re-clustered offline with experiments/scripts/recluster.py.
    Needs the `nli` extra (torch, transformers).
31. Models under test are no longer OpenAI-only (2026-10-22): gpt-4.1 and gemini-2.5-flash,
    more to follow. Non-OpenAI models use their OpenAI-compatible endpoint with an explicit
    base_url (never routed through SafeChain); provider-specific request fields go in the
    model entry's `extra` (Gemini: reasoning_effort "none", thinking off, so it samples at
    temperature like gpt-4.1). The generator stays gpt-5 (different from every target).
    Supersedes the OpenAI-only part of #6.
32. Open-weights models under test (2026-11-01): qwen3-8b (Qwen/Qwen3-8B, Apache-2.0, thinking
    off) and llama-3.1-8b (meta-llama/Llama-3.1-8B-Instruct, Llama 3.1 community licence, gated:
    needs HF_TOKEN from an account with access). Original bf16 weights, about 16 GB each, so
    they need an A100-class GPU on Colab; 4-bit builds were considered and dropped so results
    describe the published models. Revisions are pinned in the config (`repo`, `revision`);
    `model` is the name vLLM serves, with no slash, because it names the output files. Served by
    vLLM (OpenAI-compatible, same backend as the API models, a separate venv on Colab) and
    collected on Colab with experiments/scripts/colab_targets.ipynb. Local inference has no API
    price, but its calls count as calls: the budget is target-model calls, whoever pays for them.
33. Interim semantic-entropy judge: gpt-5 (2026-11-02), reasoning_effort low, one judge for
    every model, supersedes #30 for now. The strict NLI rule (mutual entailment) split
    rationales that differ only in detail: 6-16 groups per item on the smoke, against 3 for
    gpt-5 on gpt-4.1, and on gemini and qwen the signal's AUROC fell to about chance or below.
    An API judge applies the prompt's definition, "same underlying reason", directly. It costs
    1 judge call per item. To be replaced by a stronger judge once the direction is confirmed;
    experiments/scripts/recluster.py re-groups existing collections with the config's judge
    and makes no new calls to the models under test. The local NLI path stays in the code.
34. CEL-k safeguards after the Gemini pilot's dip at k = 8 (2026-11-02). With 2 draws the
    fit on 74 dev cases gave the stability term a negative coefficient (-0.71), so 1-1 splits
    scored mu = 0.98-0.995 and 5 of the top 20 were wrong; Borda on the same plan was 1.00.
    Found by reading eval items, so the three changes below were frozen before the gpt-4.1
    pilot, the first data that judges them; Gemini's interim numbers predate them.
    - Stability coefficient >= 0 (amends #16): more agreement may not lower confidence.
      A negative estimate is refitted at the bound 0. It stays unpenalised otherwise.
    - Odd total draws in every CEL-k and knapsack plan, so a survivor's majority cannot be a
      tie. |A| falls from 1,760 to 864. Adaptive CEL's 3 + 6 draws stay in the space; a
      settings change giving it an even total would break nesting (Prop. nesting).
    - Plans are chosen by dev AUARC of the Borda ranking, not dev prec@top share, which sat
      at 1.00 across budgets 5-12 and could not separate plans. Ties still go to more calls.
      The elbow k* now reads the same dev AUARC curve (amends #25). lambda stays on CV prec.
Observed in dry run: the screen trades global AUROC/PRR for top-k precision (address in
paper); plain Jaccard understates stability under top ties (add tie-aware variant).

## Naming
- Method: CEL (Confidence Evidence Ledger); package/CLI: confledger. Not "bench": this is
  a method, the benchmark is only how it is evaluated.
