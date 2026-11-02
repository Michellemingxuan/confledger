# Related Work and Novelty Positioning: Call-Efficient, Probe-Based LLM Confidence with Shapley-Ordered Sequential Probing, Pessimistic Scoring, and Conformal Selection

The core claim of your paper, that a budget of diverse probes (negation, challenge, paraphrase, etc.) gives better-calibrated confidence than the same number of i.i.d. self-consistency samples, does not appear to have been stated or tested head-to-head in prior work, but the one existing equal-call comparison (Xiong et al., ICLR 2024, M=5) partly cuts against it, so the novelty is real but must be earned empirically. Each individual component (perturbation/challenge probes, learned combination of black-box signals, adaptive stopping, conformal prediction for LLMs) has close precedent; what we could not find is (i) an explicit "probes vs. SC samples at matched call budget" calibration claim, (ii) Shapley-value ranking of probes to define a sequential, early-stopping probing order, and (iii) pessimistic (min/worst-case) aggregation over probes feeding a conformal step.

## TL;DR

- **Closest prior work:** Xiong et al. (ICLR 2024) already used "misleading hints" (induced consistency) as an alternative to temperature sampling at M=5 calls; SPUQ (Gao et al., EACL 2024) showed input perturbations improve calibration over plain sampling; Pedapati et al. (arXiv:2406.04370) learned a logistic-regression confidence model over engineered black-box perturbation features; Multi-Perspective Consistency (arXiv:2402.11279) and BSDetector (Chen & Mueller, ACL 2024) mix self-verification with sampling. None claims probes are more call-efficient than SC samples at equal budget for calibration.
- **Genuinely novel (as far as a broad search shows):** the equal-budget efficiency claim itself, Shapley-ranked sequential probing with early stopping for confidence, pessimistic/min aggregation over probes, and their combination with conformal top-k selection. Biggest novelty risk: Xiong et al.'s equal-call table shows misleading hints give slightly better ECE but *worse* AUROC than i.i.d. sampling, and reviewers will read your system as an engineering combination of known parts unless the matched-budget "SC-equivalent samples" result is strong and robust.
- **Biggest methodological risks:** the 50/50 split leaves no untouched test set unless you add one; conformal validity survives learned ordering and adaptive stopping only if the entire pipeline is frozen before calibration and stopping thresholds are calibrated jointly (e.g., Learn-Then-Test); challenge-probe flips partly measure sycophancy rather than uncertainty; and an equal-*call* comparison is unfair to SC if your probes use longer multi-turn contexts or force sequential (non-parallel) calls — report tokens, dollars, and latency too.

## Key Findings

### 1. Repeated-sampling confidence: how many samples, and how good

| Method | Typical samples | Reported quality | Notes |
|---|---|---|---|
| Self-consistency (Wang et al., ICLR 2023) | 40 sampled paths in the original setup | Accuracy method; agreement rate is the de facto confidence | The fixed 40-sample budget is the explicit target of Adaptive-Consistency and ESC |
| Semantic entropy (Farquhar et al., *Nature* 2024) | M≈5–10 | Mean AUROC 0.790 across 30 task–model combinations vs. naive entropy 0.691, P(True) 0.698, embedding regression 0.687\[1\] | Supplement: "after roughly M = 5 there are diminishing returns, although going up to M = 10 can still help"\[2\] |
| Bayesian semantic entropy (Ciosek, arXiv:2504.03579) | Adaptive, as low as 1 | Same AUROC as Farquhar et al. with 47% fewer samples; AUROC 0.796 on Llama-2/SVAMP with a single sample\[3\] | Directly relevant "budget" baseline |
| SelfCheckGPT (Manakul et al., EMNLP 2023) | N=20 | Sentence-level AUC-PR (non-factual) 93.42 (Prompt), 92.50 (NLI), 85.63 (unigram) on WikiBio | Prompt variant cost estimated at ~$200 for GPT-3 over 1,908 sentences × 20 samples; gains flatten with more samples\[4\]\[5\]\[6\] |
| Xiong et al. (ICLR 2024) consistency | M=5 | Self-Random sampling AUROC 73.0 vs. single CoT response 56.4 (GPT-3.5 average)\[7\] | Consistency beats verbalized confidence "in most cases"\[7\]\[8\] |
| Diversity-steered sampling (Park & Cho, NeurIPS 2025) | N=16 | Matches or beats vanilla sampling AUROC for semantic entropy with fewer redundant samples | Sample-efficiency via decoding, not prompts\[9\]\[10\] |

Cost-reduction work on self-consistency is mature but targets *answer accuracy*, not calibration:

- **Adaptive-Consistency** (Aggarwal et al., EMNLP 2023) uses a Dirichlet-based stopping rule and "reduces sample budget by up to 7.9 times with an average accuracy drop of less than 0.1%" across 17 reasoning and code datasets and three LLMs.\[11\]
- **Early-Stopping Self-Consistency (ESC)** (Li et al., ICLR 2024) stops once a sliding window of samples is unanimous, with a derived upper bound on the inconsistency probability introduced by stopping early.\[12\]
- **Confidence-Informed Self-Consistency (CISC)** (Taubenfeld et al., Findings of ACL 2025, arXiv:2502.06233) weights votes by the model's own confidence (P(True) worked best) and, tested on nine models and four datasets, reduces "the required number of reasoning paths by over 40% on average"; plain SC needed 18.6 samples to match CISC with 10. The authors note that "the most calibrated confidence method proved to be the least effective for CISC" — a useful warning that calibration and vote-weighting usefulness diverge.

**Implication for you:** the bar for "equivalent to more SC calls" is not SC@40; it is SC@5–10 plus adaptive stopping (Adaptive-Consistency/ESC) or Bayesian semantic entropy, because those already capture most of the achievable AUROC at small budgets.

### 2. Prompt perturbations and challenge probes as confidence signals

- **Xiong et al., "Can LLMs Express Their Uncertainty?" (ICLR 2024, arXiv:2306.13063)** — the single most important prior paper for your novelty claim. It defines a prompting × sampling × aggregation framework and explicitly includes *induced consistency* via misleading hints ("Weak Claim", "Strong Claim", "External Source" hint types) alongside self-random sampling and paraphrasing.\[7\] At equal M=5 calls (GPT-3.5, CoT, consistency aggregation, averaged over datasets): misleading hints ECE 17.3 / AUROC 69.6; self-random sampling ECE 18.7 / AUROC 73.0; paraphrase ECE 24.3 / AUROC 69.2; single response ECE 25.0 / AUROC 56.4. With Avg-Conf aggregation, sampling still leads on AUROC (74.5 vs. 70.2) while ECE is near-tied (14.8 vs. 14.2). Their recommendation is Top-K prompting + self-random sampling + Avg-Conf/Pair-Rank aggregation, and they describe the differences between sampling strategies as minimal.\[7\] **This is evidence against a naive "probes beat sampling" claim**; you must show your gains come from *heterogeneous, ordered, learned* probe sets rather than a single perturbation type.
- **SPUQ (Gao et al., EACL 2024, Intuit)** perturbs inputs (paraphrase, dummy tokens, system-message changes, temperature) and aggregates sampled outputs, framing perturbation as capturing epistemic uncertainty and sampling as aleatoric; its abstract reports "a reduction in Expected Calibration Error (ECE) by 50% on average" relative to unperturbed sampling (the 30–70% per-setting range comes from a secondary summary and should be checked against the paper's tables), with calibration plateauing near k=5 perturbations. It is meaning-preserving perturbation, not adversarial challenge, and does not state an equal-call efficiency claim.
- **Prompt agreement (Portillo Wightman, DeLucia & Dredze, TrustNLP 2023)** estimates confidence from agreement across multiple prompts/paraphrases and reports better calibration than single-prompt log-probabilities — an early multi-prompt-instead-of-resampling precedent.\[13\]\[14\]
- **Pedapati et al., "LLM Confidence Estimation via Black-Box Access" (arXiv:2406.04370)** engineer features from stochastic decoding, back-translation paraphrase, sentence permutation, entity-frequency amplification, stop-word removal, and split-response consistency, then train an interpretable logistic regression, beating baselines "by even over 10% (on AUROC) in some cases" on four QA tasks for Flan-ul2, Llama-13b, Mistral-7b and GPT-4, with zero-shot transfer of the confidence model across LLMs.\[15\]\[16\] **This is the closest analogue to your "collect signals on a labelled split, then learn a calibrator" design**, minus Shapley ranking, budgets, and conformal guarantees.
- **Multi-Perspective Consistency (Wang et al., arXiv:2402.11279)** uses K=15 verifier-style queries (MPC-Internal) plus cross-model confidence; on GPT-4 it reports +6.2% AUROC and −0.095 ECE on MedQA vs. self-consistency, and AUROC 0.851 vs. 0.804 on TruthfulQA.\[17\]
- **BSDetector (Chen & Mueller, ACL 2024, arXiv:2308.16175)** combines observed consistency over k samples with self-reflection certainty from follow-up "is this correct?" questions via a weighted average; its abstract states it "more accurately identifies incorrect LLM responses than alternative uncertainty estimation procedures (for both GPT-3 and ChatGPT)".
- **"Are You Sure?" / FlipFlop (Laban et al., arXiv:2311.08596)**: the v2 arXiv version (Feb 2024) reports that "a systematic study of ten LLMs on seven classification tasks" finds models flip answers 46% of the time on a challenge and lose 17% accuracy on average (nine LLMs in the first arXiv version), and that finetuning on synthetic data cuts the accuracy loss by 60% without resolving sycophancy entirely. FlipFlop is a robustness study, not a confidence estimator, but it legitimises "flip under challenge" as a measurable signal.
- **Sycophancy vs. uncertainty**: "Accounting for Sycophancy in Language Model Uncertainty Estimation" (Findings of NAACL 2025) proposes SyRoUP, a Platt-scaling variant conditioned on user-suggestion type, improving Brier Skill Score. MUSE ("It's Not Always Sycophancy", arXiv:2605.27288) shows yielding to pushback is more frequent when the model is uncertain, but a large share is pure sycophancy under certainty (e.g., Qwen 2.5 32B yields 52.3% overall vs. a 48.9% pure-sycophancy rate on MedXPertQA Diagnosis).\[18\]\[19\] **Your challenge probe therefore carries a model-specific sycophancy bias that the calibration step must absorb.**
- **Iterative prompting (Abbasi-Yadkori et al., NeurIPS 2024, "To Believe or Not to Believe Your LLM")** re-prompts with the model's own previous answers and derives a mutual-information lower bound on epistemic uncertainty\[20\]\[21\] — relevant both as a probe type and as a "bound-style" confidence.
- **Verbalized confidence**: Tian et al., "Just Ask for Calibration" (EMNLP 2023) find verbalized probabilities of RLHF models are better calibrated than token probabilities, with temperature scaling "often reducing expected calibration error (ECE) by over 50%" for ChatGPT, GPT-4 and Claude 2 on TriviaQA, SciQ, and TruthfulQA; considering multiple answer choices before stating confidence helps further. P(True) (Kadavath et al., 2022, arXiv:2207.05221) is the canonical self-evaluation probe.
- **Diverse prompts for accuracy, not calibration**: DivSampling ("Diversified Sampling Improves Scaling LLM Inference", arXiv:2502.11027) argues theoretically that responses from diverse prompts have lower error after Best-of-N than responses from a stationary prompt.\[22\]\[23\] This supports your intuition (diverse probes decorrelate errors) but concerns accuracy.

**Key novelty answer:** after a targeted search, no paper explicitly claims that heterogeneous probe types are more call-efficient than repeated self-consistency samples for *confidence calibration* at a matched number of calls. The nearest statements are SPUQ (perturbation + sampling beats unperturbed sampling on ECE, no matched-budget framing) and Xiong et al. (matched budget, mixed result). Frame your contribution as the first systematic matched-budget test of this hypothesis, and confront Xiong et al.'s AUROC result directly.

### 3. Shapley/attribution for ranking signals; sequential probing under budget

- We found **no LLM-confidence paper that uses Shapley values to rank or select probes/signals**. Existing Shapley work in LLM uncertainty uses it for other purposes: ShaQ (arXiv:2605.28170) localises which *input spans* cause aleatoric uncertainty;\[24\] Shapley NEAR attributes information gain across context sentences for hallucination detection; Watson et al. (NeurIPS 2023) explain predictive uncertainty with information-theoretic Shapley values.\[25\] Outside LLMs, Shapley-based feature selection (e.g., powershap/LLpowershap) is standard.\[24\]\[26\]\[27\]\[28\]
- Pedapati et al. rank features by logistic-regression importance — the most likely baseline a reviewer will ask you to compare Shapley against.
- Sequential/adaptive stopping exists only for i.i.d. samples: Adaptive-Consistency, ESC, CISC, and Conformal Language Modeling's calibrated stopping rule (below). Cascades (Gupta et al., ICLR 2024, "Language Model Cascades: Token-Level Uncertainty and Beyond") learn deferral rules from token-uncertainty quantiles\[29\]\[30\] but use white-box signals and model tiers, not prompt probes.
- **Reviewer critique to anticipate:** Shapley values measure *average* marginal contribution over all coalitions, and split credit among correlated signals. A sequential policy needs the *conditional* marginal value of the next probe given the probes already run, and ideally value per unit cost. Greedy forward selection, cost-weighted value-of-information, or a learned stopping policy are natural alternatives; you should show Shapley ordering matches or beats them, or justify Shapley for stability/interpretability.

### 4. Pessimistic/conservative confidence

- We found **no paper that aggregates LLM probe outcomes by min/worst-case or a lower confidence bound**. Prior aggregations are averages, weighted averages (BSDetector, MPC), logistic regression (Pedapati), or confidence-weighted votes (CISC). Loose analogues: low token-probability quantiles as deferral features (Gupta et al., ICLR 2024) and the mutual-information *lower bound* on epistemic uncertainty (Abbasi-Yadkori et al., NeurIPS 2024).\[21\]\[31\]
- This is a real, if modest, novelty point. Expect reviewers to ask whether "pessimistic" helps discrimination (AUROC) or merely shifts scores (fixable by any monotone recalibration — and conformal prediction is invariant to monotone transforms of the score). You should show that min-aggregation changes the *ranking* of examples usefully, not just the scale.

### 5. Conformal prediction for LLMs

- **Conformal prediction for MCQA** (Kumar et al., arXiv:2305.18404, MMLU) uses softmax scores as nonconformity scores and connects prediction-set size to selective classification.\[32\]\[33\]
- **Conformal Language Modeling (Quach et al., ICLR 2024)** calibrates a *stopping rule* for sampling and a *rejection rule* via Learn-Then-Test (LTT), guaranteeing the sampled set contains at least one acceptable answer with high probability.\[34\]\[35\] **This is the most important conformal precedent for your design, because it already shows how to calibrate adaptive stopping with guarantees.**
- **LofreeCP / "API Is Enough" (Su et al., 2024, arXiv:2403.01216)** does conformal prediction without logits using sample frequency (out of 10 samplings), and notes frequency-only scores concentrate on ties, fixed by adding normalized entropy and semantic similarity. Evaluated on TriviaQA, WebQuestions, MMLU.\[32\]
- **ConU (Wang et al., Findings of EMNLP 2024)** builds a self-consistency-based uncertainty criterion into conformal correctness-coverage guarantees for open-ended generation.\[36\]
- **Conformal factuality (Mohri & Hashimoto, ICML 2024)** filters claims so the retained output is correct with high probability, "requires very few human-annotated samples"; Cherian, Gibbs & Candès (NeurIPS 2024) add conditional guarantees.\[37\]\[38\]
- **Conformal Alignment (Gui, Jin & Ren, NeurIPS 2024)** selects which outputs to trust with guarantees — the relevant framing if "top-k selection" means *choosing which queries/answers to accept* rather than building answer sets.
- Recent black-box variants use frequency-based nonconformity with self-consistency samples, e.g., "Conformal Cascade" (arXiv:2607.25018) for multi-tier inference and "Black-Box Reliability Certification for AI Agents" (arXiv:2602.21368).\[39\]\[40\]
- **Requirements:** split conformal guarantees marginal coverage ≥ 1−α if calibration and test points are exchangeable and the score function is fixed independently of the calibration data. Coverage is marginal (not per-question), and is broken by API model updates, prompt-template changes, or domain shift. With n calibration points, coverage is at most 1−α+1/(n+1), and realized coverage for a particular calibration draw fluctuates with standard deviation roughly √(α(1−α)/n) (about 1.3 points for α=0.1, n=500); risk-control methods (conformal risk control, LTT) need larger n for tight bounds because they control expected loss or high-probability risk across a grid of thresholds.

## Details: Reviewer Critiques of the Design and How to Pre-empt Them

**1. The 50/50 split.**
- Signal collection/Shapley ranking on split A and conformal calibration on split B is valid in principle: the ranking and the scoring function are learned on A and frozen, so they are independent of B. Conformal validity then requires only B ∪ test exchangeable.
- But you need a **third, untouched test split C** (or repeated random B/C partitions, e.g., 100 random calibration/test splits, as is standard in conformal evaluation). If coverage is reported on B itself, the guarantee is not demonstrated.
- If anything else is tuned on B (pessimism parameter, budget, stopping threshold, α choice), either tune it on A or use LTT/Bonferroni over the grid.
- Data efficiency: report a learning curve over |A| and |B| (e.g., 10%–70%) and show results are not specific to 50/50; consider cross-fitting (K-fold Shapley estimation with out-of-fold scores) for the ranking stage. Nested CV is relevant only for the ranking/calibrator hyperparameters, not for the conformal step.

**2. Conformal validity with learned ordering and adaptive stopping.**
- If the entire pipeline (probe order, stopping rule, pessimistic score) is a fixed function applied identically to calibration and test items, the final score is just a complicated nonconformity score and split-conformal coverage holds — adaptive stopping does not by itself break exchangeability.
- It breaks if stopping thresholds are chosen using B, if stopping depends on test-time information unavailable at calibration (e.g., different budgets per query set by a user), or if probe order is re-learned online.
- If the stopping threshold and the conformal threshold are both calibrated, follow Quach et al. and calibrate jointly with LTT.
- Also state the guarantee precisely: marginal coverage over exchangeable draws, not conditional on difficulty; report conditional coverage by difficulty/topic bins as an empirical check.

**3. Fair cost comparison.**
- Equal-*call* comparisons favour challenge/negation probes, because multi-turn challenges re-send the question plus previous answer (more input tokens), and sequential probing cannot be parallelised the way SC samples can (higher latency). Conversely, SC with CoT can generate long outputs.
- Report three budget axes: calls, total input+output tokens (or dollar cost at a stated price), and wall-clock latency.
- Show the core plot as "metric vs. budget" with SC@k, Adaptive-Consistency/ESC, and your method on the same axes.
- Define an **"SC-equivalent sample count"**: for your method at budget B, the k at which SC@k matches its AUROC/ECE. This turns your headline claim into a single, reviewable number.

**4. Signal validity.**
- Challenge-flip signals are confounded by sycophancy, which varies by model and by alignment version (MUSE; SyRoUP). Show per-model results, and test on at least one model with low sycophancy.
- Negation is well-defined for binary/MCQ tasks but ambiguous for open-ended answers. Specify how semantic equivalence is judged (exact match, NLI, LLM judge), because judge calls are extra hidden cost.

**5. Baselines (minimum credible set).**
- SC@k at matched k.
- Adaptive-Consistency and ESC.
- Semantic entropy, including the Bayesian variant.
- SelfCheckGPT (NLI).
- Verbalized/Top-K confidence (Tian et al.) and P(True).
- Xiong et al.'s hybrid and misleading-hint consistency.
- SPUQ.
- Pedapati et al.'s feature + logistic-regression model trained on the same split A.
- MPC and BSDetector.
- CISC.
- For the conformal part: LofreeCP and ConU.
- Ablations: random probe order vs. Shapley order vs. greedy forward selection; mean vs. min aggregation; probes-only vs. probes + SC samples.

**6. Benchmarks and metrics.**
- Use the standard mix: TriviaQA, SciQ, TruthfulQA (Tian et al.); GSM8K/SVAMP (Adaptive-Consistency, BSDetector);\[41\] MMLU/MedMCQA (Kumar et al., LofreeCP); CoQA and NQ (semantic entropy, LM-Polygraph).
- Metrics:
  - ECE with fixed binning plus an adaptive or binning-robust variant.
  - Brier score.
  - AUROC and AUPRC.
  - AURC/AURAC or Prediction Rejection Ratio (LM-Polygraph, TACL 2025).\[42\]
  - For conformal: empirical coverage, average set size, singleton rate, and coverage across ≥100 random splits.

## Recommendations

**(a) Closest papers to position against, in order:**
1. Xiong et al. (ICLR 2024).
2. Pedapati et al. (arXiv:2406.04370).
3. SPUQ (EACL 2024).
4. Conformal Language Modeling (Quach et al., ICLR 2024).
5. MPC (arXiv:2402.11279) and BSDetector (ACL 2024).
6. Adaptive-Consistency/ESC/CISC, for the budget framing.
7. LofreeCP/ConU, for black-box conformal.

**(b) What appears genuinely novel:** the matched-budget hypothesis that a heterogeneous, ordered probe set replaces k SC samples for *calibration*; Shapley-ranked sequential probing with early stopping; pessimistic min/worst-case aggregation over probes; and the integration of all three with conformal selection under a call budget.

**(c) Risks to the novelty claim:**
- Xiong et al.'s matched-budget results (misleading hints lose on AUROC).
- Pedapati et al. already "collect black-box perturbation signals, learn a calibrator".
- CLM already calibrates adaptive stopping conformally.
- Shapley ordering may be seen as an arbitrary choice unless it beats greedy/VOI ordering.
- Pessimism may be dismissed as a monotone rescaling unless it changes rankings.
- Probe signals may be model-specific and could degrade as vendors reduce sycophancy.

**(d) Experiments that would most strengthen the paper:**
1. Budget curves (calls, tokens, latency) against SC@k and Adaptive-Consistency, with a headline "SC-equivalent sample count".
2. An explicit replication of Xiong et al.'s M=5 setup showing where and why your heterogeneous probe set overturns their AUROC result.
3. A three-way split (or repeated splits) with coverage reported only on held-out data, plus |A|/|B| learning curves.
4. LTT-calibrated stopping if stopping thresholds are learned.
5. Ordering ablations (random/Shapley/greedy/cost-weighted VOI) and aggregation ablations (mean/min/learned).
6. Per-model sycophancy analysis and cross-model transfer of the probe ranking.
7. At least one open-ended task (TriviaQA/NQ with semantic matching) in addition to MCQ/math.

## Caveats

- The literature moves quickly; a broad search found no paper making the matched-budget probe-efficiency claim, no Shapley-ranked probe ordering, and no min-aggregation over probes, but absence from a search is not proof of absence — re-check arXiv just before submission.
- Several numbers above come from abstracts or secondary summaries rather than full-table verification: SPUQ's 30–70% ECE reduction, SelfCheckGPT's flattening of gains with more samples, and BSDetector specifics. Verify them against the PDFs before quoting.
- Several 2026 arXiv preprints cited (MUSE, ShaQ, Conformal Cascade) are not yet peer-reviewed.
- FlipFlop's model count differs between arXiv versions (nine vs. ten LLMs).\[43\]

## Sources

1. [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0)
2. [Detecting hallucinations in large language models using semantic entropy](https://ora.ox.ac.uk/objects/uuid:0653d09e-9368-4eb1-98bb-50d9dda7d3e5/files/r0g354g31t)
3. [Hallucination Detection on a Budget: Efficient Bayesian Estimation of Semantic Entropy](https://www.alphaxiv.org/abs/2504.03579)
4. [SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection](https://arxiv.org/pdf/2303.08896)
5. [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://www.alphaxiv.org/abs/2303.08896)
6. [\[Quick Review\] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://liner.com/review/selfcheckgpt-zeroresource-blackbox-hallucination-detection-for-generative-large-language-models)
7. [can llms express their uncertainty? an empirical evaluation ...](https://arxiv.org/pdf/2306.13063)
8. [Can LLMs Express Their Uncertainty? An E...](https://axi.lims.ac.uk/paper/2306.13063)
9. [Efficient semantic uncertainty quantification in language models via diversity-steered sampling](https://openreview.net/forum?id=IiEtQPGVyV)
10. [Efficient semantic uncertainty quantification in language models via diversity-steered sampling — Lacuna](https://lacuna.tiptreesystems.com/work/efficient-semantic-uncertainty-quantification-in-language-models-via-diversity/wrk_549c5ca2c521fde7329cb9143bedc94b)
11. [Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs](https://arxiv.org/pdf/2305.11860)
12. [License: arXiv.org perpetual non-exclusive license](https://arxiv.org/html/2401.10480v1)
13. [Strength in Numbers: Estimating Confidence of Large Language Models by Prompt Agreement - Johns Hopkins University](https://pure.johnshopkins.edu/en/publications/strength-in-numbers-estimating-confidence-of-large-language-model/)
14. [Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](https://arxiv.org/html/2512.11998v1)
15. <https://arxiv.org/pdf/2406.04370v4>
16. [A Survey of Calibration Process for Black-Box LLMs](https://arxiv.org/pdf/2412.12767)
17. <https://arxiv.org/html/2402.11279>
18. [It's Not Always Sycophancy: Measuring LLM Conformity as a Function of Epistemic Uncertainty](https://arxiv.org/pdf/2605.27288)
19. [Accounting for Sycophancy in Language Model Uncertainty Estimation \[Quick Review\]](https://liner.com/review/accounting-for-sycophancy-in-language-model-uncertainty-estimation)
20. [\[Literature Review\] To Believe or Not to Believe Your LLM](https://www.themoonlight.io/en/review/to-believe-or-not-to-believe-your-llm)
21. [\[PDF\] To Believe or Not to Believe Your LLM](https://www.semanticscholar.org/paper/To-Believe-or-Not-to-Believe-Your-LLM-Abbasi-Yadkori-Kuzborskij/5322cd631e69ae484038b13ac320194afaccdc3b)
22. [Diversified Sampling Improves Scaling LLM inference](https://arxiv.org/html/2502.11027v1)
23. [On the Effect of Sampling Diversity in Scaling LLM Inference](https://proceedings.mlr.press/v337/wang26f.html)
24. [Localizing Input Uncertainty Quantification forLarge Language Models via Shapley Values](https://arxiv.org/html/2605.28170)
25. [Explaining Predictive Uncertainty with Information Theoretic Shapley Values](https://arxiv.org/pdf/2306.05724)
26. [Fact or Hallucination? An Entropy-Based Framework for ...](https://openreview.net/pdf/abcb097be6511e5d7cbc0fb220c8553ff9d29738.pdf)
27. [Explaining Predictive Uncertainty with Information Theoretic Shapley Values](https://proceedings.neurips.cc/paper_files/paper/2023/file/16e4be78e61a3897665fa01504e9f452-Paper-Conference.pdf)
28. [LLpowershap: logistic loss-based automated Shapley values feature selection method](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11515487/)
29. [LLM Cascade with Multi-Objective Optimal Consideration](https://arxiv.org/html/2410.08014v1)
30. [Language Model Cascades: Token-Level Uncertainty And Beyond](https://research.google/pubs/language-model-cascades-token-level-uncertainty-and-beyond/)
31. [Language Model Cascades: Token-level uncertainty and beyond Neha Gupta\*](https://arxiv.org/pdf/2404.10136)
32. [API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access](https://arxiv.org/html/2403.01216v1)
33. [Bhawesh Kumar on LinkedIn: Conformal Prediction with Large Language Models for Multi-Choice Question…](https://www.linkedin.com/posts/bhaweshiitk_conformal-prediction-with-large-language-activity-7076655693658779648-cpJW?trk=public_profile_like_view)
34. [Conformal Language Modeling](https://research.google/pubs/conformal-language-modeling/)
35. [\[2306.10193\] Conformal Language Modeling](https://arxiv.org/abs/2306.10193)
36. [ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees](https://arxiv.org/html/2407.00499v1)
37. [Language models with conformal factuality guarantees](https://dl.acm.org/doi/10.5555/3692070.3693538)
38. [Large language model validity via enhanced conformal prediction methods](https://arxiv.org/html/2406.09714v1)
39. [Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration](https://arxiv.org/html/2602.21368)
40. [Conformal Cascade: Distribution-Free Accuracy Guarantees for Multi-Tier LLM Inference](https://arxiv.org/pdf/2607.25018)
41. [Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs](https://blog.athina.ai/let-s-sample-step-by-step-adaptive-consistency-for-efficient-reasoning-and-coding-with-llms)
42. [arXiv:2406.15627v4 \[cs.CL\] 30 Jun 2025](https://arxiv.org/pdf/2406.15627)
43. [Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment](https://paperreading.club/page?id=194068)
