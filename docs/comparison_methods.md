---
title: "Methods for comparison: how each one ranks"
geometry: "landscape,margin=0.7in"
fontsize: 10pt
---

Source: `src/confledger/baselines/`, `ranking.py`, `evaluation_protocol.baseline_policies`,
`docs/paper/comparison_methods.tex`. Call accounting is in `docs/call_accounting.pdf`.

# 1. The shared ranking procedure

The task is to hand over the $K$ cases whose own answers are most likely correct ($K$ = top 10% of a batch).
Every method, including CEL-$k$, goes through the same four steps. Methods differ only in steps 1 and 2.

1. **Answer.** Spend the method's calls on the case and return one answer $\hat Y$ (usually the majority of
   the draws).
2. **Score.** Compute a *score key* for the case: a number, or a short tuple compared left to right (the
   second entry only breaks ties in the first). Larger means more confident.
3. **Sort and pick.** Sort the batch by key, descending, and take the first $K$. Cases with equal keys are
   tied; ties at the cut-off are broken uniformly at random with a fixed seed.
4. **Score the pick.** Precision@10% is the share of the $K$ picked cases whose *own* answer $\hat Y$ is
   correct ($Z = 1\{\hat Y = Y\}$). AUROC and AUARC use the same keys over the whole batch. Gold labels
   appear only here.

Notation: a case has $n$ usable draws; $m$ of them agree with the majority answer, so $m/n$ is the
agreement share; $c_i \in [0,100]$ is the verbalized confidence on draw $i$ (free in the main configuration).
A failed extraction counts as a draw that disagrees.

# 2. What each method returns and ranks by

| Method | Draws and calls | Answer returned | Score key (larger = ranked higher) | Why it should rank well |
|:--------------|:----------------------|:---------------------|:---------------------------------------|:-----------------------------|
| Random | 1 draw | that draw's answer | none: all cases tie, so the order is random | chance level; exact hypergeometric null |
| Verbalized | 1 draw | that draw's answer | $(c)$, the stated confidence; equal $c$ ties | the model's own claim of confidence |
| Self-consistency | $k$ same-prompt draws | majority | $(m/n)$ | answers that survive resampling are more often right |
| Self-consistency + verbal | $k$ same-prompt draws | majority | $(m/n,\ \bar c)$, $\bar c$ = mean $c_i$ | control: same draws, but also reads the free confidences |
| Semantic entropy | $k-1$ draws, 1 judge call | majority | $(-H,\ m/n)$, $H$ = entropy of the reason groups | consistent *reasoning*, not only consistent answers |
| BSDetector | $k-1$ draws | majority | $(0.7\,m/n + 0.3\,c_1/100)$ | blends observed consistency with stated confidence |
| Avg-Conf (self-random) | $k$ same-prompt draws | majority $\tilde Y$ | $(A,\ m/n)$, $A = \sum_{i:\,\hat Y_i=\tilde Y} c_i \,/\, \sum_i c_i$ | confidence mass that backs the majority |
| Avg-Conf (prompting) | 1 generator call, $k-1$ draws, one paraphrase each | majority $\tilde Y$ | $(A,\ m/n)$ | same, with varied wording |
| CISC | $k$ same-prompt draws | confidence-weighted vote | $(W)$, $W$ = weight share of the winning answer | high-confidence draws count more in the vote and the score |
| Black-box calibrator | $k$ draws; from $k=5$ add 1 generator call and paraphrase draws | majority over all draws | $(p,\ m/n)$, $p$ = fitted $P(Z=1)$ | learned map from agreement and wording similarity to correctness |
| **CEL-$k$** | plan chosen on dev, mean $\le k$ calls: screen, then more evidence for survivors | majority of all draws used | $(s)$, $s=\hat\mu-\hat\lambda\hat\sigma$ from a fitted score model (Section 4) | combines stability, probes, challenge, entropy and free text signals, and discounts thin evidence |

# 3. Mechanism of each score

**Random.** The case's single answer is returned, and the key is a constant, so the picked set is a uniform
sample of the batch. Its precision is the base accuracy.

**Verbalized.** One response gives an answer and a stated confidence $c$. Cases are sorted by $c$. Models
state coarse values (many 90s and 95s), so large tie groups form and are broken at random.

**Self-consistency.** Sample $k$ draws of the original prompt. The answer is the majority, the score is
$m/n$: a case where all $k$ draws agree ranks above one where $k-1$ do. With $k$ draws there are only $k$
distinct scores, so ties are common at small $k$.

**Self-consistency + verbal.** Same draws and the same answer. It ranks by $m/n$ first and uses the mean
confidence $\bar c$ only to order cases with equal $m/n$. This is the fair control, because CEL also reads
the free confidences.

**Semantic entropy.** Take $k-1$ same-prompt draws, each with a rationale. One judge call groups the
rationales by underlying reason and returns a group label per draw. $H$ is the Miller--Madow entropy of the
label counts: 0 if all rationales give the same reason, larger if they scatter. The key is $-H$, so
consistent reasoning ranks first. Answer-level ties are broken by $m/n$. It needs at least two draws.

**BSDetector.** A fixed blend: $0.7\cdot m/n + 0.3\cdot c_1/100$, where $c_1$ is the stated confidence of the
case's verbalized response. We reconstruct it from cached draws, with $c_1$ standing in for the original
self-reflection prompt. The weight 0.7 is our choice and is not fitted.

**Avg-Conf.** Take the majority answer $\tilde Y$. $A$ is the share of the total stated confidence carried by
the draws that agree with $\tilde Y$. Unlike $m/n$ it weights each agreeing draw by how sure it was, so
three confident votes for $\tilde Y$ can beat four hesitant ones. Ties are broken by $m/n$. The two variants
differ only in how the $k$ samples are produced: same prompt (self-random) or one paraphrase per sample
(prompting).

**CISC.** Each draw gets weight $w_i = \mathrm{softmax}(c_i/T)$ over the case's draws. The answer is the one
with the largest total weight, and the score $W$ is that total, scaled down for failed extractions. $T$ is
fitted on the development slice to maximise the vote's accuracy over 80 values in $[10^{-4}, 10^{4}]$. A large
$T$ gives equal weights, which is self-consistency. A small $T$ lets the single most confident draw decide.
CISC defines an answer rather than a selection score, so ranking by $W$ is our adaptation, and its answer can
differ from the majority.

**Black-box calibrator.** For each block of draws it computes two features: the number of distinct answers
and the mean pairwise ROUGE-L of the full responses (answer plus explanation). Below $k=5$ there is one block
(same-prompt draws). From $k=5$ there are two (same-prompt and paraphrase draws), giving four features. The
features are standardised, and an L2 logistic regression of $Z$ on them is fitted on the development slice.
The key is the predicted probability $p$, ties by $m/n$, and the answer is the majority over all draws.

# 4. CEL-$k$: how its score is produced

CEL-$k$ differs from the baselines in two ways. It spends its calls unevenly (a cheap screen, then more evidence
only for cases that pass), and its score is a *fitted* estimate of $P(Z=1)$ from many signals, discounted by how
thin the evidence is. Source: `methods.cel_alloc`, `score_model.py`, `docs/method.md` §2.

**Step 0: choose the plan (dev slice, once per budget $k$).** A plan is $a=(k_1,\text{para},e,F,\text{chal},\text{se})$:
screen size, whether the screen uses fresh paraphrases, extra draws $e$, probes $F$, challenge, rationale
entropy. Among plans whose expected cost is within one call of $k$, take the one with the best dev prec@$K$.
Nothing here touches evaluation items.

**Step 1: run the plan on each case.**

- *Screen:* $\text{para}$ generator call plus $k_1$ draws. If the draws do not all agree the case is
  *eliminated*: it stops, its answer is the screen majority, and it keeps only the screen's evidence.
- *Survivors* (unanimous screen) buy $e$ more draws, the probes (negation, relabel, distractor: pass/fail),
  and optionally the challenge and the entropy call. The answer is the majority of all draws used.

**Step 2: turn the evidence into a feature row $x$.** A signal that was not bought is marked missing.

| Feature | Meaning | Costs a call? |
|:--------------------|:-----------------------------------------------------|:--------|
| intercept | constant | no |
| $\eta$ (stability) | expected log-odds of the case's agreement rate: $\psi(\tfrac12+m)-\psi(\tfrac12+n-m)$ for $m$ of $n$ draws agreeing (Jeffreys prior) | no |
| survived | 1 if the case passed the screen | no |
| neg, relabel, distractor | probe passed $=+1$, failed $=-1$ | yes |
| chal | challenge held $=+1$, flipped $=-1$ | yes |
| verb | $(\bar c-50)/50$, mean stated confidence of the draws used | no |
| $-\text{se}$ | negative rationale entropy | yes (judge) |
| length, hedge, neg_drop | log words in rationales, share of hedged rationales, confidence lost under negation | no |
| p_first | mean probability of the answer token; gray-box, only if `scoring.gray_box` | no |

**Step 3: the score model gives $\hat\mu$.** On the labelled fitting split $\mathcal{D}_{\rm fit}$, replay the
same plan and fit a ridge logistic regression of $Z=1\{\hat Y=Y\}$ on $x$. This is the only place labels enter.
The coefficient of $\eta$ is unpenalised and constrained to be $\ge 0$ (more agreement may not lower
confidence). Then
$$\hat\mu(E)=\operatorname{sigmoid}(\hat\beta^{\top}x),$$
with any missing feature filled by its $\mathcal{D}_{\rm fit}$ mean. $\hat\mu$ estimates $P(Z=1\mid E)$.

**Step 4: the uncertainty $\hat\sigma$.** The log-odds variance of $\hat\mu$ adds three parts:

1. fit uncertainty: $x^{\top}\hat\Sigma x$, from the inverse penalised Hessian (delta method);
2. stability uncertainty: $\hat\beta_\eta^{2}\,\mathrm{Var}(\eta)$, which shrinks like $1/n$ as a case gets more draws;
3. unbought signals: $\sum_{g\ \text{missing}} \hat\beta_g^{2}\,\mathrm{Var}(x_g)$, using $\mathcal{D}_{\rm fit}$ variances.

Then $\hat\sigma=\hat\mu(1-\hat\mu)\sqrt{\text{variance}}$. A case with fewer calls has more missing signals and fewer
draws, so a larger $\hat\sigma$.

**Step 5: the ranking score.**
$$s=\hat\mu-\hat\lambda\,\hat\sigma,$$
with $\hat\lambda\in\{0,0.5,1,1.5,2,3\}$ chosen by 5-fold cross-validation on $\mathcal{D}_{\rm fit}$ to maximise held-out
prec@$K$. Ties go to the smaller $\hat\lambda$, so $\hat\lambda=0$ (rank by $\hat\mu$) is allowed. The batch is
sorted by $s$ and the top $K$ delivered, with random tie-breaks. There is no hard rule that eliminated cases rank
below survivors; it follows from their lower $\eta$, the survived flag, and larger $\hat\sigma$.

**What is exact and what is not.** The score is model-based: nothing in it is a guarantee. When every case has
the same evidence, $\hat\sigma$ is nearly a function of $\hat\mu$ and $\hat\lambda$ cannot reorder the top $K$;
the pessimism only matters when evidence is uneven, as it is here. Guarantees come from the optional conformal
layer applied on top of the frozen score.

**Example of two feature rows** (stability terms computed; coefficients are fitted, so they are left symbolic).

| | Case A: survivor | Case B: eliminated |
|:------------------|:---------------------------|:------------------------------|
| Draws | screen 3 of 3 agree, then 3 more all agree: $m=n=6$ | screen 2 of 3 agree: $m=2$, $n=3$ |
| $\eta$, $\mathrm{Var}(\eta)$ | $3.76$, $5.10$ | $0.67$, $1.43$ |
| survived | 1 | 0 |
| probes, challenge, SE | bought, entered as $\pm1$ and $-\text{se}$ | not bought: filled with $\mathcal{D}_{\rm fit}$ means; each adds $\hat\beta_g^2\mathrm{Var}(x_g)$ to $\hat\sigma^2$ |
| Result | large $\hat\mu$, small extra variance | small $\hat\mu$, large $\hat\sigma$ |

# 5. Worked example (baselines)

One case, five same-prompt draws. Answers: Yes, Yes, Yes, No, Yes. Confidences: 90, 80, 95, 60, 85.
The majority is Yes, so $m/n = 4/5 = 0.8$ and $\bar c = 82$.

| Method | Computation | Key |
|:----------------------|:-------------------------------------------|:-------------|
| Self-consistency | $4/5$ | 0.800 |
| Self-consistency + verbal | $(4/5,\ 82)$ | (0.800, 82) |
| Avg-Conf | $(90+80+95+85)\,/\,(90+80+95+60+85) = 350/410$ | (0.854, 0.800) |
| BSDetector, $c_1 = 90$ | $0.7\cdot 0.8 + 0.3\cdot 0.90$ | 0.830 |

A second case with draws Yes, Yes, Yes, Yes, No and confidences 70, 65, 70, 60, 90 has the same $m/n$, so
self-consistency ties the two. Self-consistency + verbal puts it below (mean 71 vs 82), and Avg-Conf gives
$265/355 = 0.746$, also below. That is the extra information the confidence-aware methods use.

# 6. Departures from the source papers

- On a yes/no answer the number of distinct answers is 1 or 2, and ROUGE over answers alone is a function of
  $m$, so the calibrator computes ROUGE over full responses to keep the explanation's wording in play. We call
  it a black-box calibrator, not Pedapati et al.'s method, because it omits the perturbations that need new
  prompts or a local NLI model (sentence permutation, entity-frequency amplification, stop-word removal,
  split-response consistency).
- We omit Xiong et al.'s misleading-hint sampling and Top-$K$ and Pair-Rank aggregation (Pair-Rank reduces to
  the top-2 probabilities on a binary question), and CISC's P(True) confidence, since all need new prompts.
- CISC with the answer token's probability in place of the length-normalised response probability is a
  gray-box variant, reported separately from the black-box comparisons.

# 7. Open check: BSDetector's call count

`bsdetector_like` (`baselines/others.py:30`) charges its draws plus one reflection call, but the reflection
call is 0 when `verbal_per_draw` is on, and `baseline_policies` passes $B-1$ draws expecting a total of $B$.
In the main configuration BSDetector therefore spends $k-1$ calls, one fewer than the other baselines. With
`verbal_per_draw` off it spends $k$. Either state this in the paper or pass $B$ draws when it is on.
