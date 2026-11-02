---
title: "CEL: the two operating modes"
date: "2026-10-13"
---

Notation follows `problem_setup.md`. Constants are the paper's (`sprt_posterior.py`,
pinned by `tests/test_sprt_posterior.py`). Each statement is labelled **exact** or
**model-based**.

# 1. Common ground

**Cases and target.** A test batch $\mathcal{D}_{\mathrm{test}} = \{X_j\}_{j=1}^{m}$
($m = 50$) of binary questions, a black-box model $f$, and a selection size $K$, the top $10\%$ of the batch ($K = 5$).
For case $j$, $\hat Y_j$ is the majority answer of its draws and
$Z_j = \mathbb{1}\{\hat Y_j = Y_j\}$ its correctness, the only target.

**Ledger.** Every call on case $j$ appends a pass/fail outcome to its evidence $E_j$:

- *draws* on fresh paraphrases, $A_{j1}, A_{j2}, \dots \in \{0,1\}$ (agree with $\hat Y_j$
  or not). With one paraphrase per draw they are exactly i.i.d.
  $\mathrm{Bern}(\bar p_j)$, where $\bar p_j$ is the case's stability (Lemma 1);
- *probes* $g \in \mathcal{F} = \{\text{negation}, \text{relabel}, \text{distractor}\}$, each
  pass/fail;
- *other signals*: challenge (1 call), rationale entropy (1 judge call), and verbal
  confidence and text signals (free when they ride on the draws).

One generator call per case produces the paraphrases.

**Score model** (model-based). On the labelled fitting split $\mathcal{D}_{\mathrm{fit}}$,
replayed under the same policy, a ridge logistic regression of $Z$ on features $x(E)$, with
the coefficient of the stability term constrained to be non-negative, gives

$$
\hat\mu(E) = \sigma\!\big(\hat\beta^{\top} x(E)\big), \qquad
s(E) = \hat\mu(E) - \hat\lambda\, \hat\sigma(E),
$$

where $\hat\sigma$ combines the fit's uncertainty, the stability term's posterior variance,
and the variance of unbought signals. $\hat\lambda \ge 0$ is chosen by cross-validated
prec@$K$ on $\mathcal{D}_{\mathrm{fit}}$; $\hat\lambda = 0$, ranking by $\hat\mu$, is
allowed. $\hat\mu$ is an estimate of $P(Z = 1 \mid E)$; $s$ is a ranking score.

**Conformal layer** (exact; optional in either mode). With the calibration split
$\mathcal{D}_{\mathrm{cal}}$ replayed under the same policy,

$$
p_j = \frac{1 + \#\{i \in \mathcal{D}_{\mathrm{cal}} : Z_i = 0,\ s_i \ge s_j\}}{n_{\mathrm{cal}} + 1},
$$

and Benjamini–Hochberg at level $\alpha$ on $(p_j)_{j \le m}$ returns a set
$\mathcal{S}_\alpha$ with $\mathrm{FDR}(\mathcal{S}_\alpha) \le \alpha$ (Jin & Candès 2023).
The guarantee is on correctness and marginal over calibration draws. It requires each
case's calls to depend only on that case's own outcomes, so that calibration and test scores
are exchangeable. Both modes below satisfy this; the frontier allocation, whose calls depend
on the rest of the batch, does not.

# 2. CEL-k: ranking mode (the budget is fixed)

**Input.** A budget $\bar k$: calls per case, averaged over the batch.

**Plans.** A plan $a = (k_1, \mathrm{para}, e, F, \mathrm{chal}, \mathrm{se})$ screens every
case with $k_1$ draws. A case whose screen is unanimous (it *survives*) gets $e$ more draws,
the probes $F \subseteq \mathcal{F}$, and optionally the challenge and the rationale entropy.
Case $j$ costs

$$
C_j(a) = \mathrm{para} + k_1 + \mathbb{1}\{\text{$j$ survives}\}\,\big(e + |F| + \mathrm{chal} + \mathrm{se}\big),
$$

with expected cost $\bar C(a) = \mathrm{para} + k_1 + \pi_{k_1}\,(e + |F| + \mathrm{chal} + \mathrm{se})$,
where $\pi_{k_1}$ is the screen's survival rate.

**Plan choice** (on dev only). Over the replayable plan space $\mathcal{A}$, with
$\widehat C_{\mathrm{dev}}$ and $\widehat{\mathrm{prec}}_{\mathrm{dev}}$ measured on the
dev slice,

$$
\hat a(\bar k) = \arg\max_{a \in \mathcal{A}:\ \bar k - \delta < \widehat C_{\mathrm{dev}}(a) \le \bar k}\ \widehat{\mathrm{prec@}K}_{\mathrm{dev}}(a),
$$

with ties going to the plan that spends more. The budget is a target, not only a cap: the
plan must spend within $\delta$ (default 1 call) of $\bar k$. If no plan falls in that
band, $\hat a(\bar k)$ is the best of the plans that spend the most without exceeding
$\bar k$. The dev score ranks by Borda over the signals, the
label-light criterion the plan space was designed with.

**Output.** Fit $(\hat\beta, \hat\lambda)$ on $\mathcal{D}_{\mathrm{fit}}$ under
$\hat a(\bar k)$, run the plan on the batch, and deliver

$$
\mathcal{S}_{\bar k} = \operatorname{top-}K_j\ s(E_j) \qquad (|\mathcal{S}_{\bar k}| = K,\ \text{random tie-breaks}).
$$

**Properties.**

- *Budget*: $\tfrac1m \sum_j C_j(\hat a) \approx \widehat C_{\mathrm{dev}}(\hat a) \le \bar k$,
  in expectation, not per case.
- *Optimality* (model-based): if accuracy is monotone in $s$ (Assumption 1), the top $K$
  maximise the expected number of correct selections (Prop. 1).
- *Per-answer guarantee*: none. For a set-level guarantee, add the conformal layer.

# 3. Certified CEL: adaptive mode (the evidence standard is fixed)

**Input.** An evidence standard $(\varepsilon_0, p_0^\ast) = (0.05, 0.8)$ and the SPRT
$(p_0, p_1, \alpha_s, \beta_s) = (0.7, 0.95, 0.1, 0.1)$.

**Certificate** (exact). Under the Jeffreys prior $\bar p \sim \mathrm{Beta}(\tfrac12, \tfrac12)$,
after $m$ of $n$ draws agree,

$$
\varepsilon(m, n) = P\big(\bar p < p_0^\ast \mid m \text{ of } n\big) = I_{p_0^\ast}\!\big(m + \tfrac12,\ n - m + \tfrac12\big).
$$

The certification depth $d^\ast = \min\{d : \varepsilon(d, d) \le \varepsilon_0\} = 9$
(Cor. 1; $\varepsilon(9,9) = 0.042$).

**Policy, per case** (the truncated SPRT):

1. One generator call, then a screen of $s_1 = 3$ draws.
2. If the screen is not unanimous, stop: the case costs 4 calls and is ranked by its screen
   agreement.
3. Otherwise run stage 2: draws up to the depth $d^\ast$ (6 more), the probe battery
   $\mathcal{F}$, the rationale entropy and the challenge. A survivor costs 15 calls (16 with a
   separate verbal call).

The spend is set by the data, with expectation
$\mathbb{E}[C] = 4 + \pi_3 \cdot 11$ (or $\cdot 12$), where $\pi_3 = \mathbb{E}[\bar p^{3}]$.
At the paper's survival rates the mean is about 9–13 calls.

**Output.**

$$
\mathcal{S}_{\mathrm{cert}} = \{\, j : \text{$j$ survives},\ m_j = n_j,\ \varepsilon(m_j, n_j) \le \varepsilon_0 \,\},
$$

each delivered answer with its ledger statement. The batch is also ranked: by the cascade
key, or by $s(E)$ fitted under this policy. With the conformal layer on that score, it also
delivers $\mathcal{S}_\alpha$.

**Properties.**

- *Per-answer guarantee* (exact, given Lemma 1): for every $j \in \mathcal{S}_{\mathrm{cert}}$,
  $P(\bar p_j < p_0^\ast \mid E_j) \le \varepsilon_0$. This is a statement about stability,
  not correctness: a stable answer can be wrong (Lemma 4). The probes, and the conformal
  layer on $s$, are what reach correctness.
- *Model-based variant*: counting each passed probe as $\kappa_g$ draws (Prop. 3) certifies
  earlier, but only under that model.
- *Set-level guarantee*: $\mathrm{FDR}(\mathcal{S}_\alpha) \le \alpha$ (exact), as in §1.

# 4. How the modes relate

| | CEL-k | Certified CEL |
|---|---|---|
| Fixed | budget $\bar k$ | standard $(\varepsilon_0, p_0^\ast)$ |
| Calls per case | $\le \bar k$ on average | 4 or 15–16 (data-dependent) |
| Delivers | $\mathcal{S}_{\bar k}$, exactly $K$ | $\mathcal{S}_{\mathrm{cert}}$, 0 to $m$ |
| Per-answer guarantee | none | stability certificate (exact) |
| Set-level FDR (optional) | $\mathcal{S}_\alpha$ on $s$ (exact) | $\mathcal{S}_\alpha$ on $s$ (exact) |
| Use | cost-capped triage | auto-accept with an evidence record |

- **Nesting.** Certified CEL's policy is one plan of CEL-k's space,
  $a_{\mathrm{CEL}} = (3, \text{para}, 6, \mathcal{F}, 1, 1) \in \mathcal{A}$. CEL-k at
  $\bar k = \bar C(a_{\mathrm{CEL}})$ can therefore replay it, and the budget curve meets
  certified CEL's point there.
- **Certification budget** (Cor. 2). No plan with $\bar C(a) < B^\ast$ can deliver an exact
  certificate, where $B^\ast$ is the least expected cost of a plan that reaches $d^\ast$
  agreeing draws. Below $B^\ast$, CEL-k never certifies; the conformal layer still applies.
- **Elbow.** Let $(\bar k_i, y_i)$ be CEL-k's dev curve over the budget grid, with $y_i$ the
  running maximum of $\widehat{\mathrm{AUARC}}_{\mathrm{dev}}(\hat a(\bar k_i))$. With both
  coordinates scaled to $[0,1]$ ($\tilde k$, $\tilde y$), the elbow is
  $k^\ast = \arg\max_i\ (\tilde y_i - \tilde k_i)$ (Kneedle; ties go to the smaller
  budget). It is chosen on dev only. The metrics table compares both modes there: CEL-k and
  every baseline at $k^\ast$ calls, and certified CEL at its own spend.
