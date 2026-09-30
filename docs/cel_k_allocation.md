# CEL-k: budget allocation for deployment

How CEL spends a small, fixed budget of LLM calls per case, why the allocation is
chosen the way it is, and what it does and does not guarantee. Design history:
`decisions.md` #12 and #14. Code: `src/confledger/methods.py` (`Allocation`,
`cel_alloc`, `allocation_space`) and `src/confledger/evaluation_protocol.py`
(`select_allocations`, step 2b of `run_protocol`).

## 1. The problem

Adaptive CEL (the certified mode) spends a different number of calls on each case:

| Part | Calls | Paid by |
|---|---|---|
| paraphrase generation + 3 screen draws | 4 | every case |
| 6 more draws (to reach 9/9 unanimity) | 6 | survivors |
| 3 flips (negation, relabel, distractor) | 3 | survivors |
| judge (SE) + challenge + verbal | 3 | survivors |

With survival rate *s*, the mean cost is **4 + 12·s calls per case**: about 12.6 at
s = 0.72, and higher for a model that is consistent on most questions. The standard
deployment baseline, self-consistency, spends *k* calls (typically 3). Even if
adaptive CEL ranks better, a 4× cost is hard to justify in production.

Half of a survivor's cost is the 6 extra draws, and they exist only for
**certification**: reaching the SPRT boundary and Cor. 1's tail-mass bound so the
selection carries a per-case guarantee (Prop. 2). Ranking cases well does not need
that depth. CEL-k drops certification and asks a narrower question:

> Given an **expected** budget of *k* calls per case, how should the calls be split
> between sampling and probing to rank cases best?

## 2. The allocation space

An allocation *a* has two stages.

**Screen (every case).** *k₁* draws, either on the original wording or on fresh
paraphrases (paraphrases cost one generator call per case). If the *k₁* answers
disagree, or one fails to parse, the case is **eliminated**: it gets no more calls
and ranks below every survivor. With *k₁* = 1 nothing can disagree, so every parsed
case survives.

**Survivors only.** Any combination of:

| Component | Calls | Signal | Available in the collection |
|---|---|---|---|
| *e* extra draws | *e* | agreement over *k₁* + *e* draws (lower bound *L*) | up to 12 paraphrased / 16 same-prompt |
| flips *F* ⊆ {negation, relabel, distractor} | \|F\| | flip pass rate | one of each (T = 0) |
| challenge | 1 | does the answer hold under pushback? | one |
| SE | 1 (judge) | rationale entropy over the draws; needs ≥ 2 draws | one judge call |
| verbal | 0 | mean confidence of the draws used | on every draw (`verbal_per_draw: true`) |

Verbal confidence is free because each draw's prompt asks for a `Confidence:` line
(`collect.verbal_per_draw`). Without that switch, a separate verbal call is a
further one-call option.

`allocation_space()` enumerates every combination the collection can replay:
screen 1–4 draws, original or paraphrased, 0–6 extra draws (so adaptive CEL's own
allocation, a 3-draw screen plus 6 extra draws and the whole battery, is one of them),
the 8 flip subsets, challenge on or off, SE on or off. That is **1,760 allocations**.

## 3. Cost model: expected calls

For allocation *a*, the calls one case costs are

C(a) = [paraphrased] + k₁ + 1{survives} · (e + |F| + [challenge] + [SE])

so the expected cost per case is

**E[C(a)] = [paraphrased] + k₁ + σ(a) · (e + |F| + [challenge] + [SE])**,

where σ(a) is the survival rate of the *k₁*-draw screen. For a case whose modal
answer has per-draw probability *q*, the screen survives with probability
q^k₁ + (1 − q)^k₁ (all draws agree on either answer). σ(a) averages this over cases,
so it depends on the model and the dataset. **It is estimated on the dev slice**:
Ê_dev[C(a)] is the mean number of calls the allocation actually spends when replayed
on the dev collection.

**Why an expected budget is the right unit.** Deployment cost is total calls per
batch divided by batch size. For a batch of *n* cases, the realised mean cost varies
around E[C(a)] with standard deviation c_s·√(σ(1 − σ)/n), where c_s is the survivor
cost. For *n* = 50, σ = 0.7 and c_s = 3, that is **±0.19 calls**. The average is a
stable budget for a batch, even though individual cases cost different amounts.
Self-consistency with *k* draws has no such variation: it costs exactly *k*.

## 4. Why a screen and probes can beat more draws

**The screen moves budget to the cases that matter.** For top-*k* selection, only
cases that could reach the top need detailed evidence. A disagreement in a few
draws is strong evidence against a case (each disagreement is −1.79 nats in the
SPRT, Lemma 3), so the screen stops spending on unstable cases and gives their
share to survivors. Lemma 3(a) gives the screen's pass probability: p̄³ for the
standard 3-draw screen, 0.343 at p̄ = 0.7 and 0.857 at p̄ = 0.95.

**A probe can be worth more than a draw.** Under the evidence model of Prop. 3
(model-based: it assumes draws and probes are independent given the case's
reliability), each call adds log-likelihood-ratio evidence:

| Call | Evidence if it passes / agrees | In draw-equivalents | If it fails |
|---|---|---|---|
| agreeing draw | +0.305 nats | 1 | −1.79 nats |
| negation or relabel | +0.642 nats | κ ≈ 2.1 | −2.30 nats (≈ −7.5 draws) |
| distractor | +0.172 nats | κ ≈ 0.56 | −1.39 nats (≈ −4.5 draws) |

A passed negation or relabel probe is worth about two agreeing draws, and a failed
one is strongly disconfirming. At a fixed budget, a call spent on a flip can
therefore carry more evidence than a call spent on another sample.

**Probes see errors that sampling cannot.** Lemma 4: for a case where the model
follows a surface cue, the answer is stable under resampling, so any number of
extra draws agree and resampling has **no power** to detect the error. A flip probe
changes the cue-to-answer mapping (negation flips the correct label, relabel swaps
the options), so a cue-follower fails it. This is the stable-but-wrong regime
TriviaQA was chosen for. Spending the whole budget on draws, as self-consistency
does, cannot catch it.

**Rationale diversity and pushback add different information.** SE (Miller–Madow
entropy of the judge's clusters, Eq. 9) separates "same answer for the same reason"
from "same answer for scattered reasons". It needs at least 2 draws and one judge
call. The challenge tests whether the answer survives "are you sure?".

These arguments say that *some* mix of sampling and probing should beat
sampling alone. They do not say which mix. That depends on the model and the
dataset, so it is chosen empirically (§6).

## 5. Ranking

Survivors are ranked by **Borda** over the signals the allocation produces: each
signal ranks the survivors (SE with its sign flipped, since lower entropy means more
confident), and a case's score is its average rank. Eliminated cases sit below every
survivor, ordered by how many screen draws agreed (`ranking.borda_scores`). The
answer for each case is the majority over its draws.

Borda has no weights, which matters because the allocation is chosen on only ~100
dev items per dataset: a learned combination (e.g. logistic regression over the
signals) would have to fit its weights on that same dev slice and overfit it. Borda
is also scale-free, so a 0–100 confidence and a 0–1 pass rate combine directly.

## 6. Choosing the allocation on the dev slice

For each budget *k* in `budget_matched_k` (default 2, 3, 4, 5, 6, 8):

**â(k) = argmax over a of Score_dev(a), subject to Ê_dev[C(a)] ≤ k**  (paper Eq. devalloc),

with ties broken toward fewer expected calls (`select_allocations`).

- **Score:** the expected precision@10 of the allocation's Borda ranking, averaged
  over dev batches and over random tie-breaks. When no dev batch has more than 10
  items (e.g. the 10-item smoke dev slice), precision@10 is degenerate and AUROC is
  used instead; the result records which.
- **Search:** exhaustive over all 1,760 allocations. The space is small, so there is
  no need for greedy or approximate search. Exhaustive search also captures
  interactions that per-signal rankings miss: two flips may be redundant together,
  or SE may only help once there are 3 or more draws.
- **Frozen before evaluation:** â(k) is chosen using dev items only and then
  scored once on the eval items. Nothing is tuned on eval (a CLAUDE.md rule).

**Selection optimism.** Picking the best of ~1,250 candidates on a finite dev slice
overstates the winner's dev score (a winner's-curse effect). Because the eval items
are untouched, the reported eval result is not biased by this, but the chosen
allocation may not be the true best. The full runs use 100 dev items per dataset
(the smoke run's 10 only tests the pipeline).

## 6b. The derived allocation, certification budgets, and the link to adaptive CEL

The dev-chosen allocation is empirical. The paper (§1.9) also derives one from the
theory, and the two are compared.

**Derived allocation (model-based, label-free).** Among fresh-paraphrase allocations
with expected cost ≤ k, pick the one maximising the evidence a clean survivor carries:
k_eff(a) = k₁ + e + Σ κ_g over its flips (Eq. derivedalloc), ties toward fewer calls.
Written a\*(k) in the paper. It reads only replayed call counts, never labels. Its predictions: negation and
relabel (κ ≈ 2.1) are bought before extra draws, the distractor (κ ≈ 0.56) after them,
and the challenge and SE never, since the theory gives them no exchange rate. The gap
between the derived and the dev-chosen allocation measures what the theory leaves
unpriced (challenge, SE, confidence) plus any evidence double-counting
(`evaluation_protocol.select_allocations`, `methods.evidence_depth`).

**Nesting (exact).** Adaptive CEL is itself an allocation: a 3-draw screen on fresh
paraphrases, then 6 draws, the battery, challenge and SE for survivors
(`methods.cel_as_allocation`), costing 4 + 12π₃ calls (4 + 11π₃ with confidence on the
draws). Replaying it as an allocation reproduces adaptive CEL's cases and call counts;
only the combiner differs (Borda instead of the cascade). The two modes are one method
under two constraints: adaptive CEL fixes the certificate and lets spend vary; CEL-k
fixes spend and lets the certificate vary.

**Certification budgets (Cor. certbudget).** A delivered case is certified exactly
when its tail mass ε ≤ 0.05 at p₀* = 0.8 (9 unanimous draws), and model-based when
its effective depth reaches k_eff ≥ 8.35 with no failed probe. The certification
budget B* is the least expected cost of an allocation that can certify, estimated on
the dev slice. Below it, CEL-k can only rank; above it, certified cases appear
(`evaluation_protocol.delivered_certificates` reports the certified share of each
delivered top-10). In the simulator: model-based B* ≈ 7 calls, exact B* ≈ 8.4.

**Mode-linkage figure** (`experiments/scripts/mode_linkage.py`): precision@10 and the
certified share of the top-10 for CEL-k under the derived allocation as k grows, with
adaptive CEL as a point and the axis shaded at the two certification budgets. Its
companion table `mode_linkage.tex` lists, per budget, the dev-chosen and derived
allocations and checks the nesting.

## 7. Evaluation

For each budget *k*, CEL-k with â(k) is compared with:

| Method | Calls per case |
|---|---|
| **CEL-k** | its expected calls under â(k) (reported on dev and on eval) |
| **self-consistency** with *k* draws | exactly *k* |
| **self-consistency + verbal** (control) | exactly *k* |

The comparison is paired over batches: the mean difference in precision@10 with a
bootstrap 95% CI, wins per batch, and a Wilcoxon test (`metrics.paired_compare`).
It is reported both against plain self-consistency and against the control.

**Why the control is needed.** With `verbal_per_draw` on, every draw carries a
confidence at no extra cost. CEL-k's Borda ranking uses it; plain self-consistency
ranks by agreement only. Self-consistency + verbal uses the same *k* draws and breaks
agreement ties by mean confidence. If CEL-k beats plain self-consistency but not
the control, its gain comes from reading the confidences, not from how it spends
its calls. Only a win over the control supports the allocation argument of §4.

**Outputs.**
- `budget_matched.tex`: per *k*, the chosen allocation, its expected calls
  (dev / eval), and the three methods' precision@10, with paired Δ and CI.
- The headline figure (`budget_curves_<model>.pdf`): CEL-k is the green line,
  plotted at its expected calls per case, next to adaptive CEL and the
  self-consistency curve. The 3-draw self-consistency point, the deployment
  baseline, is ringed.
- The per-cell report and notebook §8 show the same table with the dev metric and
  the number of candidates.

## 8. What CEL-k does not give

- **No certificate.** No case accumulates the unanimous depth certification needs
  (9 draws with the paper's settings, or 5 with the cheaper settings in
  `sprt_settings.tex`), so CEL-k's picks carry no tail mass ε and no Prop. 2 bound.
  CEL-k is the **ranking mode**; adaptive CEL is the **certified mode**. A deployment
  that needs a guarantee on each selected case must pay for the certified mode.
- **Stability, not correctness.** As everywhere in CEL, the signals measure how
  reliably and how robustly the model gives its answer. Turning that into "likely
  correct" rests on Assumption 1 (monotonicity), which the protocol tests separately.
- **Model-based reasoning in §4.** The exchange rates are exact given the evidence
  model of Prop. 3, but that model assumes conditional independence of draws and
  probes; the real value of a probe is what the dev search and the eval comparison
  measure.

## 9. Assumptions and limitations

- **Collection bounds.** The collection records one of each flip and one challenge
  (at T = 0), so an allocation can use at most 3 flips and 1 challenge. More would
  need more probe types, or sampled probes, at collection time.
- **SE replay.** The judge clustered all ~34 explanations of a case at collection
  time; CEL-k's SE uses those labels restricted to the draws it replays. In
  deployment the judge would see only those draws. Adaptive CEL makes the same
  approximation.
- **Challenge target.** The challenge is collected against the original-wording
  draw's answer. It counts as passed only when that answer equals the allocation's
  majority answer.
- **Elicitation effect.** Asking for confidence in the same response can change the
  answers themselves. The dev-slice audit should compare answers with and without
  the confidence line.
- **Dev-slice size.** With 100 dev items the chosen allocation is a noisy estimate
  of the best one (§6). A larger dev slice reduces this; the eval comparison stays
  honest either way.

## 10. Configuration

| Key | Default | Effect |
|---|---|---|
| `budget_matched_k` | `[2, 3, 4, 5, 6, 8]` | budgets *k* for which an allocation is chosen and compared |
| `collect.verbal_per_draw` | `true` | confidence on every draw (free verbal); `false` makes verbal a one-call option |
| `deployment_sc_draws` | `3` | the self-consistency point ringed as the deployment baseline |
| `split.n_dev` | `100` | dev items per dataset used to choose allocations |

The collection must contain at least 12 paraphrased and 16 same-prompt draws per
case for the full protocol; `evaluate.py` refuses smaller collections with a
message saying what is missing.

## 11. Status of the evidence

Everything above has run end to end on the simulator only. The simulator builds in
the paper's hypotheses (informative probes, informative verbal confidence), so its
numbers are **pipeline checks, not evidence**. They did show one thing worth
carrying into the real runs: on two of three simulated datasets, self-consistency +
verbal matched CEL-k, so the control can decide the outcome. The real question, which
the first TriviaQA × gpt-4.1 run will answer, is whether at 3–5 expected calls the
probes add ranking quality beyond what the same budget of draws plus their
confidences already give.
