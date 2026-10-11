# confledger — CEL: Confidence Evidence Ledger

> Certified confidence at minimal API cost: every call is a deposit of evidence,
> spent only until the selection is certified.

Label-free confidence ranking for black-box LLMs under a limited API budget —
the reference implementation of the methodology and experimental protocol in
the paper draft (Sec. 3–4).

**The ledger view.** Every observation is evidence in one currency, nats of
log-likelihood ratio. An agreeing draw deposits +0.305, a disagreeing draw
withdraws −1.79, a passed label-action probe deposits the equivalent of ~2.1
draws (the exchange rate κ, Prop. 3). The screen closes accounts that go into
the red early; survivors keep accumulating until the balance crosses the
certification threshold (Cor. 1). `stats.EvidenceLedger` implements exactly this.

## Quick start

```bash
pip install -e ".[dev]"            # add ,openai / ,data (and ,anthropic) for real runs
pytest -q                          # 32 tests: paper constants, metrics, backends, end-to-end
experiments/scripts/run_all.sh configs/simulation.yaml   # full dry run, no API keys (~1 min)
```

The dry run uses a **simulated model** that encodes the paper's hypotheses
(concept vs. cue-following vs. uncertain cases). It checks that every stage of
the pipeline runs and that the statistics behave; its results are **not
evidence** — the method wins there by construction.

## Real runs

1. `python -m data.prepare -c configs/default.yaml`: downloads the
   datasets and checks their schemas (no API calls).
2. Fill the model ids in a config template (`configs/default.yaml`, or
   `configs/smoke.yaml` for a first one-pair run). `experiments/scripts/collect.py` refuses
   to start while any `<placeholder>` remains.
3. Choose the transport. Leave `LLM_BACKEND` unset to call the OpenAI API
   directly (`OPENAI_API_KEY`). Set `LLM_BACKEND=safechain` in the private
   environment, where SafeChain serves the same model ids.
4. `python experiments/scripts/collect.py -c configs/<name>.yaml`, then `evaluate.py`. Each
   config writes to `experiments/results/<dataset>/<name>/`; for a variant of a run,
   copy the config under a new name rather than editing one that already has results.
5. Freeze prompts and thresholds on the dev slice with `experiments/scripts/dev_audit.py`:
   hand-audit ~50 extractions and the paraphrases for meaning preservation.

Collection is the **only** API spend (~42 calls/item at the default sizes: 12
fresh-paraphrase draws, 16 same-prompt draws, 6 reuse draws, 4 probes, challenge, verbal,
paraphrase-gen, judge; `estimate.py` prints the exact figure). For 3 datasets × 3 models ×
700 items that is ~260k calls; start with one
(dataset, model) pair — the minimum viable result — before committing the grid.
Everything downstream (baselines, ablations, budget curves, 32 Shapley
coalitions, evidence arms) is a re-scoring of cached data.

## Layout

| Folder | What | Spends API calls? |
|---|---|---|
| `src/confledger/` | the method and its evaluation protocol (a library; no CLI) | only via `collect.py` |
| `data/` | all data preprocessing: dataset loaders (TriviaQA de-duplication), dev/eval split, `prepare.py` (download + checks) | no |
| `experiments/scripts/` | every step as a script: `estimate.py`, `collect.py`, `evaluate.py`, `dev_audit.py`, one script per paper table/figure, `run_all.sh`; see `experiments/README.md` | only `collect.py` |
| `experiments/` | per-dataset notebooks (`triviaqa.ipynb`) and `results/` (one folder per dataset, a config folder inside each; `all/` for cross-dataset outputs; the shared cache) | no |
| `configs/` | templates: `default.yaml` (full grid), `smoke.yaml` (one pair), `simulation.yaml` (dry run) | — |
| `docs/` | paper draft (`paper/`), design history (`decisions.md`), methodology and theory notes, brainstorming | — |
| `tests/` | `pytest -q` | no |

Suggested order: `data.prepare` → `estimate.py` → `collect.py` one cell → `dev_audit`
(fix prompts if gates fail) → collect the grid → `experiments/scripts/run_all.sh`.

## Architecture: collect once, replay every method

`src/confledger/` holds the methodology pipeline; `src/confledger/utility/` holds the tools it
uses (model clients, parsing, config, records, report rendering).

| Module | Paper | Role |
|---|---|---|
| `sprt_posterior.py` | §3.2–3.7 | Jeffreys posterior, tail masses, SPRT, Lemma 2–3, Cor. 1, κ and k_eff (Prop. 3), Miller–Madow entropy, and `EvidenceLedger` (the per-case account) |
| `collection.py` | §4.3 | superset collection per item, resumable JSONL, one judge call per item |
| `methods.py` | §3, §4.4 | the pipeline, ablations, baselines, budget configs, evidence arms — each a replay policy **with explicit call accounting** |
| `ranking.py` | §3.5 | cascade keys, Borda, eliminated-case ordering, tie-aware top-k |
| `prompts.py` | — | every prompt template (answer, paraphrase, probes, challenge, verbal, judge) |
| `metrics.py` | §4.5 | prec@k + exact hypergeometric null, AUROC, PRR (exact ties), paired bootstrap/Wilcoxon, Holm, strata + isotonic gap, top-strata test, Jaccard + chance floor, held-out audit, empirical κ̂ |
| `signal_shapley.py` | §3.8 | exact 32-coalition Shapley |
| `evaluation_protocol.py` | §4 | the full protocol for one collection (`run_protocol`) |
| `utility/answer_parsing.py` | — | yes/no, A/B, confidence and JSON parsing of model text |
| `utility/cell_report.py` | §4.5 | per-cell markdown report, budget and monotonicity figures |
| `utility/records.py` | — | `Item`, `Draw`, `CaseData` (the collection record) |
| `utility/config.py` | — | load and check a config (refuses `<placeholder>` model ids), selected cells, run log (seed, environment, git commit) |
| `utility/llm/` | — | `request.py`, `response_cache.py` (SQLite), `backends.py` (OpenAI-compatible / SafeChain / Anthropic), `simulated_model.py` |

## Design decisions made while implementing (fold back into the paper)

1. **Held-out perturbation family** is a conversational re-framing, not entity
   renaming: renaming changes the true answer of factual questions, so it is not
   label-preserving.
2. **TriviaQA distractors** come from a separate generator model asked for a
   plausible wrong answer (checked against gold aliases), replacing the draft's
   "disjoint pool" wording, which did not produce answers to the same question.
3. **Matched budget is set automatically** to the pipeline's realised mean
   calls/case on each collection, since adaptive spend is only known after replay.
4. **Shapley holds the stage-1 screen fixed**: V(∅) = random among survivors, and
   the report separates random → screen (design) from screen → signals (Shapley).
5. **No-SPRT ablation** = flat 2 draws + all probes for every case (9 calls incl. paraphrase generation), a
   genuinely budget-matched flat allocation.
6. **Judge from the same provider** (OpenAI) as the model under test, because
   SafeChain serves only OpenAI models. The judge is a different model; judge
   sensitivity is checked with `alt`.
7. **TriviaQA is de-duplicated by `question_id`** before sampling: `rc.nocontext`
   lists most questions twice (web + wiki).
8. **Paraphrase generation costs 1 call per case** for every paraphrase-based policy
   (CEL, ablations, evidence arms); same-prompt baselines don't pay it. Raises CEL's
   counted cost and the matched budget by 1.
9. **Signal effects** are reported as leave-one-out ablations of each signal (on prec@10,
   AUROC and PRR) and as Shapley values on prec@10 with a CI (`signal_effects.tex`).
10. **`collect.verbal_per_draw`** (off by default) puts a confidence line on every draw and
    drops the separate verbal call; the protocol refuses collections smaller than it needs.
11. **Self-consistency is reported three ways:** a budget curve (3/6/9/12 draws), at the
    matched budget (the confirmatory test), and at `deployment_sc_draws` (3) as the cheap
    deployment baseline, ringed on the headline figure. `n_same` defaults to 16 so the
    matched budget (every CEL call counted, at most 16 per case) always fits.
12. **CEL-k, the budget-allocated deployment mode:** the budget is spread over a screen and
    the survivors' probes (allocation chosen on the dev slice, expected calls ≤ k), compared
    with self-consistency with k draws and with self-consistency + verbal confidence
    (`budget_matched.tex`, green line on the headline figure).
13. **SPRT settings are configurable** (`sprt:` in the config; defaults = the paper). The
    certification depth is derived from them; `sprt_settings.tex` shows how much guarantee
    each setting buys and how many draws it costs.

## Things the dry run already surfaced

- **The screen trades global AUROC for top-k precision.** Eliminated cases sit in
  a coarse bottom block, so AUROC/PRR can trail single-signal baselines while
  prec@10 wins. This is the "different regions of the ranking" point of §4.5 —
  report both, and expect reviewers to ask.
- **Jaccard understates stability when many cases tie at the top** (several
  perfect cases, random tie-breaking). `experiments/scripts/robustness.py` therefore also reports
  a tie-aware variant: the share of re-drawn selections that fall inside the
  original run's top tie-group.
- **The truncated SPRT is conservative, not calibrated to its nominal rates.**
  With 3+6 draws, false certification at p0 is 0.04 (nominal 0.10) but the miss
  rate at p1 is 0.37 (nominal 0.10): certification requires 9/9. `experiments/scripts/sprt_characteristics.py`
  reports this exactly; the paper should state it (it is the price of truncation).
