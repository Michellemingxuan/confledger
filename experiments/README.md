# experiments/

```
experiments/
  scripts/         estimate, collect, evaluate, dev audit, one script per paper table/figure, run_all.sh
  triviaqa.ipynb   per-dataset notebook
  results/         everything runs produce
```

Every step is `python experiments/scripts/<step>.py -c CONFIG`. Only `collect.py` spends API
calls. The paper scripts read the per-cell protocol results that `evaluate.py` writes, recomputing them from cached collections if
missing (`--force` recomputes). Every script takes `-c CONFIG`, plus `-d`/`-m` to
restrict to cells. Tables and PDFs go to `<results>/paper/`, plus a PNG copy of each figure for the notebook (§10 of `triviaqa.ipynb` runs them all and shows the figures).

| Script | Paper | Output |
|---|---|---|
| `estimate.py` | §4.3 | API calls per cell before any spend |
| `collect.py` | §4.3 | the collection: the only API spend (resumable) |
| `evaluate.py` | §4 | per-cell protocol results, reports, `summary.json` (Holm) |
| `dev_audit.py export/score` | §4.1 | extraction + paraphrase audit CSVs to hand-label, then accuracy with Jeffreys CIs (gates ≥0.98 / ≥0.95); run before the grid |
| `sprt_characteristics.py` | §3 | `sprt_constants.tex` (derived constants, also printed), `sprt_settings.tex` (draws vs guarantee strength under other H0/H1/α/β); exact operating characteristics of the **truncated** SPRT (`truncation.tex`, `truncation_oc.pdf`) |
| `main_ladder.py` | §4.4 | `main_ladder.tex` (ladder + ablations per cell), `confirmatory.tex` (Holm) |
| `budget_curves.py` | §4.5 headline | `budget_curves_<model>.pdf` (prec@10 and AUROC vs calls: adaptive CEL, budget-matched CEL-k, self-consistency, deployment baseline ringed), `budget_matched.tex` (CEL-k vs SC at exactly k calls), `horizontal_gap.tex` |
| `mode_linkage.py` | §3.9 | `mode_linkage_<model>.pdf` (CEL-k at growing budget, derived allocation, vs adaptive CEL: prec@10 with bootstrap band, and certified share of the top-10, exact and model-based; shading = certification budgets), `mode_linkage.tex` (dev-chosen vs derived allocation per budget; nesting check) |
| `evidence_exchange.py` | Prop. 3 | `evidence_arms.tex` (A/B/C verdicts), `exchange_rates.tex` (κ vs κ̂) |
| `monotonicity.py` | Ass. 1 | `monotonicity.pdf`, `monotonicity.tex` |
| `signal_effects.py` | §3.8 | `signal_effects.tex` (per signal: leave-one-out Δprec@10 with CI, ΔAUROC, calls saved; Shapley on prec@10 with CI), `shapley_cost.tex`; prints measured vs asserted signal order |
| `robustness.py` | §4.5 | `robustness.tex` (Jaccard, tie-aware stability, held-out audit, judge swap) |

**Notebooks** (`pip install -e ".[notebook]"`) organise one dataset end to end:
config → raw data → budget → results folders → collection (only with `SPEND = True`)
→ collection health → dev audit → evaluation → per-case drill-down with the evidence
ledger. They only call `confledger`; any logic they need goes into the package.

| Notebook | Dataset |
|---|---|
| `triviaqa.ipynb` | TriviaQA (stable-but-wrong regime) |

`scripts/_paper.py` holds the LaTeX helpers. `scripts/run_all.sh CONFIG` runs the whole chain:
`estimate.py` → `collect.py` → `evaluate.py` → every paper script above. Every
collect/evaluate appends seed, environment and git commit to `<results>/run_log.jsonl`.

## `results/`: everything a run produces

One config in `configs/` = one run = one folder here, named after the config:

```
results/
  cache/          shared response cache (gitignored): an identical request is never paid twice
  <config name>/  e.g. smoke/, default/
    collections/  raw per-item records (large, gitignored)
    results/      per-cell reports, summary.json, paper/  (default/results/data/ = data.prepare manifests)
  simulation/     dry-run outputs (gitignored)
```

```bash
python experiments/scripts/collect.py  -c configs/smoke.yaml -v
python experiments/scripts/evaluate.py -c configs/smoke.yaml
```

For a variant (another prompt, split or model), copy the config under a new name
and change its `paths` to the new folder. Don't edit a config that already has
results: the config is the record of what produced them.
