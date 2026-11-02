"""confledger -- CEL, the Confidence Evidence Ledger.

Certified confidence at minimal API cost: every call is a deposit of evidence
(in nats of log-likelihood ratio), spent only until the selection is certified.
Agreements are small deposits, disagreements large withdrawals, probes deposits
in a second currency converted at the exchange rate kappa (Prop. 3); a case is
certified when its balance crosses the SPRT threshold.

Layout. The methodology pipeline (maps to the paper):
  sprt_posterior.py       Sec 3.2-3.7: posterior, SPRT, Cor. 1, Prop. 3, EvidenceLedger
  methods.py              CEL, ablations, evidence arms: replay policies with call accounting
  baselines/              self-consistency, Xiong (Avg-Conf), CISC, Pedapati-style LR, others
  ranking.py              cascade / Borda ranking, top-k incl. eliminated cases
  prompts.py              every prompt template
  collection.py           one superset API collection per (dataset, model), cached
  evaluation_protocol.py  the full Sec 4 protocol (run_protocol, cell_results)
  metrics.py              prec@k, AUROC, AUARC, PRR, calibration, WQD, paired tests, monotonicity, Jaccard
  signal_shapley.py       exact 32-coalition signal valuation (Sec 3.8)
  score_model.py          correctness score fitted on D_fit; pessimistic score mu - lambda*sigma
  certified_selection.py  certified mode: conformal p-values on D_cal + BH; ledger e-values + e-BH
  frontier.py             CEL-k frontier allocation: LUCB on the top-k cut, call by expected move
  signal_value.py         SHAP importance, pruning, values per call, Prop. K knapsack allocation
  stability_prior.py      fitted Beta prior + bootstrap certification depth; GLR stopping rule
Tools it uses, in utility/:
  llm/                    request, response_cache, backends (OpenAI/SafeChain/Anthropic), simulated_model
  answer_parsing.py       yes/no, confidence and JSON parsing of model text
  config.py               load/check a config, cells, run log (seed, env, git)
  records.py              Item / Draw / CaseData
  cell_report.py          per-cell markdown report and figures
"""
__version__ = "0.1.0"
