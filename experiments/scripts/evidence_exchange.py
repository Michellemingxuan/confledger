"""Prop. 3 tests: evidence-equivalence arms A/B/C (pre-registered B ~ A > C) and
derived exchange rates kappa vs empirical kappa-hat, per cell.
"""
from _paper import f, latex_table, paper_dir
from confledger.utility.config import arg_parser, cells, from_args
from confledger.evaluation_protocol import cell_results

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
arm_rows, k_rows = [], []
for ds, mk in cells(cfg):
    r = cell_results(cfg, ds, mk, a.force)
    A, B, Bh, C = (r["arms"][n]["mean_prec"] for n in
                   ("A_12draws", "B_6draws_battery", "B_half_kappa", "C_9draws"))
    ba, bc = r["arms_tests"]["B_vs_A"], r["arms_tests"]["B_vs_C"]
    verdict = ("B$\\approx$A$>$C" if ba["ci_lo"] <= 0 <= ba["ci_hi"] and bc["ci_lo"] > 0 else
               "B$>$A" if ba["ci_lo"] > 0 else "B$<$A (double-counting)" if ba["ci_hi"] < 0 else "inconclusive")
    arm_rows.append([ds, mk, f(A), f(B), f(Bh), f(C), f"{ba['mean_diff']:+.3f}", f"{bc['mean_diff']:+.3f}", verdict])
    kh = r["kappa_hat"]
    for g in ("neg", "relabel", "distractor"):
        lo, hi = kh["ci"][g]
        k_rows.append([ds, mk, g, f(r["kappa_derived"][g], 2), f(kh["kappa_hat"][g], 2), f"[{f(lo, 2)}, {f(hi, 2)}]"])
latex_table(["Dataset", "Model", "A (12d)", "B (6d+probes)", "B ($\\kappa/2$)", "C (9d)",
             "B$-$A", "B$-$C", "Verdict"], arm_rows,
            "Evidence-equivalence arms. Prediction (Prop.~3): B buys A's quality at C's price.",
            "tab:arms", out / "evidence_arms.tex", "llrrrrrrl")
latex_table(["Dataset", "Model", "Probe", "$\\kappa$ (derived)", "$\\hat\\kappa$", "95\\% CI"], k_rows,
            "Exchange rates: derived from the evidence model vs.\\ fitted by logistic regression.",
            "tab:kappa", out / "exchange_rates.tex", "lllrrr")
