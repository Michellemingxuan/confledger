"""Appendix: task-specific signal importance for confidence estimation. Per (dataset, model)
cell, the SHAP importance of every signal in the score model fitted on D_fit with every
signal bought (signal_value.py, decisions.md #19): I_g = mean |beta_g (x_g - mean x_g)| in
log-odds, with a bootstrap 90% interval over D_fit refits. Shown as each signal's share of
the total, so cells are comparable. Kendall tau between datasets' importance vectors
measures how task-specific the ranking is.
Model-based: one fitted model, signals treated as independent; the stability term enters
unpenalised (decisions.md #16), which favours it against the ridge-penalised signals.
Outputs: signal_importance_<model>.pdf, signal_importance.tex.
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from _paper import latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, from_args  # noqa: E402
from confledger.evaluation_protocol import cell_results  # noqa: E402

NAMES = {"eta": "agreement (stability)", "survived": "survived the screen", "neg": "negation probe",
         "relabel": "relabel probe", "distractor": "distractor probe", "chal": "challenge",
         "verb": "verbal confidence", "se": "semantic entropy", "length": "answer length",
         "hedge": "hedging words", "neg_drop": "confidence drop under negation",
         "p_first": "first-token probability (gray-box)"}

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
gray = (cfg.get("scoring") or {}).get("gray_box", False)
sigs = [g for g in NAMES if gray or g != "p_first"]
rows = []
for mk in cfg["_models"]:
    dss = cfg["_datasets"]
    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 3.4), squeeze=False, sharex=True)
    shares = {}
    rows.append([f"\\multicolumn{{{len(dss) + 1}}}{{l}}{{\\emph{{{mk}}}: rank (share of total importance)}}"])
    for j, ds in enumerate(dss):
        imp = cell_results(cfg, ds, mk, a.force)["knapsack"]["importance"]
        tot = sum(imp["I"][g] for g in sigs) or 1.0
        sh = {g: imp["I"][g] / tot for g in sigs}
        shares[ds] = sh
        order = sorted(sigs, key=lambda g: sh[g])          # largest at the top of the bar chart
        lo = [sh[g] - imp["ci"][g][0] / tot for g in order]
        hi = [imp["ci"][g][1] / tot - sh[g] for g in order]
        ax = axes[0][j]
        ax.barh([NAMES[g] for g in order], [sh[g] for g in order], xerr=[[max(0, x) for x in lo], [max(0, x) for x in hi]],
                color="C0", ecolor="0.4", capsize=2)
        ax.set_title(f"{ds} (n_fit = {cell_results(cfg, ds, mk, a.force)['knapsack']['n_fit']})", fontsize=9)
        ax.set_xlabel("share of total |SHAP| (log-odds)")
        ax.tick_params(axis="y", labelsize=7.5)
    rank = {ds: {g: i + 1 for i, g in enumerate(sorted(sigs, key=lambda g: -shares[ds][g]))} for ds in dss}
    for g in sigs:
        rows.append([NAMES[g]] + [f"{rank[ds][g]} ({100 * shares[ds][g]:.0f}\\%)" for ds in dss])
    for d1, d2 in itertools.combinations(dss, 2):
        tau = kendalltau([shares[d1][g] for g in sigs], [shares[d2][g] for g in sigs]).statistic
        print(f"{mk}: Kendall tau({d1}, {d2}) = {tau:+.2f}")
        rows.append([f"\\quad Kendall $\\tau$ ({d1}, {d2})"] + [f"{tau:+.2f}" if ds == d2 else "" for ds in dss])
    fig.suptitle(f"Signal importance in the score model: {mk}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"signal_importance_{mk}.pdf", bbox_inches="tight")
    fig.savefig(out / f"signal_importance_{mk}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
latex_table(["Signal"] + cfg["_datasets"], rows,
            "Task-specific signal importance for confidence estimation: mean $|$SHAP$|$ of each signal "
            "in the score model fitted on $\\mathcal{D}_{\\mathrm{fit}}$ with every signal bought, as rank and "
            "share of the total (model-based; bootstrap intervals in the figure). Kendall $\\tau$ compares "
            "the datasets' importance vectors.", "tab:signal-importance", out / "signal_importance.tex")
