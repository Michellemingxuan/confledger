"""Headline figure R1 (ranking mode): quality vs realised calls per case, one row per dataset,
columns = prec at the top 1%, 5% and 10%, AUROC and AUARC, one figure per model: budget-allocated CEL-k (allocation
chosen on the dev slice, ranked by mu - lambda*sigma fitted on D_fit), certified CEL at its own
spend, and the three baselines that carry the argument: self-consistency, its verbal control
(H3) and BSDetector. 95% bootstrap intervals on prec for CEL-k and certified CEL. Every
other baseline, and every interval, is in the tables. Tables: budget_curve.tex (every method at every k: prec@10%,
calls, accuracy, AUROC, AUARC, Delta vs CEL-k), budget_matched.tex (CEL-k vs
self-consistency with k draws) and horizontal_gap.tex (calls self-consistency needs to
match CEL).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _paper import latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, from_args  # noqa: E402
from confledger.evaluation_protocol import cell_results  # noqa: E402
from confledger.metrics import horizontal_gap  # noqa: E402


def series(curve, prefixes, key, lo="lo", hi="hi"):
    """(calls, value, lo, hi) of every curve point under `prefixes`; at equal calls the
    first prefix wins."""
    pts = {}
    for p in reversed((prefixes,) if isinstance(prefixes, str) else prefixes):
        for k, v in curve.items():
            if k.startswith(p):
                pts[round(v["calls"], 6)] = (v["calls"], v[key], v.get(lo), v.get(hi))
    return list(zip(*sorted(pts.values()))) if pts else None



a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
# fixed colours: self-consistency keeps C1, the colour of the deployment-baseline ring
# Colours validated for colour-blind separation; each series also has its own marker.
OURS = "#C2417A"
LINES = (("celk:", "CEL-$k$ (ours)", OURS, "o"),
         ("k:self_consistency_verbal:", "self-consistency + verbal", "#1F5FA8", "s"),
         ("k:bsdetector_like:", "BSDetector", "#2BA58C", "^"),
         (("sc:", "k:self_consistency:"), "self-consistency", "#E08A2E", "D"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
         "font.size": 10, "axes.linewidth": 0.6, "axes.edgecolor": "0.35", "axes.grid": True,
         "grid.color": "0.92", "grid.linewidth": 0.7, "axes.axisbelow": True, "xtick.major.width": 0.6,
         "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3, "pdf.fonttype": 42})
BASELINES = (("self_consistency_verbal", "Self-consistency + verbal confidence", "C7"), ("xiong_avgconf", "Avg-Conf (Xiong et al.)", "C3"),
             ("xiong_avgconf_para", "Avg-Conf, paraphrases (Xiong et al.)", "C5"), ("cisc", "CISC (Taubenfeld et al.)", "C4"),
             ("pedapati_lr", "Black-box calibrator (Pedapati et al.)", "C6"), ("semantic_entropy", "Semantic entropy (Kuhn et al.)", "C8"),
             ("bsdetector_like", "BSDetector (Chen & Mueller)", "C9"))
NAMES = {"cel_k": "\\textbf{CEL-$k$ (ours)}", "cel_k_borda": "\\quad CEL-$k$, Borda",
         "self_consistency": "Self-consistency"} | {n: lab for n, lab, _ in BASELINES}
UNIT = {"draw2": "2nd draw", "neg": "negation", "relabel": "relabel", "distractor": "distractor", "chal": "challenge", "verb": "verbal"}
rows, bm_rows, rk_rows = [], [], []
PANELS = (("prec_0.01", "prec@1%", "lo_0.01", "hi_0.01"), ("prec_0.05", "prec@5%", "lo_0.05", "hi_0.05"),
          ("prec", "prec@10%", "lo", "hi"), ("auroc", "AUROC", None, None), ("auarc", "AUARC", None, None))
for mk in cfg["_models"]:
    dss = cfg["_datasets"]
    fig, axes = plt.subplots(len(dss), len(PANELS), figsize=(11, 2.35 * len(dss) + 0.45), squeeze=False, sharex=True)
    for j, ds in enumerate(dss):
        r = cell_results(cfg, ds, mk, a.force)
        curve = r["budget_curve"]
        for i, (key, ylabel, lok, hik) in enumerate(PANELS):
            ax = axes[j][i]
            for prefix, label, color, marker in LINES:
                s = series(curve, prefix, key, lok, hik)
                if s is None:
                    continue
                ax.plot(s[0], s[1], "-", marker=marker, ms=3.5, lw=1.4, color=color, label=label)
                if lok and color == OURS:
                    x, y, lo, hi = (np.array(v, float) for v in s)
                    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=color, elinewidth=0.7)
            c = curve.get("certified")
            if c:                                   # one policy, spend set by the data: a point
                ax.errorbar([c["calls"]], [c[key]], yerr=[[c[key] - c[lok]], [c[hik] - c[key]]] if lok else None,
                            fmt="*", ms=9, mfc="white", mew=0.9, elinewidth=0.7, color=OURS, label="certified CEL (ours)")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(2, 15, 2))
            ax.tick_params(labelsize=8.5)
            if j == len(dss) - 1:
                ax.set_xlabel("mean calls per case")
            if i == len(PANELS) - 1 and len(dss) > 1:
                ax.text(1.03, 0.5, ds, transform=ax.transAxes, rotation=270, va="center", fontsize=9)
        prec_axes = [axes[j][i] for i, panel in enumerate(PANELS) if panel[2]]   # one scale, capped at 1
        low = min(ax.get_ylim()[0] for ax in prec_axes)
        for ax in prec_axes:
            ax.set_ylim(low, 1.01)
        bm = r["budget_matched"]
        bm_rows.append([f"\\multicolumn{{9}}{{l}}{{\\emph{{{ds} / {mk}}} (allocation chosen on "
                        f"{bm.get('n_dev', 0)} dev items by {bm.get('metric', 'n/a')})}}"])
        rk_rows.append([f"\\multicolumn{{8}}{{l}}{{\\emph{{{ds} / {mk}}}}}"])
        for kk, b in bm["by_k"].items():
            ms = {"cel_k": b["cel_k"], "cel_k_borda": b.get("cel_k_borda")} | b.get("baselines", {})
            for n, m in ms.items():
                if m is None:
                    continue
                pv = b.get("paired_vs", {}).get(n)
                d = "--" if pv is None else f"{-pv['mean_diff']:+.3f} [{-pv['ci_hi']:+.3f}, {-pv['ci_lo']:+.3f}]"
                rk_rows.append([kk if n == "cel_k" else "", NAMES.get(n, n.replace("_", " ")),
                                f"{m['prec']:.3f}", f"{m['calls']:.2f}", f"{m['acc']:.3f}", f"{m['auroc']:.3f}",
                                f"{m['auarc']:.3f}", d])
        for kk, b in bm["by_k"].items():
            p = b["paired"]
            bm_rows.append([kk, b["allocation"], f"{b['dev_calls']:.1f} / {b['cel_k']['calls']:.1f}",
                            f"{b['cel_k']['prec']:.3f}", f"{b['self_consistency']['prec']:.3f}",
                            f"{b['self_consistency_verbal']['prec']:.3f}", f"{p['mean_diff']:+.3f}", f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]",
                            f"{p['p_boot']:.3f}"])
        tgt = "certified" if "certified" in curve else "cel:cel"
        g = horizontal_gap(curve, target=tgt)
        rows.append([ds, mk, f"{curve[tgt]['calls']:.1f}",
                     "$>$max" if g == float("inf") else f"{g:.2f}$\\times$"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    title = plt.Line2D([], [], lw=0)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if len(dss) > 1 else 0.88), w_pad=1.5)
    fig.legend([title] + handles, ["method"] + labels, loc="upper center", ncol=6, frameon=False, fontsize=9,
               handlelength=1.6, columnspacing=1.2, handletextpad=0.5)
    fig.savefig(out / f"budget_curves_{mk}.pdf", bbox_inches="tight")
    fig.savefig(out / f"budget_curves_{mk}.png", dpi=200, bbox_inches="tight")   # PNG copy for the notebook
    print(f"wrote {out}/budget_curves_{mk}.pdf")
latex_table(["$k$", "CEL-$k$ allocation", "E[calls] dev / eval", "CEL-$k$ p@10", "SC-$k$ p@10",
             "SC-$k$+verb.", "$\\Delta$ vs SC", "95\\% CI", "$p$ (paired bootstrap)"], bm_rows,
            "Budget-allocated CEL-$k$ against self-consistency with $k$ draws. CEL-$k$'s allocation "
            "(screen draws; survivors' extra draws, flips, challenge, SE) is the best on the dev slice "
            "among those whose expected calls per case are at most $k$; it is compared on expected "
            "calls (batch average), self-consistency on exactly $k$. SC-$k$+verb.: self-consistency also "
            "ranking by the confidence each draw carries (the control for free verbal confidence). "
            "$\\Delta$ paired over batches.",
            "tab:budget_matched", out / "budget_matched.tex", "rlrrrrrrr")
latex_table(["Dataset", "Model", "CEL calls", "SC calls to match"], rows,
            "Horizontal gap: calls self-consistency needs to match CEL's precision@10, "
            "as a multiple of CEL's spend.", "tab:hgap", out / "horizontal_gap.tex", "llrr")
latex_table(["$k$", "Method", "prec@10\\%", "calls", "acc.", "AUROC", "AUARC", "$\\Delta$ vs CEL-$k$ [95\\% CI]"],
            rk_rows,
            "Ranking mode at a fixed budget $k$ (R1). CEL-$k$: the allocation chosen on the dev slice "
            "(expected calls per case at most $k$), ranked by $\\hat\\mu - \\lambda\\hat\\sigma$ fitted on the "
            "dev slice; every baseline spends exactly $k$ calls (semantic entropy and BSDetector: "
            "$k-1$ draws plus their extra call), CISC and the black-box calibrator refit at each $k$. "
            "prec@10\\% averaged over 10 batches; accuracy of each method's own answers; AUROC and AUARC "
            "pooled over the 500 items; $\\Delta$ = method minus CEL-$k$, paired over batches.",
            "tab:budget_curve", out / "budget_curve.tex", "llrrrrrl")
