"""Headline figure (Sec. 4.5): quality vs calls per case, one column per dataset,
rows = prec@10 and AUROC, one figure per model: adaptive CEL, budget-allocated CEL-k
(allocation chosen on the dev slice, plotted at its expected calls per case) and
self-consistency, with the deployment baseline ringed. Tables: budget_matched.tex (CEL-k vs
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
from confledger.utility.cell_report import mark_deployment_baseline  # noqa: E402


def series(curve, prefix, key):
    pts = sorted((v["calls"], v[key], v.get("lo"), v.get("hi")) for k, v in curve.items()
                 if k.startswith(prefix))
    return list(zip(*pts)) if pts else None



a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
# fixed colours: self-consistency keeps C1, the colour of the deployment-baseline ring
LINES = (("cel:", "CEL, adaptive (ours)", "-o", "C0"), ("celk:", "CEL-k, budget-allocated (ours)", "-^", "C2"),
         ("sc:", "Self-consistency", "--s", "C1"))
UNIT = {"draw2": "2nd draw", "neg": "negation", "relabel": "relabel", "distractor": "distractor", "chal": "challenge", "verb": "verbal"}
rows, bm_rows = [], []
for mk in cfg["_models"]:
    dss = cfg["_datasets"]
    fig, axes = plt.subplots(2, len(dss), figsize=(4 * len(dss), 6), squeeze=False)
    for j, ds in enumerate(dss):
        r = cell_results(cfg, ds, mk, a.force)
        curve = r["budget_curve"]
        for i, key in enumerate(("prec", "auroc")):
            ax = axes[i][j]
            for prefix, label, style, color in LINES:
                s = series(curve, prefix, key)
                if s is None:
                    continue
                ax.plot(s[0], s[1], style, color=color, label=label)
                if key == "prec":
                    ax.fill_between(s[0], s[2], s[3], color=color, alpha=0.15)
            mark_deployment_baseline(ax, curve, r.get("deploy_sc_draws", 3), key)
            ax.axhline(curve["verbalized"][key], ls=":", color="gray", label="Verbalized")
            ax.set_title(ds if i == 0 else "")
            ax.set_ylabel("prec@10" if key == "prec" else "AUROC")
            if i == 1:
                ax.set_xlabel("mean calls per case")
        bm = r["budget_matched"]
        bm_rows.append([f"\\multicolumn{{9}}{{l}}{{\\emph{{{ds} / {mk}}} (allocation chosen on "
                        f"{bm.get('n_dev', 0)} dev items by {bm.get('metric', 'n/a')})}}"])
        for kk, b in bm["by_k"].items():
            p = b["paired"]
            bm_rows.append([kk, b["allocation"], f"{b['dev_calls']:.1f} / {b['cel_k']['calls']:.1f}",
                            f"{b['cel_k']['prec']:.3f}", f"{b['self_consistency']['prec']:.3f}",
                            f"{b['self_consistency_verbal']['prec']:.3f}", f"{p['mean_diff']:+.3f}", f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]",
                            f"{p['wins']}/{p['n']}"])
        g = horizontal_gap(curve)
        rows.append([ds, mk, f"{curve['cel:cel']['calls']:.1f}",
                     "$>$max" if g == float("inf") else f"{g:.2f}$\\times$"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / f"budget_curves_{mk}.pdf")
    fig.savefig(out / f"budget_curves_{mk}.png", dpi=150)   # PNG copy for the notebook
    print(f"wrote {out}/budget_curves_{mk}.pdf")
latex_table(["$k$", "CEL-$k$ allocation", "E[calls] dev / eval", "CEL-$k$ p@10", "SC-$k$ p@10",
             "SC-$k$+verb.", "$\\Delta$ vs SC", "95\\% CI", "wins"], bm_rows,
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
