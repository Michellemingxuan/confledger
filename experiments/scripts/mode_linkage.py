"""Mode-linkage figure (Sec. 3.9): CEL-k at a growing budget runs into adaptive CEL.
One column per dataset, one figure per model; x = mean calls per case.
Two curves, CEL-k under the dev-chosen allocation (the CEL-k of the budget curve, R1) and
under the derived allocation (Prop. 3, label-free), both ranked by mu - lambda*sigma fitted
on D_fit, and one point, adaptive CEL. Top: prec@10% with a 95% bootstrap band over batches.
Bottom: share of the delivered top 10% certified -- by the SPRT certificate at (eps0, p0*),
exact for stability (Prop. 2, solid) and model-based (Prop. 3, dotted), and by conformal BH
at the middle alpha, exact for correctness (dashed, needs D_cal). Shading splits the axis at the two certification budgets (Cor. 2):
no allocation can certify / only the model-based certificate is reachable / the exact one is.
Table: mode_linkage.tex (per budget: dev-chosen vs derived allocation; the nesting check).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paper import f, latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, from_args  # noqa: E402
from confledger.evaluation_protocol import cell_results  # noqa: E402

CELK, CELD, CEL = "C4", "C2", "C3"      # derived, dev-chosen (as in budget_curves), adaptive


def series(points: dict, *keys):
    pts = sorted(tuple(v[k] for k in ("calls",) + keys) for v in points.values())
    return list(zip(*pts))


a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
sim = " (SIMULATED: pipeline check, not evidence)" if cfg.get("synthetic") else ""
rows = []
for mk in cfg["_models"]:
    dss = cfg["_datasets"]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.6 * len(dss), 5.4), squeeze=False, sharex=True)
    for j, ds in enumerate(dss):
        L = cell_results(cfg, ds, mk, a.force).get("linkage")
        if L is None or "lo" not in L["adaptive"]["cel"]:
            raise SystemExit(f"{ds}/{mk}: no (current) linkage results -- rerun with --force")
        cel, B, Bm = L["adaptive"]["cel"], L["certification_budget"]["exact"], L["certification_budget"]["model"]
        x, p, lo, hi = series(L["derived"], "prec", "lo", "hi")
        _, ce, cm = series(L["derived"], "certified_exact", "certified_model")
        xd, pd_, lod, hid = series(L["celk"], "prec", "lo", "hi")
        conf = {kk: v for kk, v in L["celk"].items() if "certified_conformal" in v}
        xmax = max(max(x), max(xd), cel["calls"]) + 1
        for i, ax in enumerate(axes[:, j]):
            if Bm is not None:
                ax.axvspan(0, Bm, color="0.88", zorder=0)
                ax.axvline(Bm, color="0.5", lw=0.8, ls=":", zorder=1)
            if B is not None:
                ax.axvspan(Bm or 0, B, color="0.95", zorder=0)
                ax.axvline(B, color="0.5", lw=0.8, zorder=1)
            ax.set_xlim(1, xmax)
        top, bot = axes[0][j], axes[1][j]
        top.fill_between(x, lo, hi, color=CELK, alpha=0.15, lw=0)
        top.plot(x, p, "-o", color=CELK, ms=4, label="CEL-k (budget k, derived allocation)")
        top.fill_between(xd, lod, hid, color=CELD, alpha=0.12, lw=0)
        top.plot(xd, pd_, "-^", color=CELD, ms=4, label="CEL-k (budget k, dev-chosen allocation)")
        top.errorbar(cel["calls"], cel["prec"], yerr=[[cel["prec"] - cel["lo"]], [cel["hi"] - cel["prec"]]],
                     fmt="*", ms=13, color=CEL, capsize=3, label="adaptive CEL")
        top.set_title(ds)
        bot.plot(x, ce, "-o", color=CELK, ms=4)
        bot.plot(x, cm, ":", color=CELK, lw=1.2)
        if conf:
            xc, cc = series(conf, "certified_conformal")
            bot.plot(xc, cc, "--^", color=CELD, ms=4, lw=1.2)
        bot.plot(cel["calls"], cel["certified_exact"], "*", ms=13, color=CEL)
        bot.plot(cel["calls"], cel["certified_model"], "*", ms=13, mfc="none", color=CEL)
        bot.set_ylim(-0.04, 1.04)
        bot.set_xlabel("mean calls per case")
        if B is not None and Bm is not None:
            y = top.get_ylim()[0] + 0.04 * (top.get_ylim()[1] - top.get_ylim()[0])
            for x0, x1, txt in ((1, Bm, "ranking\nonly"), (Bm, B, "model-\nbased"), (B, xmax, "exact")):
                top.text((x0 + x1) / 2, y, txt, ha="center", fontsize=6.5, color="0.3", linespacing=0.9)
        if j == 0:
            top.set_ylabel("prec@10%")
            bot.set_ylabel("certified share of top 10%")
        n = L["nested"]
        for k, d in L["derived"].items():
            ck = L["celk"].get(k, {})
            rows.append([ds, mk, k, ck.get("allocation", "--"), f(ck.get("prec")), d["allocation"],
                         f"{d['k_eff']:.1f}", f(d.get("prec"))])
        rows.append([ds, mk, "\\multicolumn{6}{l}{certification budget: exact "
                     f"{f(B, 1)}, model-based {f(L['certification_budget'].get('model'), 1)} calls. "
                     f"Adaptive CEL {f(cel['calls'], 1)} calls, p@10 {f(cel['prec'])}; "
                     f"replayed as an allocation p@10 {f(n['prec'])}}}"])
    h = [plt.Line2D([], [], color=CELK, marker="o", ms=4, label="CEL-k at budget k (derived allocation)"),
         plt.Line2D([], [], color=CELD, marker="^", ms=4, label="CEL-k at budget k (dev-chosen; as in R1)"),
         plt.Line2D([], [], color=CELD, ls="--", label="certified: conformal BH, exact for correctness"),
         plt.Line2D([], [], color=CEL, marker="*", ms=11, ls="none", label="adaptive CEL"),
         plt.Line2D([], [], color="0.3", ls="-", label="certified: SPRT, exact for stability (Prop. 2)"),
         plt.Line2D([], [], color="0.3", ls=":", label="certified: model-based (Prop. 3; hollow star)"),
         plt.Rectangle((0, 0), 1, 1, color="0.88", label="no certificate reachable (Cor. 2)"),
         plt.Rectangle((0, 0), 1, 1, color="0.95", label="model-based certificate only (Cor. 2)")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2 if len(dss) == 1 else 3, frameon=False, fontsize=7.5)
    fig.suptitle(f"CEL-k at growing budget reaches adaptive CEL{sim}", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    fig.savefig(out / f"mode_linkage_{mk}.pdf", bbox_inches="tight")
    fig.savefig(out / f"mode_linkage_{mk}.png", dpi=150, bbox_inches="tight")   # PNG copy for the notebook
    print(f"wrote {out}/mode_linkage_{mk}.pdf")
latex_table(["Dataset", "Model", "$k$", "Dev-chosen allocation", "p@10", "Derived allocation",
             "$\\keff$", "p@10"], rows,
            "Mode linkage (Sec.~\\ref{sec:method:budget}). Per budget $k$ (expected calls per case), the "
            "CEL-$k$ allocation chosen on the dev slice and the allocation derived from "
            "Proposition~\\ref{prop:exchange} (maximal $\\keff$ of a clean survivor, label-free), each "
            "with its evaluation precision@10; per cell, the certification budgets and the nesting check "
            "(adaptive CEL vs.\\ its allocation replayed and ranked like CEL-$k$, Proposition~\\ref{prop:nesting}). Both CEL-$k$ columns rank by $\\hat\\mu - \\lambda\\hat\\sigma$ fitted on the dev slice." + sim,
            "tab:mode_linkage", out / "mode_linkage.tex", "lllllrlr")
