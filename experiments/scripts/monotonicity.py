"""Assumption 1 tests: stratum accuracy grid figure, isotonic gap, and the
per-batch top-strata test where Prop. 1 consumes the assumption.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paper import f, latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, cells, from_args  # noqa: E402
from confledger.evaluation_protocol import cell_results  # noqa: E402

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
cl = list(cells(cfg))
fig, axes = plt.subplots(1, len(cl), figsize=(3.2 * len(cl), 3), squeeze=False, sharey=True)
rows = []
for ax, (ds, mk) in zip(axes[0], cl):
    mono = cell_results(cfg, ds, mk, a.force)["monotonicity"]
    st = mono["strata"]
    x = [s["stratum"] for s in st]
    y = [s["acc"] for s in st]
    ax.errorbar(x, y, yerr=[[v - s["lo"] for v, s in zip(y, st)], [s["hi"] - v for v, s in zip(y, st)]],
                fmt="o-", capsize=2, ms=3)
    ax.set_title(f"{ds} / {mk}", fontsize=9)
    ax.set_xlabel("rank stratum")
    g, t = mono["isotonic_gap"], mono["top_strata"]
    rows.append([ds, mk, f(g["gap"], 4), f(t["1-10"]["acc"]), f(t["11-30"]["acc"]), f(t["31-50"]["acc"]),
                 f(t.get("p_1-10>11-30")), f(t.get("p_11-30>31-50"))])
axes[0][0].set_ylabel("accuracy")
fig.tight_layout()
fig.savefig(out / "monotonicity.pdf")
fig.savefig(out / "monotonicity.png", dpi=150)   # PNG copy for the notebook
print(f"wrote {out/'monotonicity.pdf'}")
latex_table(["Dataset", "Model", "Iso.\\ gap", "Acc 1--10", "Acc 11--30", "Acc 31--50",
             "$p_{1\\text{--}10>11\\text{--}30}$", "$p_{11\\text{--}30>31\\text{--}50}$"], rows,
            "Monotonicity checks. Isotonic gap = Brier cost of imposing monotonicity on stratum "
            "accuracies; strata are per-batch ranks pooled over batches.", "tab:mono",
            out / "monotonicity.tex", "llrrrrrr")
