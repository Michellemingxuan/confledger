"""Predicted-shape sketches for docs/result.md. Not data: no values on the quality axes."""
import sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
MARK = dict(x=0.99, y=0.99, s="SKETCH · NOT DATA", ha="right", va="top", color="#b45309", fontsize=8, family="monospace")

def stamp(ax): ax.text(transform=ax.transAxes, **MARK)

# R1 budget curve
k = np.array([2, 3, 4, 5, 6, 8])
fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharex=True)
for ax, lift, title in ((axes[0], 1.0, "prec@10"), (axes[1], 0.6, "AUROC")):
    sat = lambda a, r: a + (0.92 - a) * (1 - np.exp(-r * (k - 1)))
    ax.plot(k, sat(0.70, 0.55) + 0.02 * lift, "-^", color="C2", lw=2, label="CEL-k (ours), μ̂ − λσ̂")
    ax.fill_between(k, sat(0.70, 0.55) - 0.015, sat(0.70, 0.55) + 0.045 * lift, color="C2", alpha=0.12)
    ax.plot(k, sat(0.60, 0.25), "--s", color="C1", label="self-consistency")
    ax.plot(k, sat(0.62, 0.27) + 0.01, ":", marker=".", color="C7", label="SC + verbal (control)")
    ax.plot(k, sat(0.62, 0.30), "-.", lw=1, color="C3", label="Avg-Conf (Xiong)")
    ax.plot(k[k >= 2], sat(0.63, 0.33)[k >= 2], "-.", lw=1, color="C5", label="Avg-Conf, paraphrases")
    ax.plot(k, sat(0.61, 0.26), "-.", lw=1, color="C4", label="CISC")
    ax.plot(k, sat(0.60, 0.25) + 0.005, "-.", lw=1, color="C6", label="Pedapati-style")
    ax.plot(k[k >= 3], sat(0.58, 0.30)[k >= 3], "-.", lw=1, color="C8", label="semantic entropy")
    ax.axhline(0.66, ls=":", color="0.5"); ax.text(8, 0.665, "verbalized (1 call)", ha="right", va="bottom", fontsize=7, color="0.4")
    ax.axhline(0.55, ls="-", lw=0.6, color="0.7"); ax.text(8, 0.555, "random = base accuracy", ha="right", va="bottom", fontsize=7, color="0.5")
    ax.set_ylabel(title); ax.set_yticks([]); stamp(ax)
    ax.axvline(4, color="0.35", ls="--", lw=0.9)
axes[1].text(4.08, 0.57, "elbow k* (chosen on dev):\nboth modes compared here", fontsize=7, color="0.3")
sat = lambda a, r, x=3: a + (0.92 - a) * (1 - np.exp(-r * (x - 1)))
y_cel, y_sc = sat(0.70, 0.55) + 0.02, sat(0.60, 0.25)
axes[0].annotate("", xy=(2.85, y_cel - 0.005), xytext=(2.85, y_sc + 0.005), arrowprops=dict(arrowstyle="<->", color="0.2"))
axes[0].text(3.0, (y_cel + y_sc) / 2 + 0.01, "the claim: gap at k = 2–3", fontsize=7.5, ha="left", va="center", bbox=dict(fc="white", ec="none", pad=1))
axes[1].set_xlabel("calls per case k (CEL-k: batch average ≤ k)"); axes[1].set_xticks(k)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f"{out}/r1_budget_curve_sketch.png", dpi=200); plt.close(fig)

# R3 reliability
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
x = np.array([.55, .65, .75, .85, .93])
ax.plot(x, x + np.array([.02, -.01, .01, -.02, -.01]), "-o", color="C2", label="CEL μ̂ (fitted on D_fit)")
ax.fill_between(x, x - .07, x + .07, color="C2", alpha=.12)
ax.plot([.625, .75, .875, 1.0], [.58, .66, .74, .86], "--s", color="C1", label="SC agreement m/k")
ax.plot([.8, .85, .9, .95], [.66, .70, .74, .80], ":", marker=".", color="0.4", label="verbalized")
ax.plot([.7, .8, .9, .97], [.62, .68, .76, .84], "-.", color="C4", label="CISC weight share")
ax.set_xlim(.4, 1.01); ax.set_ylim(.4, 1.01)
ax.set_xlabel("stated confidence (bin mean)"); ax.set_ylabel("accuracy in the bin")
ax.text(.98, .45, "below the diagonal\n= overconfident", ha="right", fontsize=7.5, color="0.35")
ax.legend(frameon=False, fontsize=7, loc="upper left"); stamp(ax)
fig.tight_layout(); fig.savefig(f"{out}/r3_calibration_sketch.png", dpi=200); plt.close(fig)

# R4 mode linkage
kk = np.array([2, 3, 4, 5, 6, 8, 10, 12])
fig, (top, bot) = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)
Bm, B = 6.8, 9.0
for ax in (top, bot):
    ax.axvspan(1, Bm, color="0.9", zorder=0); ax.axvspan(Bm, B, color="0.96", zorder=0)
    ax.axvline(Bm, color="0.5", ls=":", lw=.8); ax.axvline(B, color="0.5", lw=.8); ax.set_xlim(1, 14)
pr = .72 + .2 * (1 - np.exp(-.35 * (kk - 1)))
top.plot(kk, pr, "-^", color="C2", label="CEL-k, dev-chosen allocation (as in R1)")
top.plot(kk, pr - .015, "-o", color="C4", ms=4, label="CEL-k, derived allocation a*(k)")
top.plot(12.8, pr[-1] + .005, "*", ms=14, color="C3", label="adaptive CEL")
top.set_ylabel("prec@10"); top.set_yticks([]); stamp(top)
for x0, x1, t in ((1, Bm, "no certificate\nreachable"), (Bm, B, "model-\nbased"), (B, 14, "exact")):
    top.text((x0 + x1) / 2, pr.min() - .01, t, ha="center", va="top", fontsize=7, color="0.35")
top.set_ylim(pr.min() - .06, pr.max() + .04)
bot.plot(kk, np.where(kk > B, .2 + .5 * (1 - np.exp(-.6 * (kk - B))), 0), "-^", color="C2", label="SPRT, exact (stability)")
bot.plot(kk, np.where(kk > Bm, .3 + .5 * (1 - np.exp(-.6 * (kk - Bm))), 0), ":", color="C2", label="SPRT, model-based")
bot.plot(kk, .15 + .55 * (1 - np.exp(-.3 * (kk - 1))), "--", color="C2", label="conformal BH, exact (correctness)")
bot.plot(12.8, .72, "*", ms=14, color="C3")
bot.set_ylabel("certified share of top-10"); bot.set_ylim(-.04, 1); bot.set_xlabel("mean calls per case"); stamp(bot)
h1, l1 = top.get_legend_handles_labels(); h2, l2 = bot.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f"{out}/r4_mode_linkage_sketch.png", dpi=200); plt.close(fig)
print("ok")
