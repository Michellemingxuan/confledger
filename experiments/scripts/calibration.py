"""Calibration figure R3 (ranking mode): reliability diagrams, one figure per model, one
column per dataset, rows = the deployment budget k = 3 and the matched budget. Only scores
that claim to be P(correct): CEL's mu (score model fitted on D_fit), self-consistency
agreement, verbalized confidence, Avg-Conf, CISC weight share, black-box calibrator (Pedapati et al.); each
against its own method's correctness, pooled over the eval items. Equal-count bins with
95% Jeffreys intervals. Table: calibration.tex (ECE, Brier, mean confidence, accuracy,
AUROC, AUARC). Secondary: nothing is tuned on these numbers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _paper import f, latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, from_args  # noqa: E402
from confledger.evaluation_protocol import cell_results  # noqa: E402

SCORES = (("cel_mu", "CEL $\\hat\\mu$ (ours)", "C2", "-o"), ("self_consistency", "SC agreement", "C1", "--s"),
          ("verbalized", "Verbalized", "0.4", ":"), ("xiong_avgconf", "Avg-Conf", "C3", "-."),
          ("xiong_avgconf_para", "Avg-Conf, paraphrases", "C5", "-."), ("cisc", "CISC", "C4", "-."),
          ("pedapati_lr", "Black-box calibrator (Pedapati et al.)", "C6", "-."))
DEPLOY_K = "3"

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
sim = " (SIMULATED: pipeline check, not evidence)" if cfg.get("synthetic") else ""
rows = []
for mk in cfg["_models"]:
    dss = cfg["_datasets"]
    fig, axes = plt.subplots(2, len(dss), figsize=(3.6 * len(dss), 6.4), squeeze=False, sharex=True, sharey=True)
    for j, ds in enumerate(dss):
        r = cell_results(cfg, ds, mk, a.force)
        panels = (("k = 3 (deployment)", r.get("budget_matched", {}).get("by_k", {}).get(DEPLOY_K, {}).get("calibration", {})),
                  (f"matched budget ({r['matched_budget']} calls)", r.get("calibration", {})))
        for i, (title, cal) in enumerate(panels):
            ax = axes[i][j]
            ax.plot([0, 1], [0, 1], color="0.6", lw=0.8)
            for key, label, color, style in SCORES:
                c = cal.get(key)
                if not c:
                    continue
                rel = c["reliability"]
                x, y = [b["conf"] for b in rel], [b["acc"] for b in rel]
                ax.plot(x, y, style, color=color, ms=3, lw=1.2, label=label)
                if key == "cel_mu":
                    ax.fill_between(x, [b["lo"] for b in rel], [b["hi"] for b in rel], color=color, alpha=0.12, lw=0)
                rows.append([ds, mk, title, label, f(c["ece"]), f(c["brier"]), f(c["mean_conf"]), f(c["acc"]),
                             f(c["auroc"]), f(c["auarc"])])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{ds}: {title}", fontsize=8)
            if i == 1:
                ax.set_xlabel("stated confidence (bin mean)")
            if j == 0:
                ax.set_ylabel("accuracy in the bin")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0), ncol=min(4, 2 * len(dss)), frameon=False, fontsize=7.5)
    fig.suptitle(f"Reliability: below the diagonal = overconfident{sim}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / f"calibration_{mk}.pdf", bbox_inches="tight")
    fig.savefig(out / f"calibration_{mk}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {out}/calibration_{mk}.pdf")
latex_table(["Dataset", "Model", "Budget", "Score", "ECE", "Brier", "mean conf.", "acc.", "AUROC", "AUARC"], rows,
            "Calibration of the scores that claim to be $P(\\text{correct})$ (R3), each against its own "
            "method's correctness, pooled over the evaluation items. ECE with 10 equal-width bins. "
            "CEL's ranking score $\\hat\\mu - \\lambda\\hat\\sigma$ is not a probability and is not listed; "
            "$\\hat\\mu$ is fitted on the dev slice (model-based)." + sim,
            "tab:calibration", out / "calibration.tex", "llllrrrrrr")
