"""SPRT characteristics (Sec. 3): the derived constants plus the exact operating
characteristic (OC) of the TRUNCATED two-stage procedure (the simulation promised in Sec. 3.2).

Certification happens iff the total log-LR after s1+s2 draws reaches A; the stage-2
depth s2 is derived from the config's `sprt:` block (certification_plan). Also the
sensitivity table: how H0/H1 and the error rates trade draws for guarantee strength.
Everything is exact binomial arithmetic -- no simulation needed.
Outputs: sprt_constants.tex, truncation.tex (OC at p0, p1), truncation_oc.pdf,
sprt_settings.tex (sensitivity).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _paper import f, latex_table, paper_dir  # noqa: E402
from confledger.utility.config import arg_parser, from_args  # noqa: E402

from confledger.evaluation_protocol import theory_constants  # noqa: E402
from confledger.methods import cel_from_settings  # noqa: E402
from confledger.sprt_posterior import SPRT, certification_plan, tail_mass, truncated_oc  # noqa: E402

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
base, sprt, plan = cel_from_settings(cfg.get("sprt"))      # the run's settings (paper defaults if absent)
S2 = plan["s2"]
print(f"certification: {plan['depth']} unanimous draws (SPRT {plan['sprt_depth']}, Cor. 1 "
      f"{plan['cor1_depth']}; {plan['binding']} binding), stage 2 = {S2} draws")


t = theory_constants(sprt)
print(json.dumps(t, indent=2))
latex_table(["Quantity", "Value"], [
    ["Agreement / disagreement step (nats)", f"{f(t['sprt']['up'])} / {f(t['sprt']['down'])}"],
    ["Boundaries $A$ / $B$", f"{f(t['sprt']['A'])} / {f(t['sprt']['B'])}"],
    ["$\\mathbb{E}[N]$ under $H_0$ / $H_1$ (Wald)", f"{f(t['sprt']['expected_n']['H0'], 1)} / {f(t['sprt']['expected_n']['H1'], 1)}"],
    ["Screen pass at $\\bar p$=0.7 / 0.95", f"{f(t['screen_pass']['p=0.7'])} / {f(t['screen_pass']['p=0.95'])}"],
    ["Certification depth, $\\varepsilon_0$=0.05 / 0.02", f"{t['certification_depth']['eps0=0.05']} / {t['certification_depth']['eps0=0.02']}"],
    ["$\\varepsilon$ at 9/9, 12/12, 8/9", " / ".join(f(v) for v in t["tail_mass"].values())],
    ["$\\kappa$ negation, relabel, distractor", " / ".join(f(t["kappa"][g], 2) for g in ("neg", "relabel", "distractor"))],
], "Derived constants of the methodology (all exact).", "tab:theory", out / "sprt_constants.tex", "lr")

rows = []
for s2 in sorted({2, S2, S2 + 3}):
    o0, o1 = truncated_oc(sprt.p0, s2=s2, sprt=sprt), truncated_oc(sprt.p1, s2=s2, sprt=sprt)
    rows.append([f"3 + {s2}", f(o0["certify"]), f(1 - o1["certify"]), f(o0["calls"], 1), f(o1["calls"], 1)])
latex_table(["Draws", "P(certify $\\mid p_0$)", "P(not certify $\\mid p_1$)",
             "E[calls $\\mid p_0$]", "E[calls $\\mid p_1$]"], rows,
            f"Exact operating characteristics of the truncated SPRT (nominal $\\alpha=\\beta={sprt.alpha}$).",
            "tab:truncation", out / "truncation.tex")

ps = np.linspace(0.5, 1.0, 101)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
for s2, ls in zip(sorted({2, S2, S2 + 3}), (":", "-", "--")):
    ax.plot(ps, [truncated_oc(p, s2=s2, sprt=sprt)["certify"] for p in ps], "-" if s2 == S2 else ls,
            lw=2 if s2 == S2 else 1.2, label=f"certify, 3+{s2} draws" + (" (configured)" if s2 == S2 else ""))
ax.plot(ps, [truncated_oc(p)["eliminate"] for p in ps], color="gray", label="eliminated at screen")
for p in (sprt.p0, sprt.p1):
    ax.axvline(p, color="k", lw=0.5, alpha=0.4)
ax.set_xlabel(r"true stability $\bar p$")
ax.set_ylabel("probability")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "truncation_oc.pdf")
fig.savefig(out / "truncation_oc.png", dpi=150)   # PNG copy for the notebook
print(f"wrote {out/'truncation_oc.pdf'}")
for s2 in (S2,):
    o0, o1 = truncated_oc(sprt.p0, s2=s2, sprt=sprt), truncated_oc(sprt.p1, s2=s2, sprt=sprt)
    print(f"3+{s2}: false-certify at p0 = {o0['certify']:.3f} (nominal {sprt.alpha}); "
          f"miss at p1 = {1-o1['certify']:.3f} (nominal {sprt.beta})")

# Sensitivity: fewer draws for a weaker certificate. Each row certifies at its SPRT depth,
# which requires eps0 relaxed to the tail mass at that depth (Cor. 1 otherwise binds).
p0s = plan["settings"]["p0_star"]
GRID = [("paper default (Cor.~1 raises it to 9)", 0.70, 0.95, 0.10), ("H0 p0 = 0.6", 0.60, 0.95, 0.10), ("H0 p0 = 0.5", 0.50, 0.95, 0.10),
        ("H1 p1 = 0.9", 0.70, 0.90, 0.10), ("alpha = beta = 0.2", 0.70, 0.95, 0.20),
        ("alpha = beta = 0.05", 0.70, 0.95, 0.05)]
srows = []
for name, q0, q1, err in GRID:
    s_ = SPRT(q0, q1, err, err)
    d = certification_plan(s_, eps0=1.0, p0_star=p0s)["sprt_depth"]      # SPRT depth alone
    o0, o1 = truncated_oc(q0, s2=max(0, d - 3), sprt=s_), truncated_oc(q1, s2=max(0, d - 3), sprt=s_)
    srows.append([name, f(s_.up), str(d), f(tail_mass(d, d, p0s)), f(o0["certify"]), f(1 - o1["certify"]),
                  f(1 + o0["calls"], 1), f(1 + o1["calls"], 1)])
latex_table(["Setting", "nats/agree", "draws", "$\\varepsilon_0$ needed", "false cert.", "missed cert.",
             "calls $\\mid p_0$", "calls $\\mid p_1$"], srows,
            "Certification cost vs.\\ guarantee strength. \\emph{draws}: unanimous draws the SPRT needs; "
            f"$\\varepsilon_0$ needed: Cor.~1 tail mass at that depth ($p_0^*={p0s}$), which the certificate "
            "must accept; error rates and expected calls (paraphrase generation + draws) exact for the "
            "truncated procedure.", "tab:sprt_settings", out / "sprt_settings.tex", "lrrrrrrr")
