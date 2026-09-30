"""Signal effects on confidence estimation (Sec. 3.8): per signal, the leave-one-out
effect on CEL (Delta prec@10 with CI, Delta AUROC, Delta PRR, calls saved) and the exact
Shapley value on prec@10 (with CI); plus the random->screen decomposition,
the stable-but-wrong flip increment and value per call.
Outputs: signal_effects.tex (one block per cell), shapley_cost.tex.
"""
from _paper import f, latex_table, paper_dir
from confledger.utility.config import arg_parser, cells, from_args
from confledger.evaluation_protocol import cell_results

SIG = ("L", "flip", "se", "chal", "verb")
a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
NAMES = {"L": "$L$ (lower bound)", "flip": "flip (invariance)", "se": "SE (rationales)",
         "chal": "challenge", "verb": "verbal"}
rows, rows2 = [], []
for ds, mk in cells(cfg):
    r = cell_results(cfg, ds, mk, a.force)
    sh, loo = r["shapley"], r["leave_one_out"]
    rows.append([f"\\multicolumn{{8}}{{l}}{{\\emph{{{ds} / {mk}}}}}"])
    for s in SIG:
        o, (lo, hi) = loo[s], sh["phi_ci"][s]
        rows.append([NAMES[s], f"{o['d_prec']:+.3f}", f"[{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}]",
                     f"{o['d_auroc']:+.3f}", f"{o['d_prr']:+.3f}", f"{o['calls_saved']:.1f}",
                     f"{sh['phi'][s]:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]"])
    rows2.append([ds, mk, f"{sh['screen_gain']:+.3f}", f"{sh['total_gain']:+.3f}",
                  f"{sh['flip_increment']:+.3f}"] + [f"{sh['phi_per_call'][s]:+.4f}" for s in SIG])
    order = sorted(SIG, key=lambda s: -sh["phi"][s])
    print(f"{ds}/{mk}: signal order by phi = {' > '.join(order)}  (cascade asserts L > flip > se > chal > verb)")
latex_table(["Signal", "LOO $\\Delta$p@10", "95\\% CI", "LOO $\\Delta$AUROC", "LOO $\\Delta$PRR",
             "calls saved", "$\\phi$ p@10", "95\\% CI"], rows,
            "Signal effects on confidence estimation. Leave-one-out (LOO): CEL minus CEL without the "
            "signal, CI paired over batches. Shapley $\\phi$: each signal's share of the prec@10 gain over "
            "random-among-survivors, exact over all 32 coalitions (screen fixed); CI bootstrapped "
            "over batches.", "tab:signal_effects", out / "signal_effects.tex", "lrrrrrrr")
latex_table(["Dataset", "Model", "screen", "signals", "flip incr."] + [f"$\\phi/c$ {s}" for s in SIG], rows2,
            "Gain decomposition (random $\\to$ screen $\\to$ signals), the flip increment "
            "$V(\\{L,\\mathrm{flip}\\})-V(\\{L\\})$, and value per API call.", "tab:shapley_cost",
            out / "shapley_cost.tex", "ll" + "r" * 8)
