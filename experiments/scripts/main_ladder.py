"""Main result (Sec. 4.4-4.5): baseline ladder + ablations at matched budget,
for every (dataset, model) cell, with the Holm-corrected confirmatory test.

Outputs: main_ladder.tex (one block per cell) and confirmatory.tex.
"""
from _paper import f, latex_table, paper_dir
from confledger.utility.config import arg_parser, cells, from_args
from confledger.evaluation_protocol import cell_results

from confledger.metrics import holm

ORDER = [("random", "Random"), ("verbalized", "Verbalized"),
         ("self_consistency", "Self-consistency"),
         ("self_consistency_deploy", "\\quad deployment baseline (fixed draws)"), ("semantic_entropy", "Semantic entropy"),
         ("bsdetector_like", "BSDetector-style"), ("cel", "\\textbf{CEL (ours)}"),
         ("cel_borda", "\\quad Borda combiner"), ("cel_no_spread", "\\quad no spread"),
         ("cel_no_sprt", "\\quad no SPRT"), ("cel_no_flip", "\\quad no invariance"),
         ("cel_no_se", "\\quad no SE"), ("cel_no_chal", "\\quad no challenge"),
         ("cel_no_verb", "\\quad no verbal"), ("cel_no_L", "\\quad no $L$")]

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
rows, conf = [], {}
for ds, mk in cells(cfg):
    r = cell_results(cfg, ds, mk, a.force)
    rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{ds} / {mk}}} (matched budget {r['matched_budget']} calls)}}"])
    for key, name in ORDER:
        m = r["methods"][key]
        pc = r["paired_vs_cel"].get(key)
        delta = "--" if pc is None else f"{-pc['mean_diff']:+.3f}"
        rows.append([name, f(m["mean_prec"]), f(m["mean_calls"], 1), f(m["auroc"]), f(m["prr"]), delta])
    conf[f"{ds}/{mk}"] = r["confirmatory"]
latex_table(["Method", "prec@10", "calls", "AUROC", "PRR", "$\\Delta$ vs CEL"], rows,
            "Baseline ladder and ablations. prec@10 averaged over 10 batches; baselines "
            "matched to CEL's realised calls per case.", "tab:main", out / "main_ladder.tex")

adj = holm({k: v["wilcoxon_p"] for k, v in conf.items()})
crow = [[k, f"{v['mean_diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]",
         f"{v['wins']}/{v['n']}", f(v["wilcoxon_p"]), f(adj[k])] for k, v in conf.items()]
latex_table(["Cell", "$\\Delta$ prec@10", "95\\% CI", "wins", "$p$", "$p_{\\mathrm{Holm}}$"], crow,
            "Confirmatory comparison: CEL minus matched-budget self-consistency.",
            "tab:confirmatory", out / "confirmatory.tex")
for k, v in conf.items():
    print(f"{k:28s} Δ={v['mean_diff']:+.3f} [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] p_holm={adj[k]:.3f}")
