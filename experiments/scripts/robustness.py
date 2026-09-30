"""Robustness and label-free audits: selection stability (Jaccard vs chance floor,
plus the tie-aware variant), held-out probe survival, and judge sensitivity.
"""
from _paper import f, latex_table, paper_dir
from confledger.utility.config import arg_parser, cells, from_args
from confledger.evaluation_protocol import cell_results

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
rows = []
for ds, mk in cells(cfg):
    r = cell_results(cfg, ds, mk, a.force)
    j, h = r["jaccard"], r["heldout_audit"]
    js = r.get("judge_sensitivity", {})
    rows.append([ds, mk, f(j.get("mean")), f(j["chance_floor"]), f(j.get("tie_aware")),
                 f(j.get("top_group_size"), 1), f(h["selected_pass"]), f(h["control_pass"]),
                 f(h["fisher_p"]), f(js.get("se_spearman")), f(js.get("top10_jaccard"))])
latex_table(["Dataset", "Model", "Jaccard", "floor", "tie-aware", "top group",
             "held-out sel.", "held-out ctl.", "$p$", "judge $\\rho$", "judge J"], rows,
            "Robustness. Selection stability under fully re-drawn stage 2 (Jaccard vs.\\ the chance "
            "floor; tie-aware = share of re-drawn selections inside the original top tie-group); "
            "held-out probe survival for selected (ranks 1--10) vs.\\ matched controls (11--30); "
            "judge-swap sensitivity where available.", "tab:robust", out / "robustness.tex", "ll" + "r" * 9)
