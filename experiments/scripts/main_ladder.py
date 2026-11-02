"""Main result (Sec. 4.4-4.5): baseline ladder + ablations at matched budget,
for every (dataset, model) cell, with the Holm-corrected confirmatory test.

Outputs: main_ladder.tex (one block per cell), confirmatory.tex, certified.tex (R2b:
conformal BH at the matched budget for CEL and every baseline, plus CEL's SPRT certificate),
and elbow.tex (R2: both modes at the elbow k* of CEL-k's dev curve -- CEL-k and every
baseline at k* calls, certified CEL at its own spend -- with conformal power at alpha = 0.1).
"""
from _paper import f, latex_table, paper_dir
from confledger.utility.config import arg_parser, cells, from_args
from confledger.evaluation_protocol import cell_results

from confledger.metrics import holm

ORDER = [("random", "Random"), ("verbalized", "Verbalized"),
         ("self_consistency", "Self-consistency"),
         ("self_consistency_deploy", "\\quad deployment baseline (fixed draws)"), ("semantic_entropy", "Semantic entropy"),
         ("bsdetector_like", "BSDetector \\citep{chen2024}"),
         ("xiong_avgconf", "Avg-Conf, one phrasing \\citep{xiong2023}"),
         ("xiong_avgconf_para", "Avg-Conf, paraphrases \\citep{xiong2023}"),
         ("cisc", "CISC, verbal \\citep{taubenfeld2025}"),
         ("cisc_p_answer", "\\quad CISC, answer-token prob.\\ (gray-box)"),
         ("pedapati_lr", "Black-box calibrator \\citep{pedapati2024}"),
         ("cel", "\\textbf{CEL (ours)}"),
         ("cel_scored", "\\quad ranked by $\\hat\\mu - \\lambda\\hat\\sigma$"),
         ("cel_borda", "\\quad Borda combiner"), ("cel_no_spread", "\\quad no spread"),
         ("cel_no_sprt", "\\quad no SPRT"), ("cel_no_flip", "\\quad no invariance"),
         ("cel_no_se", "\\quad no SE"), ("cel_no_chal", "\\quad no challenge"),
         ("cel_no_verb", "\\quad no verbal"), ("cel_no_L", "\\quad no $L$")]

a = arg_parser(__doc__).parse_args()
cfg = from_args(a)
out = paper_dir(cfg)
rows, conf, cert_rows, el_rows = [], {}, [], []
EL_NAMES = {"cel_k": "\\textbf{CEL-$k$ (ours)}, $\\hat\\mu - \\lambda\\hat\\sigma$",
            "self_consistency_verbal": "Self-consistency + verbal confidence"} | {key: name for key, name in ORDER}
CERT = {"cel": "\\textbf{CEL (ours)}, $\\hat\\mu - \\lambda\\hat\\sigma$"} | {key: name for key, name in ORDER}
for ds, mk in cells(cfg):
    r = cell_results(cfg, ds, mk, a.force)
    rows.append([f"\\multicolumn{{8}}{{l}}{{\\emph{{{ds} / {mk}}} (matched budget {r['matched_budget']} calls)}}"])
    for key, name in ORDER:
        if key not in r["methods"]:          # e.g. Avg-Conf/CISC without per-draw confidence
            continue
        m = r["methods"][key]
        pc = r["paired_vs_cel"].get(key)
        delta = "--" if pc is None else f"{-pc['mean_diff']:+.3f}"
        rows.append([name, f(m["mean_prec"]), f(m["mean_calls"], 1), f(m["base_acc"]), f(m["auroc"]),
                     f(m.get("auarc")), f(m["prr"]), delta])
    conf[f"{ds}/{mk}"] = r["confirmatory"]
    el = r.get("elbow")
    if el:
        ks = el["k_star"]
        el_rows.append([f"\\multicolumn{{8}}{{l}}{{\\emph{{{ds} / {mk}}}: elbow $k^* = {ks}$ (dev), CEL-$k$ plan {el['allocation']}}}"])
        conf_a = el.get("conformal", {}).get("by_alpha", {}).get("0.1", {})
        for key, m in el["methods"].items():
            pv = el["paired_vs"].get(key)
            d = "--" if pv is None else f"{-pv['mean_diff']:+.3f}"
            pw = conf_a.get(key, {}).get("power")
            el_rows.append([EL_NAMES.get(key, key.replace("_", " ")), f(m["prec"]), f(m["calls"], 2), f(m["acc"]),
                            f(m["auroc"]), f(m["auarc"]), f(pw), d])
        c = el["certified_cel"]
        el_rows.append(["\\textbf{certified CEL} (own spend)", f(c["prec"]), f(c["calls"], 2), f(c["acc"]),
                        f(c["auroc"]), f(c["auarc"]), "--", "--"])
    cm = r.get("certified_matched")
    if cm:
        cert_rows.append([f"\\multicolumn{{6}}{{l}}{{\\emph{{{ds} / {mk}}} ({cm['n_cal']} calibration cases)}}"])
        for alpha, routes in cm["by_alpha"].items():
            for key, v in routes.items():
                cert_rows.append([alpha, CERT.get(key, key), f(v["calls"], 1), f(v["fdr"]), f(v["power"]), f(v["size"], 1)])
        sp = cm["sprt"]
        cert_rows.append([f"\\multicolumn{{6}}{{l}}{{CEL SPRT certificate at $(\\varepsilon_0, p_0^*) = ({sp['eps0']}, "
                          f"{sp['p0_star']})$: {f(sp['certified_share'])} of cases certified, accuracy "
                          f"{f(sp['certified_accuracy'])} (exact for stability, not correctness)}}"])
latex_table(["Method", "prec@10\\%", "calls", "acc.", "AUROC", "AUARC", "PRR", "$\\Delta$ vs CEL"], rows,
            "Baseline ladder and ablations. prec@10\\% averaged over 10 batches; baselines "
            "matched to CEL's realised calls per case. Each method is scored on its own answers "
            "(acc.\\ = their accuracy); AUROC, AUARC and PRR pooled over the evaluated items.", "tab:main", out / "main_ladder.tex")

adj = holm({k: v["p_boot"] for k, v in conf.items()})
crow = [[k, f"{v['mean_diff']:+.3f}", f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]",
         f(v["p_boot"]), f(adj[k])] for k, v in conf.items()]
latex_table(["Cell", "$\\Delta$ prec@10\\%", "95\\% CI", "$p$", "$p_{\\mathrm{Holm}}$"], crow,
            "Confirmatory comparison: CEL minus matched-budget self-consistency, top 10\\% of the evaluated population; 95\\% interval and two-sided $p$ from a paired bootstrap over items (10,000 resamples).",
            "tab:confirmatory", out / "confirmatory.tex")
for k, v in conf.items():
    print(f"{k:28s} Δ={v['mean_diff']:+.3f} [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] p_holm={adj[k]:.3f}")
if cert_rows:
    latex_table(["$\\alpha$", "Method (conformal BH on its score)", "calls", "FDR", "power", "set size"], cert_rows,
                "Certified mode at the matched budget (R2b). Conformal $p$-values from the calibration split "
                "replayed under each method's own policy, Benjamini--Hochberg per batch of 50; realised FDR, "
                "power and delivered-set size averaged over the 10 batches. Exact: FDR $\\le \\alpha$ on "
                "average over calibration draws.", "tab:certified", out / "certified.tex", "llrrrr")
if el_rows:
    latex_table(["Method", "prec@10\\%", "calls", "acc.", "AUROC", "AUARC", "power ($\\alpha{=}0.1$)", "$\\Delta$ vs CEL-$k$"],
                el_rows,
                "Both modes at the elbow $k^*$ of CEL-$k$'s dev curve (Kneedle on dev prec@10\\% over budgets; "
                "chosen on dev only). CEL-$k$ and every baseline at $k^*$ calls per case; certified CEL at "
                "its own adaptive spend for reference, since the SPRT sets its spend. Power: share of correct "
                "answers delivered by conformal BH at FDR $\\alpha = 0.1$.", "tab:elbow", out / "elbow.tex",
                "lrrrrrrr")
