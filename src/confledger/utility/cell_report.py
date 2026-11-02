"""Render a protocol result dict as a markdown report plus figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _f(x, d=3):
    return "n/a" if x is None or x != x else f"{x:.{d}f}"


def mark_deployment_baseline(ax, curve: dict, draws: int, key: str = "prec") -> None:
    """Ring and label the self-consistency point used as the deployment baseline."""
    p = curve.get(f"sc:{draws}")
    if p:
        ax.plot([p["calls"]], [p[key]], "o", ms=13, mfc="none", mec="C1", mew=1.6, clip_on=False,
                label=f"deployment baseline (self-consistency, {draws} calls)")


def budget_figure(curve: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, label, style, color in (("celk:", "CEL-k, budget-allocated", "-^", "C2"),
                                        ("sc:", "Self-consistency", "--s", "C1")):
        pts = sorted((v["calls"], v["prec"], v["lo"], v["hi"]) for k, v in curve.items()
                     if k.startswith(prefix))
        if pts:
            x, y, lo, hi = zip(*pts)
            ax.plot(x, y, style, color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    c = curve.get("certified")
    if c:                                           # one policy, spend set by the data: a point
        ax.errorbar([c["calls"]], [c["prec"]], yerr=[[c["prec"] - c["lo"]], [c["hi"] - c["prec"]]],
                    fmt="*", ms=12, color="C0", label="Certified CEL, own spend")
    ax.axhline(curve["verbalized"]["prec"], ls=":", color="gray", label="Verbalized (1 call)")
    ax.set_xlabel("Mean calls per case (matched batch budget)")
    ax.set_ylabel("Precision at the top 10%")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def strata_figure(strata: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = [s["stratum"] for s in strata]
    y = [s["acc"] for s in strata]
    err = [[a - s["lo"] for a, s in zip(y, strata)], [s["hi"] - a for a, s in zip(y, strata)]]
    ax.errorbar(x, y, yerr=err, fmt="o-", capsize=3)
    ax.set_xlabel("Rank stratum (1 = most confident)")
    ax.set_ylabel("Accuracy (Jeffreys 95%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rank_rows(by_k: dict, names: dict) -> list[str]:
    """AUROC / AUARC per budget k, pooled over eval cases, for each score in `names`."""
    L = ["| k | " + " | ".join(f"{lab} AUROC / AUARC" for lab in names.values()) + " |",
         "|---|" + "---|" * len(names)]
    for kk, b in by_k.items():
        r = b.get("ranking")
        if r:
            L.append(f"| {kk} | " + " | ".join(
                f"{_f(r[n]['auroc'])} / {_f(r[n]['auarc'])}" if n in r else "—" for n in names) + " |")
    return L


def write_report(res: dict, out_dir: str | Path, tag: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    budget_figure(res["budget_curve"], f"Budget curve: {tag}", out / f"{tag}_budget.png")
    strata_figure(res["monotonicity"]["strata"], f"Monotonicity: {tag}", out / f"{tag}_strata.png")

    L = [f"# Results: {tag}\n"]
    L.append("## Baseline ladder and ablations\n")
    L.append("| Method | prec@10% | calls/case | accuracy | AUROC | AUARC | PRR | Δ vs CEL [95% CI] | p (paired bootstrap) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for n, m in res["methods"].items():
        pc = res["paired_vs_cel"].get(n)
        d = "—" if pc is None else f"{-pc['mean_diff']:+.3f} [{-pc['ci_hi']:+.3f}, {-pc['ci_lo']:+.3f}]"
        p = "—" if pc is None else _f(pc["p_boot"])
        L.append(f"| {n} | {_f(m['mean_prec'])} | {_f(m['mean_calls'], 1)} | {_f(m.get('base_acc'))} | {_f(m['auroc'])} | "
                 f"{_f(m.get('auarc'))} | {_f(m['prr'])} | {d} | {p} |")
    c = res["confirmatory"]
    L.append(f"\nConfirmatory (CEL − self-consistency): {c['mean_diff']:+.3f} "
             f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}], paired bootstrap over items p={_f(c['p_boot'])}\n")

    bl = res.get("baselines")
    if bl:
        L.append(f"## Fitted baselines (frozen on {bl['n_fit']} D_fit cases)\n")
        for src, f in bl["cisc"].items():
            L.append(f"- CISC ({src}): " + (f["note"] if "note" in f else
                     f"T = {f['T']:.3g}; D_fit vote accuracy {f['fit_acc']:.3f} vs SC {f['sc_acc']:.3f}; "
                     f"eval WQD {_f(bl['wqd'].get(src))}"))
        pf = bl["pedapati"]
        L.append("- Black-box calibrator (Pedapati et al.): " + (pf["note"] if "note" in pf else
                 "coef (intercept, SD sets, SD ROUGE-L[, PP sets, PP ROUGE-L]) = "
                 + ", ".join(f"{w:+.2f}" for w in pf["coef"])))
        L.append("")
    cal = res.get("calibration")
    if cal:
        L.append("## Calibration of probability-valued scores (ladder, matched budget)\n\n"
                 "Each against its own method's correctness; ECE with 10 equal-width bins.\n\n"
                 "| Method | ECE | Brier | mean conf | accuracy |\n|---|---|---|---|---|")
        for n, c in cal.items():
            L.append(f"| {n} | {c['ece']:.3f} | {c['brier']:.3f} | {c['mean_conf']:.3f} | {c['acc']:.3f} |")
        L.append("")
    L.append("## Budget curve\n\n| Config | calls | prec@10% | AUROC | AUARC |\n|---|---|---|---|---|")
    for n, v in sorted(res["budget_curve"].items(), key=lambda kv: kv[1]["calls"]):
        L.append(f"| {n} | {_f(v['calls'], 1)} | {_f(v['prec'])} | {_f(v['auroc'])} | {_f(v.get('auarc'))} |")

    L.append("\n## Evidence-equivalence arms (Prop. 3)\n\n| Arm | calls | prec@10% | AUROC | AUARC |\n|---|---|---|---|---|")
    for n, a in res["arms"].items():
        L.append(f"| {n} | {_f(a['mean_calls'], 1)} | {_f(a['mean_prec'])} | {_f(a.get('auroc'))} | {_f(a.get('auarc'))} |")
    for n, t in res["arms_tests"].items():
        L.append(f"\n{n}: {t['mean_diff']:+.3f} [{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]")
    kh = res["kappa_hat"]
    L.append("\nExchange rates, derived vs empirical: " + ", ".join(
        f"{g}: κ={res['kappa_derived'][g]:.2f} vs κ̂={kh['kappa_hat'][g]:.2f} "
        f"[{kh['ci'][g][0]:.2f}, {kh['ci'][g][1]:.2f}]" for g in kh["kappa_hat"]))

    mono = res["monotonicity"]
    L.append("\n## Monotonicity\n")
    L.append("Strata accuracy: " + ", ".join(_f(s["acc"], 2) for s in mono["strata"]))
    g = mono["isotonic_gap"]
    L.append(f"\nIsotonic gap: {g['gap']:.4f} [{g['ci_lo']:.4f}, {g['ci_hi']:.4f}]")
    L.append(f"\nTop strata: {json.dumps(mono['top_strata'], default=float)}")

    sh, loo = res["shapley"], res["leave_one_out"]
    bm = res.get("budget_matched")
    if bm and bm.get("by_k"):
        L.append("\n## Budget-allocated CEL-k vs self-consistency with k draws\n\n"
                 f"Allocation chosen on {bm['n_dev']} dev items by {bm['metric']} among "
                 f"{bm['n_candidates']} candidates; CEL-k on expected calls per case, SC on exactly k.\n\n"
                 "| k | CEL-k allocation | E[calls] dev / eval | CEL-k prec@10% | SC-k prec@10% | SC-k+verbal | Δ vs SC [95% CI] | p | Δ vs SC+verbal [95% CI] |\n"
                 "|---|---|---|---|---|---|---|---|---|")
        for kk, b in bm["by_k"].items():
            p = b["paired"]
            L.append(f"| {kk} | {b['allocation']} | {b['dev_calls']:.1f} / {b['cel_k']['calls']:.1f} | "
                     f"{b['cel_k']['prec']:.3f} | {b['self_consistency']['prec']:.3f} | "
                     f"{b['self_consistency_verbal']['prec']:.3f} | {p['mean_diff']:+.3f} "
                     f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | {_f(p['p_boot'])} | "
                     f"{b['paired_vs_verbal']['mean_diff']:+.3f} [{b['paired_vs_verbal']['ci_lo']:+.3f}, "
                     f"{b['paired_vs_verbal']['ci_hi']:+.3f}] |")
        L.append("\n### R1: every method at budget k (ranking mode)\n\n"
                 "CEL-k ranked by μ−λσ fitted on D_fit (Borda as a variant); baselines at exactly k calls, "
                 "CISC and the black-box calibrator refit at each k. Accuracy of each method's own answers; AUROC/AUARC "
                 "pooled over eval; Δ = method − CEL-k, paired over batches.\n\n"
                 "| k | method | prec@10% [95% CI] | calls | accuracy | AUROC | AUARC | Δ vs CEL-k [95% CI] |\n"
                 "|---|---|---|---|---|---|---|---|")
        for kk, b in bm["by_k"].items():
            ms = {"cel_k": b["cel_k"], "cel_k_borda": b.get("cel_k_borda")} | b.get("baselines", {})
            for n, m in ms.items():
                if not m:
                    continue
                pv = b.get("paired_vs", {}).get(n)
                d = "—" if pv is None else f"{-pv['mean_diff']:+.3f} [{-pv['ci_hi']:+.3f}, {-pv['ci_lo']:+.3f}]"
                lab = f"**CEL-k** (λ = {b.get('lambda', float('nan')):g})" if n == "cel_k" else n
                L.append(f"| {kk if n == 'cel_k' else ''} | {lab} | {m['prec']:.3f} [{_f(m.get('lo'))}, {_f(m.get('hi'))}] | "
                         f"{m['calls']:.2f} | {_f(m.get('acc'))} | {_f(m['auroc'])} | {_f(m.get('auarc'))} | {d} |")
        cal_k = {kk: b["calibration"] for kk, b in bm["by_k"].items() if b.get("calibration")}
        if cal_k:
            L.append("\nCalibration at each k (ECE / Brier, each on its own answers): ")
            for kk, c in cal_k.items():
                L.append(f"- k={kk}: " + "; ".join(f"{n} {v['ece']:.3f} / {v['brier']:.3f}" for n, v in c.items()))
    elif bm:
        L.append(f"\n## Budget-allocated CEL-k\n\n{bm.get('note', '')}")

    ps = res.get("pessimism")
    if ps and ps.get("by_k"):
        L.append(f"\n## Score model on D_fit ({ps['n_fit']} cases): mean vs pessimistic ranking\n\n"
                 "Same allocation and calls as CEL-k. λ by CV on D_fit (0 = rank by μ). Model-based.\n\n"
                 "| k | calls | λ | top-k changed | μ | μ−λσ | Borda | SC-k | Δ pess−mean [95% CI] | Δ pess−SC [95% CI] |\n"
                 "|---|---|---|---|---|---|---|---|---|---|")
        for kk, b in ps["by_k"].items():
            if "note" in b:
                L.append(f"| {kk} | {b['note']} |||||||||")
                continue
            pm, psc, pr = b["paired_pess_vs_mean"], b["paired_pess_vs_sc"], b["prec"]
            L.append(f"| {kk} | {b['calls']:.1f} | {b['lambda']:g} | {b['topk_changed_share']:.2f} | "
                     f"{pr['mean']:.3f} | {pr['pessimistic']:.3f} | {pr['borda']:.3f} | {pr['self_consistency']:.3f} | "
                     f"{pm['mean_diff']:+.3f} [{pm['ci_lo']:+.3f}, {pm['ci_hi']:+.3f}] | "
                     f"{psc['mean_diff']:+.3f} [{psc['ci_lo']:+.3f}, {psc['ci_hi']:+.3f}] |")
    if ps and ps.get("by_k"):
        L += ["\nWhole ranking, pooled over eval:\n"] + _rank_rows(
            ps["by_k"], {"mean": "μ", "pessimistic": "μ−λσ", "borda": "Borda", "self_consistency": "SC-k"})
    if ps and any("calibration" in b for b in ps["by_k"].values()):
        L.append("\n## Calibration on eval (μ as P(correct))\n\n"
                 "Pooled eval cases, each score against its own method's correctness. ECE: 10 equal-width "
                 "bins. μ−λσ ranks and is not a probability, so it is not listed. Model-based.\n\n"
                 "| k | score | ECE | Brier | mean conf | accuracy | AUROC | AUARC |\n|---|---|---|---|---|---|---|---|")
        for kk, b in ps["by_k"].items():
            for n, c in b.get("calibration", {}).items():
                L.append(f"| {kk} | {n} | {c['ece']:.3f} | {c['brier']:.3f} | {c['mean_conf']:.3f} | "
                         f"{c['acc']:.3f} | {_f(c['auroc'])} | {_f(c['auarc'])} |")
        kk, b = next((kk, b) for kk, b in ps["by_k"].items() if "calibration" in b)
        L.append(f"\nReliability of μ at k = {kk} (equal-count bins, accuracy [95% Jeffreys]): " + ", ".join(
            f"{r['conf']:.2f}→{r['acc']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" for r in b["calibration"]["mu"]["reliability"]))
    el = res.get("elbow")
    if el:
        L.append(f"\n## Metrics table at the elbow k* = {el['k_star']} (CEL-k's dev curve)\n\n"
                 "k* is the Kneedle elbow of CEL-k's dev prec@10% over budgets (dev only). CEL-k and every "
                 "baseline at k* calls; certified CEL at its own spend for reference (its spend is set by the "
                 "SPRT, not chosen).\n\n"
                 "| method | prec@10% [95% CI] | calls | accuracy | AUROC | AUARC | Δ vs CEL-k [95% CI] |\n"
                 "|---|---|---|---|---|---|---|")
        rows = el["methods"] | {"certified CEL (own spend)": el["certified_cel"]}
        for n, m in rows.items():
            pv = el["paired_vs"].get(n)
            d = "—" if pv is None else f"{-pv['mean_diff']:+.3f} [{-pv['ci_hi']:+.3f}, {-pv['ci_lo']:+.3f}]"
            L.append(f"| {n} | {m['prec']:.3f} [{_f(m.get('lo'))}, {_f(m.get('hi'))}] | {m['calls']:.2f} | "
                     f"{_f(m.get('acc'))} | {_f(m['auroc'])} | {_f(m.get('auarc'))} | {d} |")
        if "conformal" in el:
            L.append("\nConformal BH at k* (FDR / power / set size):")
            for a, routes in el["conformal"]["by_alpha"].items():
                L.append(f"- α={a}: " + "; ".join(f"{n} {r['fdr']:.3f} / {r['power']:.3f} / {r['size']:.1f}"
                                                  for n, r in routes.items()))
        sp = el["certified_cel"].get("sprt")
        if sp:
            L.append(f"\nCertified CEL's SPRT certificate: {sp['certified_share']:.3f} of cases, accuracy "
                     f"{_f(sp['certified_accuracy'])}.")
    cm = res.get("certified_matched")
    if cm:
        sp = cm["sprt"]
        L.append(f"\n## R2: certified mode at the matched budget ({cm['n_cal']} calibration cases)\n\n"
                 "Conformal p-values from D_cal replayed under each method's own policy, BH per batch; "
                 "mean over batches. Exact (FDR ≤ α on average over D_cal draws). CEL scored by its fitted "
                 "μ−λσ; each baseline by its primary score.\n\n"
                 "| α | method | calls | FDR | power | set size |\n|---|---|---|---|---|---|")
        for a, routes in cm["by_alpha"].items():
            for n, r in routes.items():
                L.append(f"| {a} | {n} | {r['calls']:.1f} | {r['fdr']:.3f} | {r['power']:.3f} | {r['size']:.1f} |")
        L.append(f"\nCEL SPRT certificate at (ε₀ = {sp['eps0']}, p₀* = {sp['p0_star']}): "
                 f"{sp['certified_share']:.3f} of eval cases certified ({sp['n_certified']}), accuracy of the "
                 f"certified {_f(sp['certified_accuracy'])}. Exact for stability, not for correctness.")
    ce = res.get("certified")
    if ce and ce.get("by_k"):
        L.append(f"\n## Certified mode: FDR control ({ce['n_fit']} fit, {ce['n_cal']} calibration cases)\n\n"
                 "Mean over batches of the delivered set's FDP (realised FDR), power and size. "
                 "Conformal routes are exact; e-BH is model-based. One D_cal serves every batch, so this "
                 "estimates FDR given that D_cal; the guarantee is on average over D_cal draws.\n\n"
                 "| k | α | route | calls | FDR | power | size |\n|---|---|---|---|---|---|---|")
        for kk, b in ce["by_k"].items():
            for a, rows in b.get("by_alpha", {}).items():
                for n, r in rows.items():
                    L.append(f"| {kk} | {a} | {n} | {r['calls']:.1f} | {r['fdr']:.3f} | {r['power']:.3f} | {r['size']:.1f} |")
    elif ce:
        L.append(f"\n## Certified mode\n\n{ce.get('note', '')}")
    fr = res.get("frontier")
    if fr and fr.get("by_k"):
        L.append(f"\n## Frontier allocation (LUCB on the top-k cut; model fitted on {fr['n_fit']} D_fit cases)\n\n"
                 "Mean calls per case ≤ k per batch; calls per case are uneven by design (sd, min–max). "
                 "Model-based, no guarantee.\n\n"
                 "| k | draws | calls (sd, min–max) | λ | top-k changed | μ | μ−λσ | CEL-k Borda | SC-k | "
                 "Δ pess−mean [95% CI] | Δ pess−SC [95% CI] |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for kk, b in fr["by_k"].items():
            pr, pm, psc = b["prec"], b["paired_pess_vs_mean"], b["paired_pess_vs_sc"]
            L.append(f"| {kk} | {'para' if b['para'] else 'orig'} | {b['calls']:.2f} ({b['calls_sd']:.1f}, "
                     f"{b['calls_min']}–{b['calls_max']}) | {b['lambda']:g} | {b['topk_changed_share']:.2f} | "
                     f"{pr['mean']:.3f} | {pr['pessimistic']:.3f} | {_f(pr.get('celk_borda'))} | "
                     f"{pr['self_consistency']:.3f} | {pm['mean_diff']:+.3f} [{pm['ci_lo']:+.3f}, {pm['ci_hi']:+.3f}] | "
                     f"{psc['mean_diff']:+.3f} [{psc['ci_lo']:+.3f}, {psc['ci_hi']:+.3f}] |")

    if fr and fr.get("by_k"):
        L += ["\nWhole ranking, pooled over eval:\n"] + _rank_rows(
            fr["by_k"], {"mean": "μ", "pessimistic": "μ−λσ", "self_consistency": "SC-k"})
        L.append("\nCalibration of μ on the frontier's final states: " + "; ".join(
            f"k={kk} ECE {b['calibration']['ece']:.3f}, Brier {b['calibration']['brier']:.3f}, "
            f"AUARC {b['calibration']['auarc']:.3f}" for kk, b in fr["by_k"].items() if "calibration" in b))

    ks = res.get("knapsack")
    if ks and ks.get("by_k"):
        imp = ks["importance"]
        L.append(f"\n## Signal values and the knapsack allocation (Prop. K; {ks['n_fit']} D_fit cases)\n\n"
                 "SHAP importance I = mean |β·(x − mean x)| with all signals bought [bootstrap 90%]; "
                 f"pruned free signals: {', '.join(ks['pruned']) or 'none'}. Value order: "
                 f"{' > '.join(ks['value_order'])}. Model-based.\n\n| Signal | I | 90% CI |\n|---|---|---|")
        for f in imp["rank"]:
            L.append(f"| {f} | {imp['I'][f]:.3f} | [{imp['ci'][f][0]:.3f}, {imp['ci'][f][1]:.3f}] |")
        L.append("\n| k | knapsack allocation | E[calls] dev / eval | knapsack p@10 | enumerated (Borda) | SC-k | "
                 "Δ vs enumerated [95% CI] | Δ vs SC [95% CI] |\n|---|---|---|---|---|---|---|---|")
        for kk, b in ks["by_k"].items():
            pe, ps = b.get("paired_vs_enumerated"), b["paired_vs_sc"]
            de = "—" if pe is None else f"{pe['mean_diff']:+.3f} [{pe['ci_lo']:+.3f}, {pe['ci_hi']:+.3f}]"
            L.append(f"| {kk} | {b['allocation']} | {b['expected_calls_dev']:.2f} / {b['calls']:.2f} | "
                     f"{b['prec']['knapsack']:.3f} | {_f(b['prec'].get('enumerated_borda'))} | "
                     f"{b['prec']['self_consistency']:.3f} | {de} | "
                     f"{ps['mean_diff']:+.3f} [{ps['ci_lo']:+.3f}, {ps['ci_hi']:+.3f}] |")
        L += ["\nWhole ranking, pooled over eval:\n"] + _rank_rows(
            ks["by_k"], {"knapsack": "knapsack μ−λσ", "self_consistency": "SC-k"})
    fp = res.get("fitted_prior")
    if fp:
        pr = fp["prior"]
        L.append(f"\n## Certified mode with a fitted prior\n\nBeta({pr['a']:.2f}, {pr['b']:.2f}) "
                 f"(mean {pr['prior_mean']:.2f}) from {pr['n_cases']} holdout cases × {pr['draws_per_case']} draws, "
                 f"no labels. Depth: point {pr['depth_point']}, bootstrap {pr['quantile']:.0%} quantile "
                 f"{pr['depth']} (median {pr['depth_median']:g}; {pr['share_zero']:.0%} of resamples certify "
                 "with no draws). GLR (δ = 0.1) needs "
                 f"{fp['glr']['unanimous_depth_delta_0.1']} unanimous draws. {fp['glr']['note']}\n\n"
                 "| prior | depth (draws used) | certified share | accuracy of certified | calls/case | mean ε | type |\n"
                 "|---|---|---|---|---|---|---|")
        for n, r in fp["replay"].items():
            if "note" in r:
                L.append(f"| {n} | {r['depth']} | {r['note']} ||||| ")
                continue
            L.append(f"| {n} | {r['depth']} ({r['draws_used']}) | {r['certified_share']:.3f} | "
                     f"{_f(r['certified_accuracy'])} | {r['mean_calls']:.2f} | {_f(r['eps'])} | {r['guarantee']} |")

    L.append("\n## Signal effects on confidence estimation\n\n"
             "Leave-one-out = CEL minus CEL without the signal (Δprec@10% paired over batches). "
             "Shapley = fair share of the prec@10% gain over all signal coalitions (Borda, screen fixed).\n\n"
             "| Signal | LOO Δprec@10% [95% CI] | LOO ΔAUROC | LOO ΔAUARC | LOO ΔPRR | calls saved | "
             "φ prec@10% [95% CI] | φ per call |\n|---|---|---|---|---|---|---|---|")
    for j, v in sh["phi"].items():
        o, (lo, hi) = loo[j], sh["phi_ci"][j]
        L.append(f"| {j} | {o['d_prec']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] | {o['d_auroc']:+.3f} | "
                 f"{o.get('d_auarc', float('nan')):+.3f} | {o['d_prr']:+.3f} | {o['calls_saved']:.2f} | {v:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{sh['phi_per_call'][j]:+.4f} |")
    L.append(f"\nRandom → screen: {sh['screen_gain']:+.4f}; signals total: {sh['total_gain']:+.4f}; "
             f"efficiency check: {sh['efficiency_check']:.1e}; flip increment V(L,flip)−V(L): "
             f"{sh['flip_increment']:+.4f}")

    h = res["heldout_audit"]
    L.append(f"\n## Label-free audits\n\nHeld-out probe survival: selected {_f(h['selected_pass'])} vs "
             f"controls {_f(h['control_pass'])} (Fisher p={_f(h['fisher_p'])}, "
             f"{h['executed_calls']} probe executions)")
    j = res["jaccard"]
    L.append(f"\nJaccard stability: {_f(j.get('mean'))} (chance floor {_f(j['chance_floor'])})")
    if "judge_sensitivity" in res:
        js = res["judge_sensitivity"]
        L.append(f"\nJudge swap: SE Spearman {_f(js['se_spearman'])}, top-10% Jaccard {_f(js['top10_jaccard'])}")
    p = out / f"{tag}.md"
    p.write_text("\n".join(L) + "\n")
    return p
