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


def budget_figure(curve: dict, title: str, path: Path, deploy_sc_draws: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for prefix, label, style, color in (("cel:", "CEL, adaptive", "-o", "C0"),
                                        ("celk:", "CEL-k, budget-allocated", "-^", "C2"),
                                        ("sc:", "Self-consistency", "--s", "C1")):
        pts = sorted((v["calls"], v["prec"], v["lo"], v["hi"]) for k, v in curve.items()
                     if k.startswith(prefix))
        if pts:
            x, y, lo, hi = zip(*pts)
            ax.plot(x, y, style, color=color, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    if deploy_sc_draws:
        mark_deployment_baseline(ax, curve, deploy_sc_draws)
    ax.axhline(curve["verbalized"]["prec"], ls=":", color="gray", label="Verbalized (1 call)")
    ax.set_xlabel("Mean calls per case (matched batch budget)")
    ax.set_ylabel("Precision@10")
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


def write_report(res: dict, out_dir: str | Path, tag: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    budget_figure(res["budget_curve"], f"Budget curve: {tag}", out / f"{tag}_budget.png",
                  res.get("deploy_sc_draws"))
    strata_figure(res["monotonicity"]["strata"], f"Monotonicity: {tag}", out / f"{tag}_strata.png")

    L = [f"# Results: {tag}\n"]
    L.append("## Baseline ladder and ablations\n")
    L.append("| Method | prec@10 | calls/case | AUROC | PRR | Δ vs CEL [95% CI] | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|---|")
    for n, m in res["methods"].items():
        pc = res["paired_vs_cel"].get(n)
        d = "—" if pc is None else f"{-pc['mean_diff']:+.3f} [{-pc['ci_hi']:+.3f}, {-pc['ci_lo']:+.3f}]"
        p = "—" if pc is None else _f(pc["wilcoxon_p"])
        L.append(f"| {n} | {_f(m['mean_prec'])} | {_f(m['mean_calls'], 1)} | {_f(m['auroc'])} | "
                 f"{_f(m['prr'])} | {d} | {p} |")
    c = res["confirmatory"]
    L.append(f"\nConfirmatory (CEL − self-consistency): {c['mean_diff']:+.3f} "
             f"[{c['ci_lo']:+.3f}, {c['ci_hi']:+.3f}], wins {c['wins']}/{c['n']}, p={_f(c['wilcoxon_p'])}\n")

    L.append("## Budget curve\n\n| Config | calls | prec@10 | AUROC |\n|---|---|---|---|")
    for n, v in sorted(res["budget_curve"].items(), key=lambda kv: kv[1]["calls"]):
        L.append(f"| {n} | {_f(v['calls'], 1)} | {_f(v['prec'])} | {_f(v['auroc'])} |")

    L.append("\n## Evidence-equivalence arms (Prop. 3)\n\n| Arm | calls | prec@10 |\n|---|---|---|")
    for n, a in res["arms"].items():
        L.append(f"| {n} | {_f(a['mean_calls'], 1)} | {_f(a['mean_prec'])} |")
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
                 "| k | CEL-k allocation | E[calls] dev / eval | CEL-k prec@10 | SC-k prec@10 | SC-k+verbal | Δ vs SC [95% CI] | wins | Δ vs SC+verbal [95% CI] |\n"
                 "|---|---|---|---|---|---|---|---|---|")
        for kk, b in bm["by_k"].items():
            p = b["paired"]
            L.append(f"| {kk} | {b['allocation']} | {b['dev_calls']:.1f} / {b['cel_k']['calls']:.1f} | "
                     f"{b['cel_k']['prec']:.3f} | {b['self_consistency']['prec']:.3f} | "
                     f"{b['self_consistency_verbal']['prec']:.3f} | {p['mean_diff']:+.3f} "
                     f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] | {p['wins']}/{p['n']} | "
                     f"{b['paired_vs_verbal']['mean_diff']:+.3f} [{b['paired_vs_verbal']['ci_lo']:+.3f}, "
                     f"{b['paired_vs_verbal']['ci_hi']:+.3f}] |")
    elif bm:
        L.append(f"\n## Budget-allocated CEL-k\n\n{bm.get('note', '')}")

    L.append("\n## Signal effects on confidence estimation\n\n"
             "Leave-one-out = CEL minus CEL without the signal (Δprec@10 paired over batches). "
             "Shapley = fair share of the prec@10 gain over all signal coalitions (Borda, screen fixed).\n\n"
             "| Signal | LOO Δprec@10 [95% CI] | LOO ΔAUROC | LOO ΔPRR | calls saved | "
             "φ prec@10 [95% CI] | φ per call |\n|---|---|---|---|---|---|---|")
    for j, v in sh["phi"].items():
        o, (lo, hi) = loo[j], sh["phi_ci"][j]
        L.append(f"| {j} | {o['d_prec']:+.3f} [{o['ci'][0]:+.3f}, {o['ci'][1]:+.3f}] | {o['d_auroc']:+.3f} | "
                 f"{o['d_prr']:+.3f} | {o['calls_saved']:.2f} | {v:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
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
        L.append(f"\nJudge swap: SE Spearman {_f(js['se_spearman'])}, top-10 Jaccard {_f(js['top10_jaccard'])}")
    p = out / f"{tag}.md"
    p.write_text("\n".join(L) + "\n")
    return p
