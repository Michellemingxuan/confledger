"""The Sec. 4 protocol for one (dataset, model) collection.

run_protocol() returns a JSON-serialisable dict; cell_report.py renders it. Every
number is a re-scoring of the cached collection -- no API calls happen here.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, replace
import json
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr

from data import load_split

from . import metrics as M
from .utility.cell_report import write_report
from .collection import collection_path, load_collection
from .utility.config import cells, record_run, results_dir
from .ranking import borda_scores, key_scores, ranks_within, select_top_k
from .methods import (ARMS, SIGNAL_ABLATIONS, CELConfig, CaseScore, allocation_space, budget_configs,
                      cel, cel_alloc, cel_as_allocation, cel_from_settings, evidence_depth, registry)
from .baselines import (bsdetector_like, cisc, fit_cisc, fit_pedapati, pedapati, self_consistency,
                        self_consistency_verbal, semantic_entropy, verbalized, xiong)
from .baselines.cisc import draw_conf
from .certified_selection import bh, conformal_pvalues, ebh, evalue, pass_rates, selection_outcome
from .frontier import fit_frontier, run_frontier
from .score_model import GRAY_BOX, choose_lambda, fit_score_model, top_precision
from .signal_shapley import exact_shapley
from .signal_value import full_cases, importance, item_values, knapsack_allocation, prune
from .stability_prior import fitted_prior, glr_depth, posterior_tail
from .sprt_posterior import (SPRT, DEFAULT_Q0, certification_depth, kappa, majority, tail_mass,
                             tail_mass_effective)
from .utility.records import CaseData

SIGNALS = ("L", "flip", "se", "chal", "verb")
PROB_SCORES = {                                    # ladder scores that claim to be P(Z = 1)
    "verbalized": lambda c: c.signals["verb"] / 100,
    "self_consistency": lambda c: c.signals["agree"],
    "xiong_avgconf": lambda c: c.signals["avg_conf"],
    "xiong_avgconf_para": lambda c: c.signals["avg_conf"],
    "cisc": lambda c: c.signals["cisc"],
    "cisc_p_answer": lambda c: c.signals["cisc"],
    "pedapati_lr": lambda c: c.signals["p_lr"],
}
PRIMARY_SCORE = ("verbalized", "self_consistency", "semantic_entropy", "bsdetector_like",   # key[0] is
                 "xiong_avgconf", "xiong_avgconf_para")                                       # their score
SC_CURVE_DRAWS = (1, 2, 3, 6, 9, 12)                                    # self-consistency budget curve
SIGNAL_COST = {"L": 6, "flip": 3, "se": 1, "chal": 1, "verb": 1}   # stage-2 calls per signal


def _scores(cases: list[CaseScore], ranking) -> np.ndarray:
    if ranking == "key":
        return key_scores(cases)
    _, sigs = ranking
    return borda_scores(cases, sigs)


def evaluate(coll: dict[str, CaseData], batches: list[list[str]], policy: Callable,
             k: int = 10, seed: Optional[int] = None, ranking="key",
             exact_random: bool = False, sel_seed: int = 0) -> dict:
    """prec@k of the evaluated population (decisions.md #29: one population, the top
    k = top_frac share of it; `batches` holds that one population), with its item-bootstrap
    replicates `rep` for paired comparisons; AUROC/PRR/AUARC, calls."""
    per_batch, K_b, sel_ids, ranks, all_cases, groups = [], [], [], {}, [], []
    rng = random.Random(sel_seed)
    for bi, batch in enumerate(batches):
        cases = [policy(coll[i], seed=seed) for i in batch]
        all_cases += cases
        z = [c.correct for c in cases]
        K = int(sum(z))
        K_b.append(K)
        if exact_random:
            per_batch.append(M.hypergeom_expected_precision(K, len(batch), k))
            sel_ids.append([])
            continue
        sc = _scores(cases, ranking)
        top = select_top_k(sc, k, rng)
        per_batch.append(M.precision_at_k([z[t] for t in top]))
        sel_ids.append([cases[t].item_id for t in top])
        cut = min(sc[t] for t in top)
        groups.append([c.item_id for c, v in zip(cases, sc) if v >= cut])
        for c, r in zip(cases, ranks_within(sc, random.Random(sel_seed + bi))):
            ranks[c.item_id] = int(r)
    z_all = [c.correct for c in all_cases]
    s_all = _scores(all_cases, ranking)
    if len(batches) == 1:                        # population mode: item bootstrap
        frac = k / len(batches[0])
        rep = M.mean_rep(z_all) if exact_random else M.prec_rep(s_all, z_all, frac)
    else:                                        # legacy batches: replicates over batches
        idx = np.random.default_rng(0).integers(0, len(batches), (M.N_BOOT, len(batches)))
        rep = np.concatenate([[np.mean(per_batch)], np.asarray(per_batch)[idx].mean(axis=1)])
    out = {"per_batch": per_batch, "_rep": rep, "mean_prec": float(rep[0]), "K_b": K_b,
           "base_acc": float(np.mean(z_all)),
           "mean_calls": float(np.mean([c.calls for c in all_cases])),
           "survival": float(np.mean([c.survived for c in all_cases])),
           "auroc": M.auroc(s_all, z_all), "prr": M.prr(s_all, z_all), "auarc": M.auarc(s_all, z_all),
           "selected": sel_ids}
    x = sum(int(round(p * k)) for p in per_batch) if not exact_random else None
    if x is not None:
        out["pooled_jeffreys"] = M.jeffreys_interval(x, k * len(batches))
        base = out["base_acc"]
        out["rel_error_reduction"] = ((1 - base) - (1 - out["mean_prec"])) / (1 - base) if base < 1 else float("nan")
    out["_cases"] = all_cases
    out["_groups"] = groups
    out["_ranks"] = ranks
    out["_scores"] = s_all
    return out


def _strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def theory_constants(sprt: SPRT = SPRT()) -> dict:
    return {
        "sprt": {"up": sprt.up, "down": sprt.down, "A": sprt.A, "B": sprt.B,
                 "expected_n": sprt.expected_n()},
        "screen_pass": {"p=0.7": 0.7 ** 3, "p=0.95": 0.95 ** 3},
        "certification_depth": {"eps0=0.05": certification_depth(0.05),
                                "eps0=0.02": certification_depth(0.02)},
        "tail_mass": {"9/9": tail_mass(9, 9), "12/12": tail_mass(12, 12), "8/9": tail_mass(8, 9)},
        "kappa": {g: kappa(q0, sprt=sprt) for g, q0 in DEFAULT_Q0.items()},
        "glr_depth": {f"delta={d}": {"unanimous": glr_depth(d), "one_disagreement": glr_depth(d, disagreements=1)}
                      for d in (0.1, 0.05)},
    }


def select_allocations(dev_coll: dict[str, CaseData], dev_batches: list[list[str]],
                       budgets: Sequence[int], verbal_per_draw: bool, max_extra: int = 6,
                       kappas: Optional[dict] = None, cert: Optional[dict] = None,
                       band: float = 1.0) -> dict:
    """Choose CEL-k's allocation per budget on the DEV slice (nothing tuned on eval): every
    replayable allocation is scored on dev -- AUARC of Borda over its signals (decisions.md
    #34: prec@k saturates at 1). The budget B is a target, not only a cap (decisions.md #26):
    per B, the best allocation whose EXPECTED calls per case on dev lie in (B - band, B] is
    kept; if none does, the best among those spending the most without exceeding B. Ties
    go to more calls. max_extra >= adaptive CEL's s2, so adaptive CEL is in the space
    (Sec. 3.9, nesting).

    With `kappas`, also the DERIVED allocation per budget (Sec. 3.9): among fresh-paraphrase
    allocations (Lemma 1), the one maximising k_eff (Prop. 3, model-based) under the same
    expected-call constraint. Label-free: it reads only dev call counts. With `cert`
    ({eps0, p0_star, depth}), also the certification budgets: the least expected calls at
    which a unanimous survivor can be certified, exactly (>= depth draws, Cor. 1) and
    model-based (eps_eff <= eps0, Prop. 3)."""
    max_draws = min(min(len(cd.same), len(cd.fresh)) for cd in dev_coll.values())
    space = allocation_space(max_extra=max_extra, verbal_per_draw=verbal_per_draw, max_draws=max_draws)
    rows = []
    for alloc in space:
        cases = [[cel_alloc(dev_coll[i], alloc) for i in b] for b in dev_batches]
        calls = float(np.mean([c.calls for cs in cases for c in cs]))
        sc = np.concatenate([borda_scores(cs, SIGNALS) for cs in cases])
        rows.append((alloc, calls, M.auarc(sc, [c.correct for cs in cases for c in cs])))
    chosen = {}
    for B in budgets:
        feasible = [r for r in rows if r[1] <= B + 1e-9]
        if feasible:
            target = [r for r in feasible if r[1] > B - band + 1e-9]
            if not target:                              # nothing near B: spend the most that fits
                top = max(r[1] for r in feasible)
                target = [r for r in feasible if r[1] >= top - 1e-9]
            alloc, calls, score = max(target, key=lambda r: (round(r[2], 9), r[1]))
            chosen[str(B)] = {"allocation": alloc, "dev_calls": calls, "dev_score": score}
    out = {"metric": "AUARC", "n_dev": sum(map(len, dev_batches)), "n_candidates": len(space), "chosen": chosen}
    if kappas is None:
        return out
    para = [(a, c, *evidence_depth(a, kappas)) for a, c, _ in rows if a.para]
    derived = {}
    for B in budgets:
        feasible = [r for r in para if r[1] <= B + 1e-9]
        if feasible:                                   # max k_eff; ties: fewer calls
            a, c, d, ke = max(feasible, key=lambda r: (round(r[3], 9), -r[1]))
            derived[str(B)] = {"allocation": a, "dev_calls": c, "draws": d, "k_eff": ke}
    out["derived"] = derived
    if cert:
        least = lambda ok: min((r[1] for r in para if ok(r)), default=None)
        out["certification_budget"] = {
            "exact": least(lambda r: r[2] >= cert["depth"]),
            "model": least(lambda r: tail_mass_effective(r[3], cert["p0_star"]) <= cert["eps0"])}
    return out


def delivered_certificates(ev: dict, kappas: dict, eps0: float, p0_star: float) -> dict:
    """Certificate carried by the DELIVERED top-k sets: share of selected cases with
    eps_i <= eps0 (Prop. 2, exact, draws only) and with eps_eff <= eps0 (Prop. 3,
    model-based: unanimous draws, no failed probe, k_eff = draws + passed kappas), and the
    mean over batches of sum_{i in T} eps_i (expected sub-standard cases per set, exact)."""
    by_id = {c.item_id: c for c in ev["_cases"]}
    exact, model, sums = [], [], []
    for ids in ev["selected"]:
        tot = 0.0
        for i in ids:
            c = by_id[i]
            m, n = c.signals.get("m", 0), c.signals.get("k", 0)
            eps = tail_mass(m, n, p0_star) if n else 1.0
            tot += eps
            exact.append(eps <= eps0)
            clean = c.survived and n > 0 and m == n and False not in c.probe_passes.values()
            ke = n + sum(kappas.get(g, 0.0) for g, v in c.probe_passes.items() if v)
            model.append(clean and tail_mass_effective(ke, p0_star) <= eps0)
        sums.append(tot)
    mean = lambda x: float(np.mean(x)) if x else float("nan")
    return {"certified_exact": mean(exact), "certified_model": mean(model), "eps_sum": mean(sums)}


def check_collection(coll: dict[str, CaseData], budget: Optional[int] = None,
                     deploy_sc_draws: int = 3, budget_ks: Sequence[int] = (),
                     base: CELConfig = CELConfig()) -> None:
    """Refuse a collection too small for the protocol; otherwise methods would replay
    fewer draws than their labels say (e.g. 'self-consistency at 12' on 3 draws)."""
    need_fresh = max(12, *(c.s1 + c.s2 for c in budget_configs(base).values()))  # richest CEL point, arm A
    need_reuse = base.s1 + base.s2                                                  # no-spread ablation
    need_same = max(SC_CURVE_DRAWS + (deploy_sc_draws,) + ((budget,) if budget else ()))
    need_same = max(need_same, *budget_ks) if budget_ks else need_same
    have = {"fresh": min(len(cd.fresh) for cd in coll.values()),
            "same": min(len(cd.same) for cd in coll.values()),
            "reuse": min(sum(len(r) for r in cd.reuse) for cd in coll.values())}
    short = [f"{have['fresh']} fresh-paraphrase draws (n_paraphrases), need {need_fresh}"] * (have["fresh"] < need_fresh) \
        + [f"{have['same']} same-prompt draws (n_same), need {need_same} (budget curve up to "
           f"{max(SC_CURVE_DRAWS)}{f', matched budget {budget}' if budget else ''})"] * (have["same"] < need_same) \
        + [f"{have['reuse']} reuse draws (n_reuse_paras x n_reuse_samples), need {need_reuse}"] * (have["reuse"] < need_reuse)
    if short:
        raise ValueError("collection too small for the evaluation protocol: " + "; ".join(short))


SCORING = {"ridge": 1.0, "prior": [0.5, 0.5], "lambdas": [0, 0.5, 1, 1.5, 2, 3], "cv_folds": 5,
           "alphas": [0.05, 0.1, 0.2], "q0_level": 0.95, "gray_box": False,
           "budget_band": 1.0,   # CEL-k spends within (k - band, k] calls per case on dev (#26)
           "extra_fracs": [0.05, 0.01]}   # further top shares reported on the budget curve (nothing is tuned on them)


def _gray(o: dict) -> tuple:
    """Features held out of every fit: the gray-box first-token probability unless enabled."""
    return () if o.get("gray_box") else GRAY_BOX


def fit_scorer(fit_cases: Sequence[CaseScore], k_frac: float, o: dict):
    """CEL's ranking score mu - lambda*sigma for one policy: the score model fitted on the
    policy's D_fit cases, lambda by CV (0 allowed, i.e. rank by mu). None if D_fit has one
    class. Model-based."""
    z = np.array([c.correct for c in fit_cases])
    if z.all() or not z.any():
        return None
    prior = tuple(o["prior"])
    lam, cv = choose_lambda(fit_cases, z, o["lambdas"], k_frac, o["cv_folds"], o["ridge"], prior, drop=_gray(o))
    return fit_score_model(fit_cases, z, o["ridge"], prior, _gray(o)), lam, cv


def scored(policy: Callable, model, lam: float) -> Callable:
    """`policy` re-keyed by mu - lambda*sigma; mu and sigma kept in the signals."""
    def pol(cd, seed=None):
        c = policy(cd, seed=seed)
        (mu,), (sd,) = model.mu_sigma([c])
        return replace(c, key=(float(mu - lam * sd),), signals={**c.signals, "mu": float(mu), "sigma": float(sd)})
    return pol


def celk_policy(alloc, fit_coll: dict[str, CaseData], k_frac: float, o: dict) -> tuple[Callable, dict]:
    """CEL-k under allocation `alloc`, ranked by mu - lambda*sigma fitted on D_fit (Borda if
    D_fit has one class)."""
    raw = lambda cd, seed=None: cel_alloc(cd, alloc, seed)
    f = fit_scorer([raw(cd) for cd in fit_coll.values()], k_frac, o)
    if f is None:
        return None, {"score": "borda (D_fit has one class)"}
    return scored(raw, f[0], f[1]), {"score": "mu - lambda*sigma", "lambda": f[1], "cv": f[2]}


def baseline_policies(fit: Sequence[CaseData], B: int, vpd: bool, gray: bool = False) -> tuple[dict, dict]:
    """Every baseline at exactly B calls per case (semantic entropy and BSDetector-style:
    B - 1 draws + their extra call), from its first feasible B. CISC's T and the
    Pedapati-style regression are refit on D_fit at this B."""
    pol: dict[str, Callable] = {
        "self_consistency": lambda cd, seed=None: self_consistency(cd, B, seed),
        "self_consistency_verbal": lambda cd, seed=None: self_consistency_verbal(cd, B, seed)}
    info: dict = {"cisc": {}}
    if B >= 2:
        pol["bsdetector_like"] = lambda cd, seed=None: bsdetector_like(cd, B - 1, seed=seed)
    if B >= 3:
        pol["semantic_entropy"] = lambda cd, seed=None: semantic_entropy(cd, B - 1, seed)
    if vpd:                                           # Avg-Conf needs a confidence on every draw
        pol["xiong_avgconf"] = lambda cd, seed=None: xiong(cd, B, "self_random", seed)
        if 2 <= B <= min(len(cd.fresh) for cd in fit):    # one paraphrase per call
            pol["xiong_avgconf_para"] = lambda cd, seed=None: xiong(cd, B, "prompting", seed)
    if B >= 2:                                        # B = 1: CISC is self-consistency
        for src in ["verbal"] * vpd + ["p_answer"] * gray:
            if not all(draw_conf(d, src) is not None for cd in fit for d in cd.same[:B] if d.answer):
                info["cisc"][src] = {"note": f"no per-draw {src} confidence in the collection"}
                continue
            f = fit_cisc(fit, B, src)
            info["cisc"][src] = f
            pol["cisc" if src == "verbal" else "cisc_p_answer"] = (
                lambda T, s: (lambda cd, seed=None: cisc(cd, T, B, s, seed)))(f["T"], src)
        pf = fit_pedapati(fit, B)
        info["pedapati"] = pf
        if "note" not in pf:
            pol["pedapati_lr"] = lambda cd, seed=None: pedapati(cd, pf, seed)
    return pol, info


def fit_baselines(coll: dict[str, CaseData], batches: list[list[str]], fit_coll: dict[str, CaseData],
                  budget: int, reg: dict, o: dict) -> dict:
    """Add the fitted baselines (CISC, Pedapati-style) at the matched budget to `reg` (in
    place), and the within-question discrimination of each per-draw confidence source on
    eval, which Taubenfeld et al. find predicts CISC's gain."""
    vpd = all(cd.meta.get("verbal_per_draw") for cd in coll.values())
    pols, info = baseline_policies(list(fit_coll.values()), budget, vpd, bool(o.get("gray_box")))
    for n in ("cisc", "cisc_p_answer", "pedapati_lr"):
        if n in pols:
            reg[n] = pols[n]
    out = {"n_fit": len(fit_coll), "cisc": info["cisc"], "pedapati": info.get("pedapati", {}), "wqd": {}}
    for src, f in info["cisc"].items():
        if "note" in f:
            continue
        groups = [([draw_conf(d, src) for d in coll[i].same[:budget] if d.answer],
                   [d.answer == coll[i].gold for d in coll[i].same[:budget] if d.answer])   # metric only
                  for b in batches for i in b]
        out["wqd"][src] = M.wqd(groups)
    return out


def _band(rep) -> dict:
    return M.rep_band(rep)


def _summ(e: dict) -> dict:
    return {"prec": e["mean_prec"], "calls": e["mean_calls"], "acc": e["base_acc"], "auroc": e["auroc"],
            "auarc": e["auarc"], "prr": e["prr"], "survival": e["survival"], **_band(e["_rep"])}


def _top_shares(runs: Sequence[dict], fracs: Sequence[float]) -> dict:
    """prec of the top share f with its bootstrap band, as prec_<f>, lo_<f>, hi_<f>; same
    rankings as the main cut, averaged over `runs` (the curve seeds)."""
    out = {}
    for f in fracs:
        rep = np.mean([M.prec_rep(r["_scores"], [c.correct for c in r["_cases"]], f) for r in runs], axis=0)
        band = _band(rep)
        out |= {f"prec_{f:g}": float(rep[0]), f"lo_{f:g}": band["lo"], f"hi_{f:g}": band["hi"]}
    return out


def _calib(cases: Sequence[CaseScore], f: Callable) -> dict:
    return M.calibration([f(c) for c in cases], [c.correct for c in cases])


def certified_matched_section(coll: dict[str, CaseData], batches: list[list[str]],
                              cal_coll: dict[str, CaseData], reg: dict, cel_scored: Callable,
                              k: int, o: dict, eps0: float, p0_star: float) -> dict:
    """Certified mode at the matched budget (R2b). For adaptive CEL (ranked by its fitted
    mu - lambda*sigma) and every baseline (its primary score, key[0]): conformal p-values
    from D_cal replayed under the SAME policy, BH per batch; FDP, power and size of the
    delivered set -- exact (marginal over D_cal). Valid because each policy's calls depend
    only on the case's own draws, so cal and test scores are exchangeable. Plus the SPRT
    certificate of adaptive CEL: exact for stability pbar, not for correctness."""
    names = {"cel": cel_scored} | {n: p for n, p in reg.items()
                                   if n in PRIMARY_SCORE or n in ("cisc", "cisc_p_answer", "pedapati_lr")}
    out, scores = conformal_routes(coll, batches, cal_coll, names, o)
    cases = [c for cs in scores["cel"][2] for c in cs]
    cert = [c for c in cases if c.survived and c.signals.get("k", 0) > 0 and c.signals["m"] == c.signals["k"]
            and tail_mass(c.signals["m"], c.signals["k"], p0_star) <= eps0]
    out["sprt"] = {"eps0": eps0, "p0_star": p0_star, "certified_share": len(cert) / len(cases),
                   "certified_accuracy": float(np.mean([c.correct for c in cert])) if cert else float("nan"),
                   "n_certified": len(cert), "guarantee": "exact for stability, not correctness"}
    out["calibration_cel_mu"] = _calib(cases, lambda c: c.signals["mu"])
    return out


def conformal_routes(coll: dict[str, CaseData], batches: list[list[str]], cal_coll: dict[str, CaseData],
                     names: dict, o: dict) -> tuple[dict, dict]:
    """Conformal BH for each policy in `names` on its primary score key[0]: p-values from
    D_cal replayed under the same policy, BH per batch; mean FDP, power, size over batches.
    Exact for any policy whose calls depend only on the case's own draws."""
    cal_ids = list(cal_coll)
    out = {"n_cal": len(cal_ids), "by_alpha": {}, "calls": {}}
    scores = {}
    for n, pol in names.items():
        cal = [pol(cal_coll[i]) for i in cal_ids]
        test = [[pol(coll[i]) for i in b] for b in batches]
        scores[n] = (np.array([c.key[0] for c in cal], float), np.array([c.correct for c in cal]), test)
        out["calls"][n] = float(np.mean([c.calls for cs in test for c in cs]))
    for alpha in o["alphas"]:
        rows = {}
        for n, (s_cal, z_cal, test) in scores.items():
            outc = [selection_outcome(bh(conformal_pvalues(np.array([c.key[0] for c in cs], float), s_cal, z_cal),
                                         alpha), np.array([c.correct for c in cs])) for cs in test]
            rows[n] = {"fdr": float(np.mean([r["fdp"] for r in outc])),
                       "power": float(np.mean([r["power"] for r in outc])),
                       "size": float(np.mean([r["size"] for r in outc])), "calls": out["calls"][n],
                       "guarantee": "exact"}
        out["by_alpha"][str(alpha)] = rows
    return out, scores


def elbow_budget(dev_curve: dict) -> int:
    """Elbow of CEL-k's DEV curve (Kneedle, Satopaa et al. 2011): with budgets and dev
    AUARC (running max, the best plan within each budget) scaled to [0, 1], the budget
    furthest above the chord from the smallest to the largest. Ties: the smaller budget.
    Chosen on dev only, so the eval table at the elbow is not tuned on eval."""
    ks = sorted(int(b) for b in dev_curve)
    y = np.maximum.accumulate(np.array([dev_curve[k] for k in ks], float))
    if len(ks) < 3 or y[-1] == y[0]:
        return ks[0]
    d = (y - y[0]) / (y[-1] - y[0]) - (np.array(ks, float) - ks[0]) / (ks[-1] - ks[0])
    return ks[int(np.argmax(d))]


def elbow_section(coll, batches, cal_coll, fit_only, sel, k, k_frac, o, vpd, main) -> dict:
    """The metrics table at the elbow budget k* of CEL-k's dev curve: CEL-k (mu - lambda*sigma)
    and every baseline at exactly k* calls -- prec@k, calls, accuracy, AUROC, AUARC, paired
    deltas -- and, with D_cal, conformal BH for each. Certified CEL cannot be placed at k*
    (its spend is set by the SPRT), so it is reported as a reference at its own spend."""
    curve = {int(B): ch["dev_score"] for B, ch in sel["chosen"].items()}
    ks = elbow_budget(curve)
    alloc = sel["chosen"][str(ks)]["allocation"]
    pol, how = celk_policy(alloc, fit_only, k_frac, o)
    raw = lambda cd, seed=None: cel_alloc(cd, alloc, seed)
    ck = evaluate(coll, batches, pol, k) if pol else evaluate(coll, batches, raw, k, ranking=("borda", SIGNALS))
    pols, _ = baseline_policies(list(fit_only.values()), ks, vpd, bool(o.get("gray_box")))
    evs = {n: evaluate(coll, batches, p, k) for n, p in pols.items()}
    ref = main.get("cel_scored", main["cel"])
    out = {"k_star": ks, "dev_curve": {str(b): v for b, v in sorted(curve.items())}, "allocation": alloc.label(),
           **how, "methods": {"cel_k": _summ(ck)} | {n: _summ(e) for n, e in evs.items()},
           "paired_vs": {n: M.paired_compare(ck["_rep"], e["_rep"]) for n, e in evs.items()},
           "certified_cel": _summ(ref)}
    if cal_coll and pol:
        out["conformal"], _ = conformal_routes(coll, batches, cal_coll, {"cel_k": pol} | pols, o)
    return out


def score_model_section(coll: dict[str, CaseData], batches: list[list[str]], chosen: dict,
                        budget_ks: Sequence[int], k: int, fit_coll: dict[str, CaseData],
                        cal_coll: Optional[dict[str, CaseData]] = None,
                        scoring: Optional[dict] = None) -> tuple[dict, dict]:
    """Per budget B, on CEL-k's dev-chosen allocation (same calls, so matched budget):

    pessimism  the score model fitted on D_fit (= the dev slice), ranked by mu (lambda = 0)
               and by mu - lambda*sigma (lambda by CV on D_fit), vs Borda CEL-k and
               self-consistency with B draws, on prec@k paired over batches. Model-based.
    certified  per FDR level alpha, delivered sets of (i) conformal p-values on D_cal + BH
               with the pessimistic score, the mean score and self-consistency agreement
               (draws only: the separation theorem's baseline) -- exact -- and (ii) e-BH on
               ledger e-values with D_fit pass rates -- model-based. Needs `cal_coll`.
    Both are frozen on D_fit / D_cal before any eval case is scored."""
    o = {**SCORING, **(scoring or {})}
    prior = tuple(o["prior"])
    fit_ids = list(fit_coll)
    pess, cert = {"n_fit": len(fit_ids), "by_k": {}}, {"n_fit": len(fit_ids), "by_k": {}}
    if cal_coll:
        cert["n_cal"] = len(cal_coll)
    for B, ch in chosen.items():
        if int(B) not in budget_ks:
            continue
        a = ch["allocation"]
        pol = lambda cd, a=a: cel_alloc(cd, a)
        fit_cases = [pol(fit_coll[i]) for i in fit_ids]
        z_fit = np.array([c.correct for c in fit_cases])
        if z_fit.all() or not z_fit.any():
            pess["by_k"][B] = cert["by_k"][B] = {"note": "D_fit has one class only; no score model"}
            continue
        lam, cv = choose_lambda(fit_cases, z_fit, o["lambdas"], k / len(batches[0]), o["cv_folds"],
                                o["ridge"], prior, drop=_gray(o))
        model = fit_score_model(fit_cases, z_fit, o["ridge"], prior, _gray(o))
        test = [[pol(coll[i]) for i in b] for b in batches]
        sc = [[self_consistency(coll[i], int(B)) for i in b] for b in batches]
        per = {"mean": [], "pessimistic": [], "borda": [], "self_consistency": []}
        changed, calls = [], []
        pool = {"mu": ([], []), "self_consistency": ([], []), "verbalized": ([], [])}
        rk = {"mean": [], "pessimistic": [], "self_consistency": []}
        z_pool, all_cases = [], []
        for cases, sccases, b in zip(test, sc, batches):
            z = np.array([c.correct for c in cases])
            mu, sd = model.mu_sigma(cases)
            pool["mu"][0].extend(mu), pool["mu"][1].extend(z)
            rk["mean"] += list(mu)
            rk["pessimistic"] += list(mu - lam * sd)
            z_pool += list(z)
            all_cases += cases
            pool["self_consistency"][0].extend(c.signals["agree"] for c in sccases)
            pool["self_consistency"][1].extend(c.correct for c in sccases)
            for i in b:
                if coll[i].verbal_conf is not None:
                    pool["verbalized"][0].append(coll[i].verbal_conf / 100)
                    pool["verbalized"][1].append(coll[i].verbal_answer == coll[i].gold)   # metric only
            fr = k / len(z)                       # one population: point + item-bootstrap replicates
            per["mean"] = M.prec_rep(mu, z, fr)
            per["pessimistic"] = M.prec_rep(mu - lam * sd, z, fr)
            per["borda"] = M.prec_rep(borda_scores(cases, SIGNALS), z, fr)
            per["self_consistency"] = M.prec_rep(np.array([c.signals["agree"] for c in sccases]),
                                                 np.array([c.correct for c in sccases]), fr)
            top = lambda s: set(np.lexsort((np.arange(len(s)), -s))[:k])
            changed.append(top(mu) != top(mu - lam * sd))
            calls += [c.calls for c in cases]
        pess["by_k"][B] = {
            "allocation": a.label(), "calls": float(np.mean(calls)), "lambda": lam, "cv": cv,
            "model": model.summary(), "topk_changed_share": float(np.mean(changed)),
            "prec": {n: float(v[0]) for n, v in per.items()},
            # pooled over eval cases, each against its own method's Z
            "ranking": {"mean": M.rank_metrics(rk["mean"], z_pool),
                        "pessimistic": M.rank_metrics(rk["pessimistic"], z_pool),
                        "borda": M.rank_metrics(borda_scores(all_cases, SIGNALS), z_pool),
                        "self_consistency": M.rank_metrics(*pool["self_consistency"])},
            "paired_pess_vs_mean": M.paired_compare(per["pessimistic"], per["mean"]),
            "paired_pess_vs_sc": M.paired_compare(per["pessimistic"], per["self_consistency"]),
            "paired_pess_vs_borda": M.paired_compare(per["pessimistic"], per["borda"]),
            # pooled eval cases; each score against its own method's Z. mu - lambda*sigma is a
            # ranking score, not a probability, so it is not calibrated here.
            "calibration": {n: M.calibration(p, zz) for n, (p, zz) in pool.items() if p}}
        if not cal_coll:
            continue
        cal_ids = list(cal_coll)
        cal_cases = [pol(cal_coll[i]) for i in cal_ids]
        z_cal = np.array([c.correct for c in cal_cases])
        mu_c, sd_c = model.mu_sigma(cal_cases)
        sc_cal = np.array([self_consistency(cal_coll[i], int(B)).signals["agree"] for i in cal_ids])
        rates = pass_rates(fit_cases, z_fit, o["q0_level"])
        by_alpha = {}
        for alpha in o["alphas"]:
            rows = {"conformal_pessimistic": [], "conformal_mean": [], "conformal_sc": [], "ebh": []}
            for cases, sccases in zip(test, sc):
                z = np.array([c.correct for c in cases])
                mu, sd = model.mu_sigma(cases)
                s_sc = np.array([c.signals["agree"] for c in sccases])
                rows["conformal_pessimistic"].append(selection_outcome(
                    bh(conformal_pvalues(mu - lam * sd, mu_c - lam * sd_c, z_cal), alpha), z))
                rows["conformal_mean"].append(selection_outcome(
                    bh(conformal_pvalues(mu, mu_c, z_cal), alpha), z))
                rows["conformal_sc"].append(selection_outcome(
                    bh(conformal_pvalues(s_sc, sc_cal, z_cal), alpha), z))
                rows["ebh"].append(selection_outcome(
                    ebh(np.array([evalue(c, rates) for c in cases]), alpha), z))
            by_alpha[str(alpha)] = {
                n: {"fdr": float(np.mean([r["fdp"] for r in v])),
                    "power": float(np.mean([r["power"] for r in v])),
                    "size": float(np.mean([r["size"] for r in v])),
                    "calls": float(B) if n == "conformal_sc" else float(np.mean(calls)),
                    "guarantee": "model-based" if n == "ebh" else "exact"}
                for n, v in rows.items()}
        cert["by_k"][B] = {"allocation": a.label(), "lambda": lam, "pass_rates": rates,
                           "label_calls_per_case": float(np.mean([c.calls for c in fit_cases + cal_cases])),
                           "by_alpha": by_alpha}
    return pess, cert


FRONTIER = {"k1": 1, "z": 1.0, "reps": 8}


def frontier_section(coll: dict[str, CaseData], batches: list[list[str]], chosen: dict,
                     budget_ks: Sequence[int], k: int, fit_coll: dict[str, CaseData],
                     dev_batches: list[list[str]], scoring: Optional[dict] = None,
                     frontier: Optional[dict] = None) -> dict:
    """Per budget B: the frontier allocation (frontier.py) with its draw type (paraphrase or
    original wording) chosen on the dev batches by pessimistic prec@k, then run on eval and
    compared, paired over batches, with self-consistency (B draws) and Borda CEL-k on its
    dev-chosen fixed allocation. Reports how uneven the calls per case are and how often
    pessimism changes the top-k. Model-based: the frontier carries no guarantee."""
    o, f = {**SCORING, **(scoring or {})}, {**FRONTIER, **(frontier or {})}
    fit = list(fit_coll.values())
    fits = {para: fit_frontier(fit, para, k / len(batches[0]), f["reps"], o["ridge"], tuple(o["prior"]),
                               o["lambdas"], o["cv_folds"], drop=_gray(o)) for para in (False, True)}
    out = {"n_fit": len(fit), "settings": f, "by_k": {}}
    for B in budget_ks:
        feasible = [p for p in (False, True) if B >= int(p) + f["k1"]]
        if not feasible:
            continue
        k_dev = max(1, math.ceil(k / len(batches[0]) * len(dev_batches[0]) - 1e-9))   # same top share on dev
        dev = {p: run_frontier(fit_coll, dev_batches, fits[p], B, k_dev, f["k1"], f["z"]) for p in feasible}
        para = max(feasible, key=lambda p: (round(float(dev[p]["per_batch"]["pessimistic"][0]), 9), -p))
        ff = fits[para]
        r = run_frontier(coll, batches, ff, B, k, f["k1"], f["z"])
        scc = [[self_consistency(coll[i], B) for i in b] for b in batches]
        flat_sc = [c for cs in scc for c in cs]
        sc = M.prec_rep([c.signals["agree"] for c in flat_sc], [c.correct for c in flat_sc], k / len(flat_sc))
        flat = [c for cs in scc for c in cs]
        row = {"para": para, "lambda": ff.lam, "cv": ff.cv, "model": ff.model.summary(),
               "pass_rates": ff.rates, "n_states": ff.n_states,
               "dev_prec": {str(p): float(d["per_batch"]["pessimistic"][0]) for p, d in dev.items()},
               **{kk: r[kk] for kk in ("calls", "calls_sd", "calls_min", "calls_max", "topk_changed_share",
                                       "calibration")},
               "ranking": r["ranking"] | {"self_consistency": M.rank_metrics(
                   [c.signals["agree"] for c in flat], [c.correct for c in flat])},
               "prec": {n: float(v[0]) for n, v in r["per_batch"].items()} | {"self_consistency": float(sc[0])},
               "paired_pess_vs_mean": M.paired_compare(r["per_batch"]["pessimistic"], r["per_batch"]["mean"]),
               "paired_pess_vs_sc": M.paired_compare(r["per_batch"]["pessimistic"], sc)}
        if str(B) in chosen:
            a = chosen[str(B)]["allocation"]
            bc = [cel_alloc(coll[i], a) for b in batches for i in b]
            borda = M.prec_rep(borda_scores(bc, SIGNALS), [c.correct for c in bc], k / len(bc))
            row["prec"]["celk_borda"] = float(borda[0])
            row["paired_pess_vs_celk"] = M.paired_compare(r["per_batch"]["pessimistic"], borda)
        out["by_k"][str(B)] = row
    return out


KNAPSACK = {"max_extra": 6, "prune_tol": 0.01, "n_boot": 200}
PRIOR = {"draws": 5, "n_boot": 1000, "quantile": 0.95}


def knapsack_section(coll: dict[str, CaseData], batches: list[list[str]], chosen: dict,
                     budget_ks: Sequence[int], k: int, fit_coll: dict[str, CaseData],
                     scoring: Optional[dict] = None, knapsack: Optional[dict] = None) -> dict:
    """SHAP importance of every signal on D_fit (all signals bought), pruning of free
    signals, values per call, and per budget B the Prop. K allocation (label-free survival,
    model-based values). The allocation is scored with the pessimistic score fitted on D_fit
    and compared, paired over batches, with self-consistency (B draws) and the dev-enumerated
    CEL-k (Borda), the offline reference it replaces."""
    o, kc = {**SCORING, **(scoring or {})}, {**KNAPSACK, **(knapsack or {})}
    prior = tuple(o["prior"])
    fit = list(fit_coll.values())
    full = full_cases(fit, para=True)
    z_full = np.array([c.correct for c in full])
    if z_full.all() or not z_full.any():
        return {"note": "D_fit has one class only"}
    imp = importance(full, z_full, o["ridge"], prior, kc["n_boot"], drop=_gray(o))
    dropped = prune(imp, kc["prune_tol"], _gray(o))
    if set(dropped) - set(_gray(o)):
        imp = importance(full, z_full, o["ridge"], prior, kc["n_boot"], drop=dropped)
    values = item_values(imp, float(np.mean([c.signals["k"] for c in full])))
    out = {"n_fit": len(fit), "importance": imp, "pruned": dropped, "values": values,
           "value_order": sorted(values, key=values.get, reverse=True), "by_k": {}}
    for B in budget_ks:
        ks = knapsack_allocation(values, fit, B, max_extra=kc["max_extra"])
        if not ks:
            continue
        a = ks["allocation"]
        pol = lambda cd, a=a: cel_alloc(cd, a)
        fc = [pol(cd) for cd in fit]
        zf = np.array([c.correct for c in fc])
        if zf.all() or not zf.any():
            continue
        lam, _ = choose_lambda(fc, zf, o["lambdas"], k / len(batches[0]), o["cv_folds"], o["ridge"], prior,
                               drop=dropped)
        model = fit_score_model(fc, zf, o["ridge"], prior, dropped)
        per, sc, calls = [], [], []
        pooled = {"knapsack": ([], []), "self_consistency": ([], [])}
        for b in batches:
            cases = [pol(coll[i]) for i in b]
            zz = np.array([c.correct for c in cases])
            mu, sd = model.mu_sigma(cases)
            per = M.prec_rep(mu - lam * sd, zz, k / len(zz))
            scc = [self_consistency(coll[i], B) for i in b]
            pooled["knapsack"][0].extend(mu - lam * sd), pooled["knapsack"][1].extend(zz)
            pooled["self_consistency"][0].extend(c.signals["agree"] for c in scc)
            pooled["self_consistency"][1].extend(c.correct for c in scc)
            sc = M.prec_rep([c.signals["agree"] for c in scc], [c.correct for c in scc], k / len(scc))
            calls += [c.calls for c in cases]
        row = {"allocation": a.label(), "expected_calls_dev": ks["expected_calls"], "survival_dev": ks["survival"],
               "calls": float(np.mean(calls)), "lambda": lam,
               "prec": {"knapsack": float(per[0]), "self_consistency": float(sc[0])},
               "ranking": {n: M.rank_metrics(*v) for n, v in pooled.items()},
               "paired_vs_sc": M.paired_compare(per, sc)}
        if str(B) in chosen:
            e = chosen[str(B)]["allocation"]
            ec = [cel_alloc(coll[i], e) for b in batches for i in b]
            en = M.prec_rep(borda_scores(ec, SIGNALS), [c.correct for c in ec], k / len(ec))
            row.update(enumerated=e.label(), paired_vs_enumerated=M.paired_compare(per, en))
            row["prec"]["enumerated_borda"] = float(en[0])
        out["by_k"][str(B)] = row
    return out


def prior_section(coll: dict[str, CaseData], holdout: dict[str, CaseData], plan: dict,
                  prior: Optional[dict] = None) -> dict:
    """Certified mode with a fitted prior (stability_prior.py, model-based): Beta(a, b) from
    the agreement counts of the first `draws` fresh-paraphrase draws of the labelled holdout
    (D_fit + D_cal; labels not read), depth = bootstrap quantile. Replayed on eval next to
    the exact Jeffreys depth: a case is certified when its first max(depth, screen) fresh
    draws all agree; calls stop at the first disagreement. Accuracy of the certified set is
    a metric (reads labels); the certificate itself is about stability."""
    pc = {**PRIOR, **(prior or {})}
    st, n = plan["settings"], pc["draws"]
    counts = []
    for cd in holdout.values():
        ans, m, _ = majority([d.answer for d in cd.fresh[:n]])
        counts.append(m if ans is not None else 0)
    fp = fitted_prior(counts, n, st["eps0"], st["p0_star"], pc["n_boot"], pc["quantile"])
    s1, have = CELConfig().s1, min(len(cd.fresh) for cd in coll.values())
    rows = {}
    for name, depth, ab in (("fitted", fp["depth"], (fp["a"], fp["b"])), ("jeffreys", plan["depth"], (0.5, 0.5))):
        D = max(depth, s1)
        if D > have:
            rows[name] = {"depth": depth, "note": f"needs {D} draws, collection has {have}"}
            continue
        cert, calls, eps = [], [], []
        for cd in coll.values():
            ans = [d.answer for d in cd.fresh[:D]]
            first = ans[0]
            stop = next((i for i, a in enumerate(ans) if a is None or a != first), None)
            ok = first is not None and stop is None
            calls.append(1 + (D if stop is None else max(stop + 1, min(s1, D))))
            if ok:
                cert.append(first == cd.gold)
                eps.append(posterior_tail(D, D, *ab, st["p0_star"]))
        rows[name] = {"depth": depth, "draws_used": D, "certified_share": len(cert) / len(coll),
                      "certified_accuracy": float(np.mean(cert)) if cert else float("nan"),
                      "mean_calls": float(np.mean(calls)), "eps": float(np.mean(eps)) if eps else float("nan"),
                      "guarantee": "exact" if name == "jeffreys" else "model-based"}
    return {"prior": fp, "replay": rows, "glr": {"unanimous_depth_delta_0.1": glr_depth(0.1),
            "note": "GLR needs more draws than the collection holds" if glr_depth(0.1) > have else ""}}


def run_protocol(coll: dict[str, CaseData], batches: list[list[str]], k: int = 10,
                 budget: Optional[int] = None, n_curve_seeds: int = 20,
                 reseed_coll: Optional[dict[str, CaseData]] = None,
                 deploy_sc_draws: int = 3, budget_ks: Sequence[int] = (2, 3, 4, 5),
                 dev_coll: Optional[dict[str, CaseData]] = None,
                 dev_batches: Optional[list[list[str]]] = None,
                 sprt_settings: Optional[dict] = None,
                 linkage_ks: Sequence[int] = (),
                 cal_coll: Optional[dict[str, CaseData]] = None,
                 scoring: Optional[dict] = None,
                 frontier: Optional[dict] = None,
                 knapsack: Optional[dict] = None,
                 prior: Optional[dict] = None) -> dict:
    base, sprt, plan = cel_from_settings(sprt_settings)        # stage-2 depth from the SPRT settings
    res: dict = {"theory": theory_constants(sprt), "certification": plan, "k": k,
                 "n_batches": len(batches), "deploy_sc_draws": deploy_sc_draws}
    # Matched budget: unless fixed, baselines get the pipeline's REALISED mean calls
    # per case on this collection (its spend is adaptive, so it is only known here).
    check_collection(coll, deploy_sc_draws=deploy_sc_draws, budget_ks=budget_ks, base=base)
    if budget is None:
        probe = evaluate(coll, batches, registry(8, base=base, sprt=sprt)["cel"], k)
        budget = max(2, int(round(probe["mean_calls"])))
    check_collection(coll, budget, deploy_sc_draws, budget_ks, base)
    res["matched_budget"] = budget
    vpd = all(cd.meta.get("verbal_per_draw") for cd in coll.values())
    o = {**SCORING, **(scoring or {})}
    k_frac = k / len(batches[0])
    reg = registry(budget, deploy_sc_draws, base, sprt, vpd)
    fit_only = {i: dev_coll[i] for b in dev_batches for i in b} if dev_coll and dev_batches else {}
    cel_scored = None
    if fit_only:                                                 # fitted baselines, frozen on D_fit
        res["baselines"] = fit_baselines(coll, batches, fit_only, budget, reg, o)
        f = fit_scorer([reg["cel"](cd) for cd in fit_only.values()], k_frac, o)
        if f is not None:                                        # adaptive CEL ranked by mu - lambda*sigma
            cel_scored = scored(reg["cel"], f[0], f[1])
            res["cel_scored"] = {"lambda": f[1], "cv": f[2]}

    # 1. baseline ladder + ablations at matched budget ---------------------------
    main = {}
    for name, pol in reg.items():
        main[name] = evaluate(coll, batches, pol, k, exact_random=(name == "random"))
    main["cel_borda"] = evaluate(coll, batches, reg["cel"], k, ranking=("borda", SIGNALS))
    if cel_scored:
        main["cel_scored"] = evaluate(coll, batches, cel_scored, k)
    res["methods"] = {n: _strip(v) for n, v in main.items()}
    cel_rep = main["cel"]["_rep"]
    res["paired_vs_cel"] = {n: M.paired_compare(cel_rep, v["_rep"])
                             for n, v in main.items() if n != "cel"}
    res["confirmatory"] = res["paired_vs_cel"]["self_consistency"]
    res["calibration"] = {n: _calib(main[n]["_cases"], f) for n, f in PROB_SCORES.items() if n in main}
    if cel_scored:
        res["calibration"] = {"cel_mu": _calib(main["cel_scored"]["_cases"], lambda c: c.signals["mu"])} \
            | res["calibration"]
    # 1b. certified mode at the matched budget (R2b): conformal BH for CEL and every baseline
    if cel_scored and cal_coll:
        st = plan["settings"]
        res["certified_matched"] = certified_matched_section(coll, batches, cal_coll, reg, cel_scored, k, o,
                                                             st["eps0"], st["p0_star"])

    # 2. budget curve (x = realised mean calls/case; averaged over subsample seeds) ----
    curve = {}
    specs = {f"cel:{n}": (lambda c: (lambda cd, seed=None: cel(cd, c, seed, sprt)))(cfg)
             for n, cfg in budget_configs(base).items()}
    specs.update({f"sc:{b}": (lambda b: (lambda cd, seed=None: self_consistency(cd, b, seed)))(b)
                  for b in sorted(set(SC_CURVE_DRAWS) | {deploy_sc_draws})})
    for name, pol in specs.items():
        runs = [evaluate(coll, batches, pol, k, seed=s, sel_seed=s) for s in range(n_curve_seeds)]
        pb = np.mean([r["_rep"] for r in runs], axis=0)            # seed-averaged point and replicates
        curve[name] = {"calls": float(np.mean([r["mean_calls"] for r in runs])), "prec": float(pb[0]),
                       **_band(pb), "auroc": float(np.mean([r["auroc"] for r in runs])),
                       "auarc": float(np.mean([r["auarc"] for r in runs])), **_top_shares(runs, o["extra_fracs"])}
    v = main["verbalized"]
    curve["verbalized"] = {"calls": 1.0, "prec": v["mean_prec"], "lo": v["mean_prec"],
                           "hi": v["mean_prec"], "auroc": v["auroc"], "auarc": v["auarc"]}
    # certified CEL is one policy with a data-set spend: ONE point at its realised mean calls,
    # ranked like CEL-k (mu - lambda*sigma fitted under its own policy); the cel:* variants
    # above are design ablations, not a curve
    ref = main.get("cel_scored", main["cel"])
    curve["certified"] = {k2: v2 for k2, v2 in _summ(ref).items() if k2 in ("calls", "prec", "lo", "hi", "auroc", "auarc")} \
        | {"score": "mu - lambda*sigma" if "cel_scored" in main else "cascade (D_fit has one class)"} \
        | _top_shares([ref], o["extra_fracs"])
    res["budget_curve"] = curve

    # 2b. R1, ranking mode at a fixed budget k: CEL-k (dev-chosen allocation, expected
    #     calls in (k - band, k], ranked by mu - lambda*sigma fitted on D_fit) against EVERY baseline at
    #     exactly k calls (CISC and Pedapati-style refit at each k). Paired over batches.
    if fit_only:
        kap = {g: kappa(q0, sprt=sprt) for g, q0 in DEFAULT_Q0.items()}
        st = plan["settings"]
        cert = {"eps0": st["eps0"], "p0_star": st["p0_star"], "depth": plan["depth"]}
        sel = select_allocations(dev_coll, dev_batches, sorted(set(budget_ks) | set(linkage_ks)), vpd,
                                 max_extra=max(4, base.s2), kappas=kap, cert=cert,
                                 band=o["budget_band"])
        bm = {kk: sel[kk] for kk in ("metric", "n_dev", "n_candidates")} | {"by_k": {}}
        for B, ch in sel["chosen"].items():
            if int(B) not in budget_ks:
                continue
            alloc = ch["allocation"]
            raw = lambda cd, seed=None, a=alloc: cel_alloc(cd, a, seed)
            pol, how = celk_policy(alloc, fit_only, k_frac, o)
            ckb = evaluate(coll, batches, raw, k, ranking=("borda", SIGNALS))
            ck = evaluate(coll, batches, pol, k) if pol else ckb
            pols, info = baseline_policies(list(fit_only.values()), int(B), vpd, bool(o.get("gray_box")))
            evs = {n: evaluate(coll, batches, p, k) for n, p in pols.items()}
            pv = {n: M.paired_compare(ck["_rep"], e["_rep"]) for n, e in evs.items()}
            cal = {n: _calib(evs[n]["_cases"], PROB_SCORES[n]) for n in evs if n in PROB_SCORES}
            if pol:
                cal = {"cel_mu": _calib(ck["_cases"], lambda c: c.signals["mu"])} | cal
            bm["by_k"][B] = {"allocation": alloc.label(), "allocation_params": asdict(alloc),
                             "dev_calls": ch["dev_calls"], "dev_score": ch["dev_score"], **how,
                             "cel_k": _summ(ck), "cel_k_borda": _summ(ckb),
                             "baselines": {n: _summ(e) for n, e in evs.items()},
                             "self_consistency": _summ(evs["self_consistency"]),
                             "self_consistency_verbal": _summ(evs["self_consistency_verbal"]),
                             "paired_vs": pv | {"cel_k_borda": M.paired_compare(ck["_rep"], ckb["_rep"])},
                             "paired": pv["self_consistency"], "paired_vs_verbal": pv["self_consistency_verbal"],
                             "fitted": info, "calibration": cal}
            curve[f"celk:{B}"] = {k2: v2 for k2, v2 in _summ(ck).items() if k2 in ("calls", "prec", "lo", "hi", "auroc", "auarc")} \
                | _top_shares([ck], o["extra_fracs"])
            for n, e in evs.items():
                curve[f"k:{n}:{B}"] = {k2: v2 for k2, v2 in _summ(e).items()
                                       if k2 in ("calls", "prec", "lo", "hi", "auroc", "auarc")} \
                    | _top_shares([e], o["extra_fracs"])
        # 2c'. the metrics table at the elbow of CEL-k's dev curve (both modes side by side)
        res["elbow"] = elbow_section(coll, batches, cal_coll, fit_only, sel, k, k_frac, o, vpd, main)
        if "certified_matched" in res:
            res["elbow"]["certified_cel"]["sprt"] = res["certified_matched"]["sprt"]
        # 2d. score model on D_fit: pessimistic ranking, and certified selection on D_cal
        res["pessimism"], res["certified"] = score_model_section(
            coll, batches, sel["chosen"], budget_ks, k, fit_only, cal_coll, scoring)
        if not cal_coll:
            res["certified"] = {"note": "no calibration collection (split.n_cal = 0)"}
        # 2e. frontier allocation: uneven evidence, where pessimism can change the top-k
        res["frontier"] = frontier_section(coll, batches, sel["chosen"], budget_ks, k, fit_only,
                                           dev_batches, scoring, frontier)
        # 2f. SHAP values and the Prop. K knapsack allocation; 2g. fitted prior for the certificate
        res["knapsack"] = knapsack_section(coll, batches, sel["chosen"], budget_ks, k, fit_only, scoring, knapsack)
        res["fitted_prior"] = prior_section(coll, {**fit_only, **(cal_coll or {})}, plan, prior)
    else:
        bm = {"note": "no dev collection: CEL-k's allocation is chosen on the dev slice", "by_k": {}}
    res["budget_matched"] = bm

    # 2c. mode linkage (Sec. 3.9, R4): adaptive CEL is one allocation of CEL-k's space; CEL-k
    #     under the dev-chosen and the derived (Prop. 3) allocation, each ranked by its own
    #     mu - lambda*sigma like R1; certificates of the delivered top-k: SPRT (exact for
    #     stability; model-based) and, with D_cal, conformal BH at the middle alpha (exact).
    if fit_only:
        alpha = sorted(o["alphas"])[len(o["alphas"]) // 2]

        def point(alloc):
            pol, how = celk_policy(alloc, fit_only, k_frac, o)
            raw = lambda cd, seed=None: cel_alloc(cd, alloc, seed)
            ev = evaluate(coll, batches, pol, k) if pol else evaluate(coll, batches, raw, k, ranking=("borda", SIGNALS))
            out = {"calls": ev["mean_calls"], "prec": ev["mean_prec"], "auroc": ev["auroc"], "auarc": ev["auarc"],
                   **_band(ev["_rep"]), **delivered_certificates(ev, kap, cert["eps0"], cert["p0_star"]),
                   "score": how["score"]}
            if pol and cal_coll:
                cal = [pol(cd) for cd in cal_coll.values()]
                s_cal, z_cal = np.array([c.key[0] for c in cal]), np.array([c.correct for c in cal])
                by_id = {c.item_id: c for c in ev["_cases"]}
                share = []
                for b, ids in zip(batches, ev["selected"]):
                    cs = [by_id[i] for i in b]
                    rej = {cs[j].item_id for j in bh(conformal_pvalues(np.array([c.key[0] for c in cs]), s_cal, z_cal), alpha)}
                    share.append(np.mean([i in rej for i in ids]))
                out["certified_conformal"] = float(np.mean(share))
                out["conformal_alpha"] = alpha
            return out
        nested = cel_as_allocation(base, vpd)
        link = {"certification_budget": sel["certification_budget"], "nested_allocation": nested.label(),
                "adaptive": {n: {kk: vv for kk, vv in _summ(ev).items() if kk in ("calls", "prec", "lo", "hi", "auroc", "auarc")}
                             | delivered_certificates(ev, kap, cert["eps0"], cert["p0_star"])
                             for n, c in budget_configs(base).items()
                             for ev in [evaluate(coll, batches, lambda cd, seed=None, c=c: cel(cd, c, None, sprt), k)]},
                "nested": point(nested), "celk": {}, "derived": {}}
        for B in sorted(set(budget_ks) | set(linkage_ks)):
            for kind, src in (("celk", sel["chosen"]), ("derived", sel["derived"])):
                if str(B) in src:
                    ch = src[str(B)]
                    link[kind][str(B)] = {"allocation": ch["allocation"].label(), "dev_calls": ch["dev_calls"],
                                          **({"k_eff": ch["k_eff"]} if "k_eff" in ch else {}),
                                          **point(ch["allocation"])}
        res["linkage"] = link

    # 3. evidence-equivalence arms (Prop. 3) --------------------------------------
    arms = {n: evaluate(coll, batches, pol, k) for n, pol in ARMS.items()}
    res["arms"] = {n: _strip(a) for n, a in arms.items()}
    res["arms_tests"] = {
        "B_vs_A": M.paired_compare(arms["B_6draws_battery"]["_rep"], arms["A_12draws"]["_rep"]),
        "B_vs_C": M.paired_compare(arms["B_6draws_battery"]["_rep"], arms["C_9draws"]["_rep"]),
    }

    # 4. monotonicity (Assumption 1) ----------------------------------------------
    o = main["cel"]
    z_all = [c.correct for c in o["_cases"]]
    res["monotonicity"] = {
        "strata": M.strata_accuracy(o["_scores"], z_all),
        "isotonic_gap": M.isotonic_gap(o["_scores"], z_all, n_boot=500),
        "top_strata": M.top_strata_test([o["_ranks"][c.item_id] for c in o["_cases"]], z_all),
    }

    # 5. held-out probe audit: selected (1-10) vs matched controls (11-30) ----------------
    sel = [c.heldout_pass for c in o["_cases"] if o["_ranks"][c.item_id] <= 10]
    ctl = [c.heldout_pass for c in o["_cases"] if 11 <= o["_ranks"][c.item_id] <= 30]
    res["heldout_audit"] = M.heldout_audit(sel, ctl)
    res["heldout_audit"]["executed_calls"] = sum(1 for c in o["_cases"] if o["_ranks"][c.item_id] <= 30)

    # 6. signal effects: leave-one-out (prec@k, AUROC, PRR) and Shapley (prec@k) ---------
    # (a) leave-one-out: the cascade itself (CEL) minus CEL without signal j, paired over
    #     batches for prec@k; AUROC/PRR pooled; calls saved by dropping j.
    loo = {}
    for j, name in SIGNAL_ABLATIONS.items():
        full, abl, pc = main["cel"], main[name], res["paired_vs_cel"][name]
        loo[j] = {"d_prec": pc["mean_diff"], "ci": [pc["ci_lo"], pc["ci_hi"]],
                  "d_auroc": full["auroc"] - abl["auroc"], "d_auarc": full["auarc"] - abl["auarc"],
                  "d_prr": full["prr"] - abl["prr"],
                  "calls_saved": full["mean_calls"] - abl["mean_calls"]}
    res["leave_one_out"] = loo

    # (b) exact Shapley over signals on prec@k: Borda over each of the 32 coalitions,
    #     screen fixed, V(empty) = random among survivors. V is computed per batch; phi is
    #     linear in V, so the per-batch phi gives a bootstrap CI over batches.
    per_batch_cases = [[cel(coll[i], base, sprt=sprt) for i in b] for b in batches]
    z_batches = [np.array([c.correct for c in cases], float) for cases in per_batch_cases]

    def prec_by_batch(C: frozenset) -> np.ndarray:
        # one population: [point (averaged over tie-breaks), item-bootstrap replicates];
        # phi is linear in V, so the replicates give phi's bootstrap interval
        cases, z = per_batch_cases[0], z_batches[0]
        return M.prec_rep(borda_scores(cases, sorted(C)), z, k / len(z), n_ties=50)

    shb = exact_shapley(list(SIGNALS), prec_by_batch)          # V and phi: replicate arrays
    sh = {"phi": {j: float(v[0]) for j, v in shb["phi"].items()},
          "phi_ci": {j: [float(q) for q in np.quantile(v[1:], [.025, .975])] for j, v in shb["phi"].items()},
          "V": {c: float(v[0]) for c, v in shb["V"].items()},
          "total_gain": float(shb["total_gain"][0]),
          "efficiency_check": float(np.max(np.abs(shb["efficiency_check"])))}
    sh["phi_per_call"] = {j: sh["phi"][j] / SIGNAL_COST[j] for j in SIGNALS}
    sh["random_reference"] = main["random"]["mean_prec"]
    sh["screen_gain"] = sh["V"]["{}"] - main["random"]["mean_prec"]
    sh["flip_increment"] = sh["V"]["L,flip"] - sh["V"]["L"]
    res["shapley"] = sh

    # 7. empirical exchange rate kappa_hat (Prop. 3) -----------------------------------
    agree, passes, zz = [], {g: [] for g in ("neg", "relabel", "distractor")}, []
    for cd in coll.values():
        ans, m, _ = majority([d.answer for d in cd.fresh[:9]])
        if ans is None:
            continue
        agree.append(m)
        zz.append(ans == cd.gold)
        for g in passes:
            pa = cd.probes.get(g)
            target = ({"yes": "no", "no": "yes"}[ans] if g == "neg" else ans)
            passes[g].append(float(pa == target))
    res["kappa_hat"] = M.empirical_kappa(agree, passes, zz, n_boot=300)
    res["kappa_derived"] = {g: kappa(DEFAULT_Q0[g]) for g in passes}

    # 8. selection stability (Jaccard vs chance floor), if a re-drawn collection exists --
    floor = M.jaccard_chance_floor(len(batches[0]), k)
    if reseed_coll is not None:
        idx = [i for i, b in enumerate(batches) if all(x in reseed_coll for x in b)]
        re_eval = evaluate(reseed_coll, [batches[i] for i in idx], reg["cel"], k)
        js = [M.jaccard(set(o["selected"][i]), set(re_eval["selected"][j]))
              for j, i in enumerate(idx)]
        tc = [M.tie_aware_consistency(set(re_eval["selected"][j]), set(o["_groups"][i]), k)
              for j, i in enumerate(idx)]
        res["jaccard"] = {"per_batch": js, "mean": float(np.mean(js)), "chance_floor": floor,
                          "tie_aware": float(np.mean(tc)),
                          "top_group_size": float(np.mean([len(o["_groups"][i]) for i in idx]))}
    else:                                        # no re-collection: re-draw from the cached draws
        a1, a2 = (evaluate(coll, batches, reg["cel"], k, seed=s, sel_seed=s) for s in (1, 2))
        js = [M.jaccard(set(x), set(y)) for x, y in zip(a1["selected"], a2["selected"])]
        tc = [M.tie_aware_consistency(set(y), set(g), k) for y, g in zip(a2["selected"], a1["_groups"])]
        res["jaccard"] = {"mean": float(np.mean(js)), "chance_floor": floor, "tie_aware": float(np.mean(tc)),
                          "top_group_size": float(np.mean([len(g) for g in a1["_groups"]])),
                          "source": "two re-draws of the cached draws (seeds 1, 2); no re-collection"}

    # 9. judge sensitivity, if an alternative judge's labels exist -----------------------
    if all(cd.alt_clusters for cd in coll.values()):
        alt = {}
        for i, cd in coll.items():
            c2 = copy.deepcopy(cd)
            pool = c2.fresh + c2.same + [d for row in c2.reuse for d in row[1:]]
            for d, lab in zip(pool, cd.alt_clusters["labels"]):
                d.cluster = lab
            alt[i] = c2
        alt_eval = evaluate(alt, batches, reg["cel"], k)
        se1 = [c.signals.get("se") for c in o["_cases"] if c.survived]
        se2 = [c.signals.get("se") for c in alt_eval["_cases"] if c.survived]
        res["judge_sensitivity"] = {
            "se_spearman": float(spearmanr(se1, se2).statistic),
            "top10_jaccard": float(np.mean([M.jaccard(set(a), set(b)) for a, b in
                                            zip(o["selected"], alt_eval["selected"])]))}
    return res


def cell_results(cfg: dict, ds: str, mk: str, force: bool = False) -> dict:
    """Protocol result for one (dataset, model) cell of a config, cached as
    <results>/<ds>/<config>/<ds>__<mk>.json and recomputed if missing or forced."""
    path = results_dir(cfg, ds) / f"{ds}__{mk}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    coll_path = collection_path(cfg, ds, mk)
    if not coll_path.exists():
        raise SystemExit(f"missing collection {coll_path} -- run experiments/scripts/collect.py first")
    record_run({**cfg, "_datasets": [ds], "_models": [mk]}, "evaluate")
    _, split = load_split(cfg, ds)
    rs = collection_path(cfg, ds, mk, "__reseed")
    dev_path = collection_path(cfg, ds, mk, "__dev")
    cal_path = collection_path(cfg, ds, mk, "__cal")
    bs = cfg["split"].get("batch_size") or max(1, len(split.dev))   # default: one dev population
    dev_ids = [it.item_id for it in split.dev]
    try:
        res = run_protocol(load_collection(coll_path), split.batches, cfg["k"], cfg.get("budget"),
                           reseed_coll=load_collection(rs) if rs.exists() else None,
                           deploy_sc_draws=cfg.get("deployment_sc_draws", 3),
                           budget_ks=tuple(cfg.get("budget_matched_k", (2, 3, 4, 5))),
                           dev_coll=load_collection(dev_path) if dev_path.exists() else None,
                           dev_batches=[dev_ids[i:i + bs] for i in range(0, len(dev_ids), bs)],
                           sprt_settings=cfg.get("sprt"),
                           linkage_ks=tuple(cfg.get("linkage_k", ())),
                           cal_coll=load_collection(cal_path) if cal_path.exists() else None,
                           scoring=cfg.get("scoring"), frontier=cfg.get("frontier"),
                           knapsack=cfg.get("knapsack"), prior=cfg.get("prior"))
    except ValueError as e:                        # e.g. collection too small for the protocol
        raise SystemExit(f"{ds}/{mk}: {e}") from None
    write_report(res, results_dir(cfg, ds), f"{ds}__{mk}")
    return json.loads(path.read_text())


def evaluate_config(cfg: dict) -> dict:
    """Evaluate every selected cell (recomputed), then Holm-correct the confirmatory
    p-values across cells; writes <results>/all/<config>/summary.json and returns it."""
    summary = {}
    for ds, mk in cells(cfg):
        res = cell_results(cfg, ds, mk, force=True)
        summary[f"{ds}__{mk}"] = {"cel": res["methods"]["cel"]["mean_prec"],
                                  "self_consistency": res["methods"]["self_consistency"]["mean_prec"],
                                  "confirmatory_p": res["confirmatory"]["p_boot"]}
    adj = M.holm({k: v["confirmatory_p"] for k, v in summary.items()})
    for k in summary:
        summary[k]["confirmatory_p_holm"] = adj[k]
    out = results_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
