"""The Sec. 4 protocol for one (dataset, model) collection.

run_protocol() returns a JSON-serialisable dict; cell_report.py renders it. Every
number is a re-scoring of the cached collection -- no API calls happen here.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
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
from .utility.config import cells, record_run
from .ranking import borda_scores, key_scores, ranks_within, select_top_k
from .methods import (ARMS, SIGNAL_ABLATIONS, CELConfig, CaseScore, allocation_space, budget_configs,
                      cel, cel_alloc, cel_as_allocation, cel_from_settings, evidence_depth, registry,
                      self_consistency_verbal,
                       self_consistency, verbalized)
from .signal_shapley import exact_shapley
from .sprt_posterior import (SPRT, DEFAULT_Q0, certification_depth, kappa, majority, tail_mass,
                             tail_mass_effective)
from .utility.records import CaseData

SIGNALS = ("L", "flip", "se", "chal", "verb")
SC_CURVE_DRAWS = (3, 6, 9, 12)                                      # self-consistency budget curve
SIGNAL_COST = {"L": 6, "flip": 3, "se": 1, "chal": 1, "verb": 1}   # stage-2 calls per signal


def _scores(cases: list[CaseScore], ranking) -> np.ndarray:
    if ranking == "key":
        return key_scores(cases)
    _, sigs = ranking
    return borda_scores(cases, sigs)


def evaluate(coll: dict[str, CaseData], batches: list[list[str]], policy: Callable,
             k: int = 10, seed: Optional[int] = None, ranking="key",
             exact_random: bool = False, sel_seed: int = 0) -> dict:
    """Per-batch prec@k (+ hypergeometric null), global AUROC/PRR, calls."""
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
    out = {"per_batch": per_batch, "mean_prec": float(np.mean(per_batch)), "K_b": K_b,
           "base_acc": float(np.mean(z_all)),
           "mean_calls": float(np.mean([c.calls for c in all_cases])),
           "survival": float(np.mean([c.survived for c in all_cases])),
           "auroc": M.auroc(s_all, z_all), "prr": M.prr(s_all, z_all),
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
    }


def _expected_prec(cases: list[CaseScore], signals: Sequence[str], k: int, n_draws: int = 50) -> float:
    """prec@k of Borda over `signals`, averaged over random tie-breaks."""
    sc = borda_scores(cases, sorted(signals))
    z = np.array([c.correct for c in cases], float)
    return float(np.mean([z[select_top_k(sc, k, random.Random(s))].mean() for s in range(n_draws)]))


def select_allocations(dev_coll: dict[str, CaseData], dev_batches: list[list[str]], k: int,
                       budgets: Sequence[int], verbal_per_draw: bool, max_extra: int = 6,
                       kappas: Optional[dict] = None, cert: Optional[dict] = None) -> dict:
    """Choose CEL-k's allocation per budget on the DEV slice (nothing tuned on eval): every
    replayable allocation is scored on dev -- prec@k (AUROC if no dev batch exceeds k) of
    Borda over its signals -- and, per budget B, the best one whose EXPECTED calls per case
    on dev are <= B is kept (ties: fewer calls). max_extra >= adaptive CEL's s2, so adaptive
    CEL is in the space (Sec. 3.9, nesting).

    With `kappas`, also the DERIVED allocation per budget (Sec. 3.9): among fresh-paraphrase
    allocations (Lemma 1), the one maximising k_eff (Prop. 3, model-based) under the same
    expected-call constraint. Label-free: it reads only dev call counts. With `cert`
    ({eps0, p0_star, depth}), also the certification budgets: the least expected calls at
    which a unanimous survivor can be certified, exactly (>= depth draws, Cor. 1) and
    model-based (eps_eff <= eps0, Prop. 3)."""
    usable = [b for b in dev_batches if len(b) > k]
    max_draws = min(min(len(cd.same), len(cd.fresh)) for cd in dev_coll.values())
    space = allocation_space(max_extra=max_extra, verbal_per_draw=verbal_per_draw, max_draws=max_draws)
    rows = []
    for alloc in space:
        cases = [[cel_alloc(dev_coll[i], alloc) for i in b] for b in dev_batches]
        calls = float(np.mean([c.calls for cs in cases for c in cs]))
        if usable:
            score = float(np.mean([_expected_prec(cs, SIGNALS, k, 20) for cs in cases if len(cs) > k]))
        else:
            sc = np.concatenate([borda_scores(cs, SIGNALS) for cs in cases])
            score = M.auroc(sc, [c.correct for cs in cases for c in cs])
        rows.append((alloc, calls, score))
    chosen = {}
    for B in budgets:
        feasible = [r for r in rows if r[1] <= B + 1e-9]
        if feasible:
            alloc, calls, score = max(feasible, key=lambda r: (round(r[2], 9), -r[1]))
            chosen[str(B)] = {"allocation": alloc, "dev_calls": calls, "dev_score": score}
    out = {"metric": "prec@k" if usable else f"AUROC (no dev batch larger than k={k})",
           "n_dev": sum(map(len, dev_batches)), "n_candidates": len(space), "chosen": chosen}
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


def run_protocol(coll: dict[str, CaseData], batches: list[list[str]], k: int = 10,
                 budget: Optional[int] = None, n_curve_seeds: int = 20,
                 reseed_coll: Optional[dict[str, CaseData]] = None,
                 deploy_sc_draws: int = 3, budget_ks: Sequence[int] = (2, 3, 4, 5),
                 dev_coll: Optional[dict[str, CaseData]] = None,
                 dev_batches: Optional[list[list[str]]] = None,
                 sprt_settings: Optional[dict] = None,
                 linkage_ks: Sequence[int] = ()) -> dict:
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
    reg = registry(budget, deploy_sc_draws, base, sprt)

    # 1. baseline ladder + ablations at matched budget ---------------------------
    main = {}
    for name, pol in reg.items():
        main[name] = evaluate(coll, batches, pol, k, exact_random=(name == "random"))
    main["cel_borda"] = evaluate(coll, batches, reg["cel"], k, ranking=("borda", SIGNALS))
    res["methods"] = {n: _strip(v) for n, v in main.items()}
    cel_pb = main["cel"]["per_batch"]
    res["paired_vs_cel"] = {n: M.paired_compare(cel_pb, v["per_batch"])
                             for n, v in main.items() if n != "cel"}
    res["confirmatory"] = res["paired_vs_cel"]["self_consistency"]

    # 2. budget curve (x = realised mean calls/case; averaged over subsample seeds) ----
    curve = {}
    specs = {f"cel:{n}": (lambda c: (lambda cd, seed=None: cel(cd, c, seed, sprt)))(cfg)
             for n, cfg in budget_configs(base).items()}
    specs.update({f"sc:{b}": (lambda b: (lambda cd, seed=None: self_consistency(cd, b, seed)))(b)
                  for b in sorted(set(SC_CURVE_DRAWS) | {deploy_sc_draws})})
    for name, pol in specs.items():
        runs = [evaluate(coll, batches, pol, k, seed=s, sel_seed=s) for s in range(n_curve_seeds)]
        pb = np.mean([r["per_batch"] for r in runs], axis=0)       # seed-averaged per batch
        boots = np.random.default_rng(0).choice(pb, (2000, len(pb))).mean(axis=1)
        curve[name] = {"calls": float(np.mean([r["mean_calls"] for r in runs])),
                       "prec": float(pb.mean()), "lo": float(np.quantile(boots, .025)),
                       "hi": float(np.quantile(boots, .975)),
                       "auroc": float(np.mean([r["auroc"] for r in runs]))}
    v = main["verbalized"]
    curve["verbalized"] = {"calls": 1.0, "prec": v["mean_prec"], "lo": v["mean_prec"],
                           "hi": v["mean_prec"], "auroc": v["auroc"]}
    res["budget_curve"] = curve

    # 2b. budget-allocated CEL-k vs self-consistency with k draws, compared on EXPECTED
    #     calls per case: CEL-k's mean under its dev-chosen allocation (screen + survivors'
    #     probes, batch average), self-consistency's exact k.
    if dev_coll and dev_batches:
        vpd = bool(next(iter(coll.values())).meta.get("verbal_per_draw"))
        kap = {g: kappa(q0, sprt=sprt) for g, q0 in DEFAULT_Q0.items()}
        st = plan["settings"]
        cert = {"eps0": st["eps0"], "p0_star": st["p0_star"], "depth": plan["depth"]}
        sel = select_allocations(dev_coll, dev_batches, k, sorted(set(budget_ks) | set(linkage_ks)), vpd,
                                 max_extra=max(4, base.s2), kappas=kap, cert=cert)
        bm = {kk: sel[kk] for kk in ("metric", "n_dev", "n_candidates")} | {"by_k": {}}
        summ = lambda e: {"prec": e["mean_prec"], "calls": e["mean_calls"], "auroc": e["auroc"], "prr": e["prr"],
                          "survival": e["survival"]}
        for B, ch in sel["chosen"].items():
            if int(B) not in budget_ks:
                continue
            alloc = ch["allocation"]
            ck = evaluate(coll, batches, lambda cd, seed=None, a=alloc: cel_alloc(cd, a), k, ranking=("borda", SIGNALS))
            sc = evaluate(coll, batches, lambda cd, seed=None, b=int(B): self_consistency(cd, b, seed), k)
            scv = evaluate(coll, batches, lambda cd, seed=None, b=int(B): self_consistency_verbal(cd, b, seed), k)
            bm["by_k"][B] = {"allocation": alloc.label(), "allocation_params": asdict(alloc),
                             "dev_calls": ch["dev_calls"], "dev_score": ch["dev_score"],
                             "cel_k": summ(ck), "self_consistency": summ(sc),
                             "self_consistency_verbal": summ(scv),
                             "paired_vs_verbal": M.paired_compare(ck["per_batch"], scv["per_batch"]),
                             "paired": M.paired_compare(ck["per_batch"], sc["per_batch"])}
            pb = np.asarray(ck["per_batch"], float)
            boots = np.random.default_rng(0).choice(pb, (2000, len(pb))).mean(axis=1)
            curve[f"celk:{B}"] = {"calls": ck["mean_calls"], "prec": float(pb.mean()),
                                  "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975)),
                                  "auroc": ck["auroc"]}
    else:
        bm = {"note": "no dev collection: CEL-k's allocation is chosen on the dev slice", "by_k": {}}
    res["budget_matched"] = bm

    # 2c. mode linkage (Sec. 3.9): adaptive CEL is one allocation of CEL-k's space; CEL-k's
    #     dev-chosen allocation vs the derived (Prop. 3) one; certificates of delivered sets.
    if dev_coll and dev_batches:
        def point(ev):
            boots = np.random.default_rng(0).choice(ev["per_batch"], (2000, len(ev["per_batch"]))).mean(axis=1)
            return {"calls": ev["mean_calls"], "prec": ev["mean_prec"], "auroc": ev["auroc"],
                    "lo": float(np.quantile(boots, .025)), "hi": float(np.quantile(boots, .975)),
                    **delivered_certificates(ev, kap, cert["eps0"], cert["p0_star"])}
        borda = lambda a: evaluate(coll, batches, lambda cd, seed=None: cel_alloc(cd, a), k,
                                   ranking=("borda", SIGNALS))
        nested = cel_as_allocation(base, vpd)
        link = {"certification_budget": sel["certification_budget"], "nested_allocation": nested.label(),
                "adaptive": {n: point(evaluate(coll, batches, lambda cd, seed=None, c=c: cel(cd, c, None, sprt), k))
                             for n, c in budget_configs(base).items()},
                "nested": point(borda(nested)), "celk": {}, "derived": {}}
        for B in sorted(set(budget_ks) | set(linkage_ks)):
            for kind, src in (("celk", sel["chosen"]), ("derived", sel["derived"])):
                if str(B) in src:
                    ch = src[str(B)]
                    link[kind][str(B)] = {"allocation": ch["allocation"].label(), "dev_calls": ch["dev_calls"],
                                          **({"k_eff": ch["k_eff"]} if "k_eff" in ch else {}),
                                          **point(borda(ch["allocation"]))}
        res["linkage"] = link

    # 3. evidence-equivalence arms (Prop. 3) --------------------------------------
    arms = {n: evaluate(coll, batches, pol, k) for n, pol in ARMS.items()}
    res["arms"] = {n: _strip(a) for n, a in arms.items()}
    res["arms_tests"] = {
        "B_vs_A": M.paired_compare(arms["B_6draws_battery"]["per_batch"], arms["A_12draws"]["per_batch"]),
        "B_vs_C": M.paired_compare(arms["B_6draws_battery"]["per_batch"], arms["C_9draws"]["per_batch"]),
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
                  "d_auroc": full["auroc"] - abl["auroc"], "d_prr": full["prr"] - abl["prr"],
                  "calls_saved": full["mean_calls"] - abl["mean_calls"]}
    res["leave_one_out"] = loo

    # (b) exact Shapley over signals on prec@k: Borda over each of the 32 coalitions,
    #     screen fixed, V(empty) = random among survivors. V is computed per batch; phi is
    #     linear in V, so the per-batch phi gives a bootstrap CI over batches.
    per_batch_cases = [[cel(coll[i], base, sprt=sprt) for i in b] for b in batches]
    z_batches = [np.array([c.correct for c in cases], float) for cases in per_batch_cases]

    def prec_by_batch(C: frozenset) -> np.ndarray:
        out = []
        for cases, z in zip(per_batch_cases, z_batches):
            sc = borda_scores(cases, sorted(C))
            # expected prec@k under random tie-breaking, computed exactly by averaging draws
            out.append(np.mean([z[select_top_k(sc, k, random.Random(s))].mean() for s in range(50)]))
        return np.array(out)

    shb = exact_shapley(list(SIGNALS), prec_by_batch)          # V and phi: per-batch arrays
    idx = np.random.default_rng(0).integers(0, len(batches), (2000, len(batches)))
    sh = {"phi": {j: float(v.mean()) for j, v in shb["phi"].items()},
          "phi_ci": {j: [float(q) for q in np.quantile(v[idx].mean(axis=1), [.025, .975])]
                     for j, v in shb["phi"].items()},
          "V": {c: float(v.mean()) for c, v in shb["V"].items()},
          "total_gain": float(np.mean(shb["total_gain"])),
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
    else:
        res["jaccard"] = {"chance_floor": floor, "note": "run with a re-drawn collection"}

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
    <results>/<ds>__<mk>.json and recomputed if missing or forced."""
    path = Path(cfg["paths"]["results"]) / f"{ds}__{mk}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    coll_path = collection_path(cfg, ds, mk)
    if not coll_path.exists():
        raise SystemExit(f"missing collection {coll_path} -- run experiments/scripts/collect.py first")
    record_run({**cfg, "_datasets": [ds], "_models": [mk]}, "evaluate")
    _, split = load_split(cfg, ds)
    rs = collection_path(cfg, ds, mk, "__reseed")
    dev_path = collection_path(cfg, ds, mk, "__dev")
    bs = cfg["split"]["batch_size"]
    dev_ids = [it.item_id for it in split.dev]
    try:
        res = run_protocol(load_collection(coll_path), split.batches, cfg["k"], cfg.get("budget"),
                           reseed_coll=load_collection(rs) if rs.exists() else None,
                           deploy_sc_draws=cfg.get("deployment_sc_draws", 3),
                           budget_ks=tuple(cfg.get("budget_matched_k", (2, 3, 4, 5))),
                           dev_coll=load_collection(dev_path) if dev_path.exists() else None,
                           dev_batches=[dev_ids[i:i + bs] for i in range(0, len(dev_ids), bs)],
                           sprt_settings=cfg.get("sprt"),
                           linkage_ks=tuple(cfg.get("linkage_k", ())))
    except ValueError as e:                        # e.g. collection too small for the protocol
        raise SystemExit(f"{ds}/{mk}: {e}") from None
    write_report(res, cfg["paths"]["results"], f"{ds}__{mk}")
    return json.loads(path.read_text())


def evaluate_config(cfg: dict) -> dict:
    """Evaluate every selected cell (recomputed), then Holm-correct the confirmatory
    p-values across cells; writes <results>/summary.json and returns it."""
    summary = {}
    for ds, mk in cells(cfg):
        res = cell_results(cfg, ds, mk, force=True)
        summary[f"{ds}__{mk}"] = {"cel": res["methods"]["cel"]["mean_prec"],
                                  "self_consistency": res["methods"]["self_consistency"]["mean_prec"],
                                  "confirmatory_p": res["confirmatory"]["wilcoxon_p"]}
    adj = M.holm({k: v["confirmatory_p"] for k, v in summary.items()})
    for k in summary:
        summary[k]["confirmatory_p_holm"] = adj[k]
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
