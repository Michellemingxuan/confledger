"""Signal importance, pruning and the knapsack spending order (Props. K, K'; decisions.md #19).

1. Fit the score model on D_fit cases replayed with every signal bought.
2. SHAP of a model linear in the log-odds has a closed form (interventional, features
   independent): phi_g(x) = beta_g * (x_g - mean x_g). Importance I_g = mean |phi_g|, with
   a bootstrap interval over D_fit cases (refits).
3. Prune free signals whose importance is small or unstable (lower bound < tol); a
   pruned signal is fixed at zero weight in every later fit.
4. Value per call v_g = I_g / c_g. A draw's value is the stability term's importance
   divided by the draws behind it.
5. Prop. K (budget form, model-based): for a screen of k1 draws with dev survival rate
   sigma (label-free), every stage-2 item costs sigma in expectation, so the best plan
   within budget B buys the floor((B - para - k1) / sigma) items of largest value; the
   overall plan is the best of the screens.

Model-based: values come from one fitted model and treat signals as independent.
Coalition Shapley over refits (signal_shapley.py) stays the audit.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .methods import FLIPS, Allocation, CaseScore, cel_alloc
from .score_model import FEATURES, fit_score_model, _design, _fill
from .utility.records import CaseData

ITEM_FEATURE = {"draw": "eta", "neg": "neg", "relabel": "relabel", "distractor": "distractor",
                "chal": "chal", "se": "se"}
FREE = ("verb", "survived", "length", "hedge", "p_first")   # signals that cost no call


def full_allocation(max_draws: int, para: bool = True) -> Allocation:
    """Every signal bought: 1-draw screen (nobody eliminated), all draws, flips, challenge, SE."""
    return Allocation(1, para, max_draws - 1, FLIPS, True, True)


def shap_importance(model, cases: Sequence[CaseScore]) -> dict[str, float]:
    X, _ = _design(cases, model.prior)
    X, _ = _fill(X, model.mean)
    phi = model.coef * (X - X.mean(axis=0))
    return {f: float(np.abs(phi[:, i]).mean()) for i, f in enumerate(FEATURES) if f != "intercept"}


def importance(cases: Sequence[CaseScore], z: Sequence[bool], ridge: float = 1.0,
               prior=(0.5, 0.5), n_boot: int = 200, seed: int = 0, drop=(),
               standardize: bool | Sequence[str] = False) -> dict:
    """I_g on D_fit with a bootstrap 90% interval, and the importance ranking."""
    z = np.asarray(z, bool)
    model = fit_score_model(cases, z, ridge, prior, drop, standardize)
    est = shap_importance(model, cases)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cases), len(cases))
        if z[idx].all() or not z[idx].any():
            continue
        sub = [cases[i] for i in idx]
        boots.append(shap_importance(fit_score_model(sub, z[idx], ridge, prior, drop, standardize), sub))
    ci = {f: [float(np.quantile([b[f] for b in boots], q)) for q in (0.05, 0.95)] for f in est}
    return {"I": est, "ci": ci, "rank": sorted(est, key=est.get, reverse=True)}


def prune(imp: dict, tol: float = 0.01, keep_out=()) -> list[str]:
    """Free signals whose importance lower bound is below `tol`, plus `keep_out`."""
    return sorted(set(keep_out) | {f for f in FREE if imp["ci"][f][0] < tol})


def item_values(imp: dict, draws_in_fit: float) -> dict[str, float]:
    """v_g = I_g / c_g per stage-2 item (each costs 1 call)."""
    v = {g: imp["I"][f] for g, f in ITEM_FEATURE.items()}
    v["neg"] += imp["I"]["neg_drop"]               # confidence under negation rides on the negation call
    v["draw"] = imp["I"]["eta"] / max(draws_in_fit, 1.0)
    return v


def screen_survival(fit: Sequence[CaseData], k1: int, para: bool) -> float:
    """Share of D_fit cases whose first k1 draws agree (label-free)."""
    ok = 0
    for cd in fit:
        ans = [d.answer for d in (cd.fresh if para else cd.same)[:k1]]
        ok += ans[0] is not None and all(a == ans[0] for a in ans)
    return ok / len(fit)


def knapsack_allocation(values: dict[str, float], fit: Sequence[CaseData], budget: float,
                        max_screen: int = 4, max_extra: int = 6, paras=(False, True)) -> dict:
    """Prop. K: per screen, buy the highest-value items that fit the expected budget;
    keep the screen with the largest total value (ties: fewer expected calls). Odd total
    draws only (decisions.md #34): an even total gives up its last draw, or the screen."""
    best = None
    for para in paras:
        for k1 in range(1, max_screen + 1):
            base = int(para) + k1
            if base > budget + 1e-9:
                continue
            sig = screen_survival(fit, k1, para)
            slots = math.inf if sig == 0 else math.floor((budget - base) / sig + 1e-9)
            paid = ("neg", "relabel", "distractor", "chal") + (("se",) if k1 >= 2 else ())   # SE needs 2 rationales
            items = sorted([(values[g], g) for g in paid] + [(values["draw"], "draw")] * max_extra, reverse=True)
            chosen = [g for v, g in items if v > 0][:slots] if slots > 0 else []
            extra = chosen.count("draw")
            if (k1 + extra) % 2 == 0:
                if not extra:
                    continue
                chosen.remove("draw")
                extra -= 1
            total = k1 * values["draw"] + sum(values[g] for g in chosen)
            calls = base + sig * len(chosen)
            cand = (round(total, 12), -calls, Allocation(k1, para, extra, tuple(g for g in FLIPS if g in chosen),
                                                          "chal" in chosen, "se" in chosen), sig)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return {}
    return {"allocation": best[2], "value": best[0], "expected_calls": -best[1], "survival": best[3]}


def full_cases(fit: Sequence[CaseData], para: bool = True) -> list[CaseScore]:
    a = full_allocation(min(len(cd.fresh if para else cd.same) for cd in fit), para)
    return [cel_alloc(cd, a) for cd in fit]
