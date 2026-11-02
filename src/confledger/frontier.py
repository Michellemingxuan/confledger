"""Frontier allocation for CEL-k (methodology.md §6): spend a batch's B*m calls where they
can move the top-k cut.

Every case of the batch starts with k1 draws. Then, one call at a time until the batch
has spent B calls per case on average, the LUCB rule for top-k identification
(Kalyanakrishnan et al. 2012) picks a case: among the current top-k by mu the one with the
lowest mu - z*sigma, among the rest the one with the highest mu + z*sigma; the less
certain of the two (larger sigma) gets the call. The call is the one with the largest
expected move of mu: a one-step lookahead whose pass probabilities mix D_fit's pass rates
by the case's current mu. Final ranking by mu or by the pessimistic mu - lambda*sigma.

Evidence ends up uneven on purpose, which is the setting where pessimism can change the
top-k (decisions.md #17-18). Model-based: the score model, lambda and pass rates are frozen
on D_fit; there is no guarantee. Because a case's calls depend on the other cases in its
batch, frontier scores are not exchangeable with calibration scores, so the conformal
route is not applied to them.

Calls menu: the next draw, each flip probe once, the challenge once; verbal confidence is
free with verbal_per_draw. SE is left out (one judge call with no pass/fail outcome to
look ahead on). Labels are read only on D_fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .certified_selection import pass_rates
from .metrics import calibration, prec_rep, rank_metrics
from .methods import FLIPS, CaseScore, _passes, free_signals
from .score_model import ScoreModel, choose_lambda, fit_score_model
from .sprt_posterior import majority
from .utility.records import CaseData

CALLS = ("draw", *FLIPS, "chal")


@dataclass
class _State:
    """Evidence bought so far for one case."""
    n: int
    probes: list = field(default_factory=list)
    chal: bool = False


def state_case(cd: CaseData, st: _State, para: bool) -> CaseScore:
    """The CaseScore of a partial evidence state (same signal conventions as cel_alloc;
    'survived' = every draw so far agrees)."""
    pool = cd.fresh if para else cd.same
    used = pool[:st.n]
    ans, m, n = majority([d.answer for d in used])
    sig = {"m": m, "k": n}
    if cd.meta.get("verbal_per_draw"):
        confs = [d.confidence for d in used if d.confidence is not None]
        sig["verb"] = sum(confs) / len(confs) if confs else 50.0
    passes = _passes(cd, ans, tuple(st.probes))
    if st.chal:
        sig["chal"] = float(cd.challenge_answer is not None and cd.challenge_base == ans
                            and cd.challenge_answer == cd.challenge_base)
    free_signals(cd, used, ans, sig, passes)
    calls = int(para) + st.n + len(st.probes) + int(st.chal)
    return CaseScore(cd.item_id, ans, ans == cd.gold, calls, ans is not None and m == n, m, sig,
                     probe_passes=passes)


def _options(cd: CaseData, st: _State, cs: CaseScore, para: bool) -> list[str]:
    pool = cd.fresh if para else cd.same
    out = ["draw"] if st.n < len(pool) else []
    if cs.answer is not None:
        out += [g for g in FLIPS if g not in st.probes and cd.probes.get(g) is not None]
        out += ["chal"] * (not st.chal and cd.challenge_answer is not None)
    return out


def _hypothetical(cs: CaseScore, g: str, passed: bool) -> CaseScore:
    sig = dict(cs.signals)
    if g == "draw":
        m, n = sig["m"] + int(passed), sig["k"] + 1
        sig.update(m=m, k=n)
        return replace(cs, signals=sig, survived=cs.answer is not None and m == n)
    if g == "chal":
        sig["chal"] = float(passed)
        return replace(cs, signals=sig)
    return replace(cs, probe_passes={**cs.probe_passes, g: passed})


def expected_move(model: ScoreModel, cs: CaseScore, mu: float, g: str, rates: dict) -> float:
    """E|mu after the call - mu now|, the pass probability being mu*q1 + (1-mu)*q0_hat."""
    r = rates.get(g)
    if r is None:
        return 0.0
    pi = mu * r["q1"] + (1 - mu) * r["q0_hat"]
    (mp, mf), _ = model.mu_sigma([_hypothetical(cs, g, True), _hypothetical(cs, g, False)])
    return pi * abs(mp - mu) + (1 - pi) * abs(mf - mu)


def frontier_batch(cds: Sequence[CaseData], model: ScoreModel, rates: dict, budget: float,
                   k: int, para: bool, k1: int = 1, z: float = 1.0, seed: int = 0) -> list[CaseScore]:
    """Run the frontier allocation on one batch; returns each case's final CaseScore (its
    `calls` are what that case cost, so the batch mean is <= budget)."""
    m = len(cds)
    left = int(math.floor(budget * m + 1e-9)) - m * (int(para) + k1)
    if left < 0:
        raise ValueError(f"budget {budget} below the initial {int(para) + k1} calls per case")
    states = [_State(k1) for _ in cds]
    cases = [state_case(cd, st, para) for cd, st in zip(cds, states)]
    mu, sd = model.mu_sigma(cases)
    tie = np.random.default_rng(seed).random(m)
    while left > 0:
        opts = [_options(cd, st, cs, para) for cd, st, cs in zip(cds, states, cases)]
        order = np.lexsort((tie, -mu))
        top, rest = order[:k], order[k:]
        h = [i for i in top if opts[i]]
        l = [i for i in rest if opts[i]]
        if not h and not l:
            break
        cand = []
        if h:
            cand.append(min(h, key=lambda i: mu[i] - z * sd[i]))
        if l:
            cand.append(max(l, key=lambda i: mu[i] + z * sd[i]))
        j = max(cand, key=lambda i: sd[i])
        g = max(opts[j], key=lambda g: expected_move(model, cases[j], mu[j], g, rates))
        st = states[j]
        if g == "draw":
            st.n += 1
        elif g == "chal":
            st.chal = True
        else:
            st.probes.append(g)
        cases[j] = state_case(cds[j], st, para)
        (mu[j],), (sd[j],) = model.mu_sigma([cases[j]])
        left -= 1
    return cases


# -- frozen on D_fit -------------------------------------------------------------------

@dataclass
class FrontierFit:
    model: ScoreModel
    rates: dict
    lam: float
    cv: dict
    para: bool
    n_states: int


def fit_frontier(fit: Sequence[CaseData], para: bool, k_frac: float, reps: int = 8,
                 ridge: float = 1.0, prior: Sequence[float] = (0.5, 0.5), lambdas=(0, 1, 2),
                 folds: int = 5, seed: int = 0, drop: Sequence[str] = ()) -> FrontierFit:
    """Score model for partial evidence: each fitting case contributes `reps` random
    states (draws 1..all, each probe and the challenge bought with probability 1/2), so the
    model has seen the uneven states the frontier produces. Pass rates come from each
    case's full state; lambda by CV with every case's states in one fold."""
    rng = np.random.default_rng(seed)
    rows, groups = [], []
    for cd in fit:
        pool = cd.fresh if para else cd.same
        for _ in range(reps):
            st = _State(int(rng.integers(1, len(pool) + 1)),
                        [g for g in FLIPS if rng.random() < 0.5], bool(rng.random() < 0.5))
            rows.append(state_case(cd, st, para))
            groups.append(cd.item_id)
    z = [c.correct for c in rows]
    full = [state_case(cd, _State(len(cd.fresh if para else cd.same), list(FLIPS), True), para)
            for cd in fit]
    model = fit_score_model(rows, z, ridge, prior, drop)
    model.cov = model.cov * reps                    # `reps` rows per case are not independent
    lam, cv = choose_lambda(rows, z, lambdas, k_frac, folds, ridge, prior, seed, groups, drop)
    return FrontierFit(model, pass_rates(full, [c.correct for c in full]), lam, cv, para, len(rows))


def run_frontier(coll: dict[str, CaseData], batches: list[list[str]], ff: FrontierFit,
                 budget: float, k: int, k1: int = 1, z: float = 1.0) -> dict:
    """Frontier allocation on every batch: prec@k by mu and by mu - lambda*sigma, calls,
    pooled AUROC/AUARC, and the calibration of mu on the final (uneven) evidence states."""
    per = {"mean": [], "pessimistic": []}
    calls, changed, mus, zs, ss = [], [], [], [], []
    for bi, b in enumerate(batches):
        cases = frontier_batch([coll[i] for i in b], ff.model, ff.rates, budget, k, ff.para, k1, z, bi)
        zz = np.array([c.correct for c in cases])
        mu, sd = ff.model.mu_sigma(cases)
        s = mu - ff.lam * sd
        per["mean"] = prec_rep(mu, zz, k / len(zz))        # one population: point + replicates
        per["pessimistic"] = prec_rep(s, zz, k / len(zz))
        top = lambda v: set(np.lexsort((np.arange(len(v)), -v))[:k])
        changed.append(top(mu) != top(s))
        calls += [c.calls for c in cases]
        mus += list(mu)
        ss += list(s)
        zs += list(zz)
    c = np.asarray(calls, float)
    return {"per_batch": per, "calls": float(c.mean()), "calls_sd": float(c.std()),
            "calls_min": int(c.min()), "calls_max": int(c.max()),
            "topk_changed_share": float(np.mean(changed)), "calibration": calibration(mus, zs),
            "ranking": {"mean": rank_metrics(mus, zs), "pessimistic": rank_metrics(ss, zs)}}
