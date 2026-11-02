"""Certified mode (docs/problem_setup.md): deliver only answers whose error rate is controlled.

Two routes, each applied to one test batch at a time:

  conformal_bh  distribution-free, EXACT (Jin & Candès 2023): conformal p-values from the
                labelled calibration set D_cal, then Benjamini-Hochberg. E[FDP] <= alpha for
                ANY score frozen before D_cal is touched, if calibration and test cases are
                exchangeable. A weak score costs deliveries, not errors.
  ebh           MODEL-BASED (methodology.md §4b-5): per-case evidence-ledger e-values with
                likelihood ratios from pass rates estimated on D_fit (q0 = an upper confidence
                bound, never below the theory anchor), then e-BH (Wang & Ramdas 2022).
                E[FDP] <= alpha only if an incorrect answer passes each call with conditional
                probability <= its q0. Draws of a stably wrong answer violate this (the
                separation theorem's type B), so this route is reported, not relied on.

Selection reads only scores / e-values of test cases; labels enter only through the
calibration and fitting cases.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import beta as _beta

from .methods import CaseScore
from .sprt_posterior import DEFAULT_Q0

LEDGER_SIGNALS = ("draw", "neg", "relabel", "distractor", "chal")


# -- distribution-free route -------------------------------------------------------

def conformal_pvalues(s_test: np.ndarray, s_cal: np.ndarray, z_cal: np.ndarray) -> np.ndarray:
    """p_j = (1 + #{i in D_cal: Z_i = 0, s_i >= s_j}) / (n_cal + 1)  (Jin & Candès 2023,
    deterministic form: conservative under ties)."""
    bad = np.sort(np.asarray(s_cal, float)[~np.asarray(z_cal, bool)])
    n_ge = len(bad) - np.searchsorted(bad, np.asarray(s_test, float), side="left")
    return (1 + n_ge) / (len(s_cal) + 1)


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg: indices of the rejected nulls (here, the delivered cases)."""
    m = len(p)
    order = np.argsort(p, kind="stable")
    ok = np.nonzero(p[order] <= alpha * np.arange(1, m + 1) / m)[0]
    return order[:ok[-1] + 1] if len(ok) else order[:0]


# -- model-based route -------------------------------------------------------------

def _outcomes(c: CaseScore) -> dict[str, tuple[int, int]]:
    """(passes, fails) per ledger signal: draws agreeing with the majority answer, probes
    passed, challenge held. Unbought or unparsed calls are absent."""
    m, n = c.signals.get("m", 0), c.signals.get("k", 0)
    out = {"draw": (m, n - m)}
    for g in ("neg", "relabel", "distractor"):
        v = c.probe_passes.get(g)
        if v is not None:
            out[g] = (int(v), int(not v))
    if "chal" in c.signals:
        out["chal"] = (int(bool(c.signals["chal"])), int(not c.signals["chal"]))
    return out


def pass_rates(cases: Sequence[CaseScore], z: Sequence[bool], level: float = 0.95) -> dict:
    """Per ledger signal, on D_fit: q0 = Jeffreys upper bound (at `level`) of the pass rate
    among incorrect answers, raised to the theory anchor where one exists (methodology.md
    §5b, safeguard 1); q1 = pass rate among correct answers; q0_hat = the point estimate
    (for prediction, never for validity). Draws are pooled over cases,
    so q0['draw'] ignores within-case correlation (model-based)."""
    tot = {g: [[0, 0], [0, 0]] for g in LEDGER_SIGNALS}       # [incorrect, correct] x [pass, fail]
    for c, zi in zip(cases, z):
        for g, (p, f) in _outcomes(c).items():
            tot[g][int(zi)][0] += p
            tot[g][int(zi)][1] += f
    out = {}
    for g, ((p0, f0), (p1, f1)) in tot.items():
        if p0 + f0 == 0 or p1 + f1 == 0:
            continue
        q0 = float(_beta.ppf(level, p0 + 0.5, f0 + 0.5))
        q0 = max(q0, DEFAULT_Q0.get(g, 0.0))
        q1 = (p1 + 0.5) / (p1 + f1 + 1)
        out[g] = {"q0": q0, "q1": q1, "q0_hat": (p0 + 0.5) / (p0 + f0 + 1), "n0": p0 + f0, "n1": p1 + f1}
    return out


def evalue(c: CaseScore, rates: dict) -> float:
    """Ledger e-value: product of likelihood ratios q1/q0 (pass), (1-q1)/(1-q0) (fail).
    A signal whose q1 <= q0 is dropped (ratio 1). Under H0 with pass rate <= q0 each
    factor has mean <= 1, so the product is an e-value."""
    log_e = 0.0
    for g, (p, f) in _outcomes(c).items():
        r = rates.get(g)
        if r is None or r["q1"] <= r["q0"] or r["q0"] >= 1:
            continue
        log_e += p * np.log(r["q1"] / r["q0"]) + f * np.log((1 - r["q1"]) / (1 - r["q0"]))
    return float(np.exp(log_e))


def ebh(e: np.ndarray, alpha: float) -> np.ndarray:
    """e-BH: the r largest e-values, r = max{r : e_(r) >= m / (alpha r)}."""
    m = len(e)
    order = np.argsort(-np.asarray(e, float), kind="stable")
    ok = np.nonzero(e[order] >= m / (alpha * np.arange(1, m + 1)))[0]
    return order[:ok[-1] + 1] if len(ok) else order[:0]


# -- outcome of one delivered set ------------------------------------------------------

def selection_outcome(sel: np.ndarray, z: np.ndarray) -> dict:
    """FDP, power and size of a delivered set (metrics only: reads labels)."""
    z = np.asarray(z, bool)
    hit = int(z[sel].sum()) if len(sel) else 0
    return {"fdp": (len(sel) - hit) / max(len(sel), 1), "power": hit / max(int(z.sum()), 1),
            "size": int(len(sel))}
