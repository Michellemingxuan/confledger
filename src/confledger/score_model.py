"""Correctness score fitted on D_fit, and the pessimistic score (decisions.md #16).

A CEL-k allocation is replayed on the labelled fitting cases; a logistic model of
Z = 1{Yhat = Y} is fitted on their signals and frozen. For a new case it returns
mu = P(Z = 1 | evidence) and sigma, the standard deviation of that estimate, from three
sources: the fitted coefficients (parameter uncertainty, delta method), the case's own
stability evidence (the log-odds of its agreement rate under a Beta posterior, whose
variance shrinks as 1/n), and the signals the case did not buy (filled with their D_fit
mean, each adding beta_g^2 * Var(x_g)). So a case with fewer calls gets a wider sigma. The pessimistic score s = mu - lambda * sigma ranks
cases with thin evidence lower; lambda is chosen by K-fold CV on D_fit, lambda = 0 is
ranking by mu.

Model-based throughout: no guarantee comes from here. With equal evidence per case,
sigma is (nearly) a function of mu and pessimism cannot reorder the top-k; it helps
only when evidence is uneven (decisions.md #17). Guarantees come from
certified_selection.py, applied on top of the frozen score.

Labels enter only as `z` of the fitting cases; scoring a case reads only its signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import digamma, polygamma

from .methods import CaseScore
from .metrics import logistic_fit

PROBES = ("neg", "relabel", "distractor")
FEATURES = ("intercept", "eta", "survived", *PROBES, "chal", "verb", "se",
            "length", "hedge", "neg_drop", "p_first")
GRAY_BOX = ("p_first",)                      # needs logprobs: excluded unless scoring.gray_box
UNPENALISED = ("intercept", "eta")          # stability enters unpenalised (decisions.md #16)
ETA = FEATURES.index("eta")                  # its coefficient is kept >= 0 (decisions.md #34)


def _code(v) -> float:
    """Binary call outcome: pass +1, fail -1, not bought / unparsed NaN (filled later)."""
    return np.nan if v is None else (1.0 if v else -1.0)


def features(c: CaseScore, prior: Sequence[float] = (0.5, 0.5)) -> tuple[np.ndarray, float]:
    """Feature row of one case (NaN = signal not bought) and Var[logit p | its draws]."""
    a, b = prior
    m, n = c.signals.get("m", 0), c.signals.get("k", 0)
    eta = digamma(a + m) - digamma(b + n - m)              # E[logit p | m of n draws agree]
    var = polygamma(1, a + m) + polygamma(1, b + n - m)    # Var[logit p | ...], O(1/n)
    s = c.signals
    x = [1.0, eta, float(c.survived), *(_code(c.probe_passes.get(g)) for g in PROBES),
         _code(bool(s["chal"])) if "chal" in s else np.nan,
         (s["verb"] - 50) / 50 if "verb" in s else np.nan,
         -s["se"] if "se" in s else np.nan,
         math.log1p(s["length"]) if "length" in s else np.nan,
         s["hedge"] if "hedge" in s else np.nan,
         s["neg_drop"] / 50 if "neg_drop" in s else np.nan,
         2 * s["p_first"] - 1 if "p_first" in s else np.nan]
    return np.asarray(x), float(var)


def _design(cases: Sequence[CaseScore], prior) -> tuple[np.ndarray, np.ndarray]:
    rows = [features(c, prior) for c in cases]
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


def _fill(X: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    miss = np.isnan(X)
    return np.where(miss, mean, X), miss


@dataclass
class ScoreModel:
    coef: np.ndarray
    cov: np.ndarray
    prior: tuple
    mean: np.ndarray          # D_fit mean of each feature where bought (fills unbought ones)
    var: np.ndarray           # D_fit variance of each feature where bought

    def mu_sigma(self, cases: Sequence[CaseScore]) -> tuple[np.ndarray, np.ndarray]:
        """mu and sigma; sigma^2 (log-odds) = fit + stability + unbought-signal variance."""
        X, var_eta = _design(cases, self.prior)
        X, miss = _fill(X, self.mean)
        eta = X @ self.coef
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        var_logit = (np.einsum("ij,jk,ik->i", X, self.cov, X) + self.coef[1] ** 2 * var_eta
                     + (miss * self.coef ** 2 * self.var).sum(axis=1))
        return mu, mu * (1 - mu) * np.sqrt(var_logit)

    def score(self, cases: Sequence[CaseScore], lam: float) -> np.ndarray:
        mu, sd = self.mu_sigma(cases)
        return mu - lam * sd

    def summary(self) -> dict:
        return {"coef": dict(zip(FEATURES, map(float, self.coef))),
                "se": dict(zip(FEATURES, map(float, np.sqrt(np.diag(self.cov)))))}


def fit_score_model(cases: Sequence[CaseScore], z: Sequence[bool], ridge: float = 1.0,
                    prior: Sequence[float] = (0.5, 0.5), drop: Sequence[str] = ()) -> ScoreModel:
    """Logistic fit of Z on the fitting cases' features: ridge on every signal but the
    intercept and the stability term; covariance = inverse penalised Hessian. Unbought
    signals are filled with their D_fit mean (0, variance 0, if never bought). Features in
    `drop` (pruned, signal_value.py) are held at zero weight. More agreement may not lower
    confidence: a negative stability coefficient is refitted at its bound, 0, which is the
    constrained optimum because the objective is convex (decisions.md #34)."""
    X, _ = _design(cases, prior)
    seen = ~np.isnan(X)
    n = seen.sum(axis=0)
    mean = np.where(n > 0, np.nansum(X, axis=0) / np.maximum(n, 1), 0.0)
    var = np.where(n > 1, np.nansum((X - mean) ** 2, axis=0) / np.maximum(n - 1, 1), 0.0)
    X, _ = _fill(X, mean)
    y = np.asarray(z, float)
    pen = np.array([1e8 if f in drop else 1e-6 if f in UNPENALISED else ridge for f in FEATURES])
    w = logistic_fit(X, y, pen)
    if w[ETA] < 0:
        pen[ETA] = 1e8
        w = logistic_fit(X, y, pen)
    p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
    H = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
    return ScoreModel(w, np.linalg.pinv(H), tuple(prior), mean, var)


def top_precision(s: np.ndarray, z: np.ndarray, k: int, n_ties: int = 20) -> float:
    """Precision of the top-k by s, averaged over random tie-breaks."""
    rng = np.random.default_rng(0)
    return float(np.mean([z[np.lexsort((rng.random(len(s)), -s))[:k]].mean() for _ in range(n_ties)]))


def choose_lambda(cases: Sequence[CaseScore], z: Sequence[bool], grid: Sequence[float],
                  top_frac: float, folds: int = 5, ridge: float = 1.0,
                  prior: Sequence[float] = (0.5, 0.5), seed: int = 0,
                  groups: Sequence | None = None, drop: Sequence[str] = ()) -> tuple[float, dict]:
    """lambda by K-fold CV on D_fit: held-out (mu, sigma) from the other folds' model,
    pooled; criterion = precision of the top `top_frac` share (the budget-mode target).
    Ties go to the smaller lambda, so lambda = 0 unless pessimism helps on D_fit.
    `groups` (e.g. item ids) keeps every row of one case in the same fold."""
    z = np.asarray(z, bool)
    g = np.asarray(groups if groups is not None else np.arange(len(cases)))
    uniq = np.random.default_rng(seed).permutation(np.unique(g))
    fold = dict(zip(uniq, np.arange(len(uniq)) % folds))
    fold_of = np.array([fold[x] for x in g])
    mu, sd = np.empty(len(cases)), np.empty(len(cases))
    for f in range(folds):
        test = np.nonzero(fold_of == f)[0]
        train = np.nonzero(fold_of != f)[0]
        m = fit_score_model([cases[i] for i in train], z[train], ridge, prior, drop)
        mu[test], sd[test] = m.mu_sigma([cases[i] for i in test])
    k = max(1, math.ceil(top_frac * len(cases)))
    table = {float(l): top_precision(mu - l * sd, z, k) for l in grid}
    best = max(sorted(table), key=lambda l: (round(table[l], 9), -l))
    return best, table
