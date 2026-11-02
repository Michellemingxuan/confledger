"""Certified mode, two alternatives to the fixed Jeffreys certificate (decisions.md #16, #19).

Fitted prior (model-based). Each cell's stability prior Beta(a, b) is fitted by
beta-binomial maximum likelihood to the agreement counts of labelled-holdout cases
(D_fit and D_cal; draws only, no labels are read). The certification depth is the first
d whose posterior after d unanimous draws, Beta(a + d, b), puts at most eps0 below p0*.
The fit is noisy, so the depth used is a bootstrap upper quantile (pessimistic). The
certificate is then about stability, exact only if the Beta family describes the cell.
The Jeffreys depth of sprt_posterior.certification_depth stays the exact reference; no
constant there changes.

GLR stopping (exact, frequentist). Certify when p_hat > p0* and n * kl(p_hat, p0*) >=
log(1/delta) + log n (Track-and-Stop threshold, one arm; Garivier & Kaufmann 2016).
P(certify | pbar <= p0*) <= delta for every pbar and any stopping rule, at a price:
about 25 unanimous draws at delta = 0.1.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import beta as _beta

# -- fitted prior --------------------------------------------------------------------

def fit_beta_binomial(m: Sequence[int], n: int) -> tuple[float, float]:
    """MLE of Beta(a, b) from agreement counts m_i out of n draws each."""
    m = np.asarray(m, float)

    def nll(t):
        a, b = np.exp(np.clip(t, -7, 9))         # keeps the fit finite when every case is unanimous
        return -float(np.sum(betaln(a + m, b + n - m) - betaln(a, b)))

    mean = np.clip((m.mean() + 0.5) / (n + 1), 0.05, 0.95)
    res = minimize(nll, np.log([2 * mean, 2 * (1 - mean)]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000})
    a, b = np.exp(np.clip(res.x, -7, 9))
    return float(a), float(b)


def prior_depth(a: float, b: float, eps0: float = 0.05, p0_star: float = 0.8, d_max: int = 200) -> int:
    """First d >= 0 with P(pbar < p0* | d of d agree) <= eps0 under Beta(a, b)."""
    for d in range(d_max + 1):
        if _beta.cdf(p0_star, a + d, b) <= eps0:
            return d
    return d_max


def fitted_prior(m: Sequence[int], n: int, eps0: float = 0.05, p0_star: float = 0.8,
                 n_boot: int = 1000, quantile: float = 0.95, seed: int = 0) -> dict:
    """Fitted prior, its point depth, and the bootstrap `quantile` of the depth (the one
    to use). `share_zero` is the share of resamples that would certify with no draws."""
    m = np.asarray(m, int)
    a, b = fit_beta_binomial(m, n)
    rng = np.random.default_rng(seed)
    depths = np.array([prior_depth(*fit_beta_binomial(m[rng.integers(0, len(m), len(m))], n), eps0, p0_star)
                       for _ in range(n_boot)])
    return {"a": a, "b": b, "prior_mean": a / (a + b), "n_cases": int(len(m)), "draws_per_case": n,
            "depth_point": prior_depth(a, b, eps0, p0_star),
            "depth": int(np.ceil(np.quantile(depths, quantile))), "quantile": quantile,
            "depth_median": float(np.median(depths)), "share_zero": float(np.mean(depths == 0))}


def posterior_tail(m: int, n: int, a: float, b: float, p0_star: float = 0.8) -> float:
    """eps = P(pbar < p0* | m of n agree) under the prior Beta(a, b)."""
    return float(_beta.cdf(p0_star, a + m, b + n - m))


# -- GLR stopping ----------------------------------------------------------------------

def _kl(p: float, q: float) -> float:
    out = 0.0
    if p > 0:
        out += p * math.log(p / q)
    if p < 1:
        out += (1 - p) * math.log((1 - p) / (1 - q))
    return out


def glr_threshold(n: int, delta: float) -> float:
    return math.log(1 / delta) + math.log(n)


def glr_decision(m: int, n: int, p0_star: float = 0.8, delta: float = 0.1) -> str:
    """'certify', 'reject' or 'continue' after m of n draws agree."""
    if n == 0:
        return "continue"
    p = m / n
    if n * _kl(p, p0_star) < glr_threshold(n, delta):
        return "continue"
    return "certify" if p > p0_star else "reject"


def glr_depth(delta: float = 0.1, p0_star: float = 0.8, disagreements: int = 0, n_max: int = 1000) -> int:
    """Draws needed to certify with `disagreements` disagreements and every other draw agreeing."""
    for n in range(disagreements + 1, n_max + 1):
        if glr_decision(n - disagreements, n, p0_star, delta) == "certify":
            return n
    return n_max
