"""Evaluation metrics (Sec. 4.5). The target of every confidence metric is the
correctness indicator Z = 1{Yhat = Y}, never the label Y itself."""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from scipy import stats as st


def horizontal_gap(curve: dict, target: str = "cel:cel", baseline: str = "sc:") -> float:
    """Calls the baseline needs to match the target's prec@k, as a multiple of the
    target's calls (linear interpolation along the baseline's budget curve; inf if it
    never matches)."""
    t = curve[target]
    xs, ys = zip(*sorted((v["calls"], v["prec"]) for k, v in curve.items() if k.startswith(baseline)))
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 < t["prec"] <= y1:
            return (x0 + (t["prec"] - y0) * (x1 - x0) / (y1 - y0)) / t["calls"]
    return float("inf") if max(ys) < t["prec"] else xs[0] / t["calls"]


# -- selection quality --------------------------------------------------------

def precision_at_k(z_selected: Sequence[bool]) -> float:
    return float(np.mean(z_selected)) if len(z_selected) else float("nan")


def hypergeom_expected_precision(K: int, N: int, k: int) -> float:
    """Exact expected precision@k of random selection from a batch with K correct."""
    return K / N


def hypergeom_pvalue(x: int, K: int, N: int, k: int) -> float:
    """P(X >= x) under random selection: exact null for 'beats random'."""
    return float(st.hypergeom.sf(x - 1, N, K, k))


def jeffreys_interval(x: int, n: int, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    lo = 0.0 if x == 0 else st.beta.ppf(a, x + 0.5, n - x + 0.5)
    hi = 1.0 if x == n else st.beta.ppf(1 - a, x + 0.5, n - x + 0.5)
    return float(lo), float(hi)


# -- global discrimination -------------------------------------------------------

def auroc(scores: Sequence[float], z: Sequence[bool]) -> float:
    """P(score_correct > score_incorrect) + 0.5 P(tie)  (Mann-Whitney, mid-ranks)."""
    s, z = np.asarray(scores, float), np.asarray(z, bool)
    n1, n0 = z.sum(), (~z).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = st.rankdata(s)
    return float((r[z].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _expected_cumcorrect(s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Expected #correct among the top-n (n = 1..N), most confident first, with
    ties broken uniformly at random -- computed exactly, group by group."""
    out, done, correct = [], 0, 0.0
    for v in np.unique(s)[::-1]:
        grp = z[s == v]
        g, c = len(grp), grp.sum()
        for t in range(1, g + 1):
            out.append(correct + t * c / g)
        correct += c
        done += g
    return np.asarray(out)


def prr(scores: Sequence[float], z: Sequence[bool]) -> float:
    """Prediction Rejection Ratio: (AUC_s - AUC_rand) / (AUC_oracle - AUC_rand),
    AUC = mean accuracy of the retained set over all coverage levels; ties exact."""
    s, z = np.asarray(scores, float), np.asarray(z, float)
    n = np.arange(1, len(s) + 1)
    auc_s = float((_expected_cumcorrect(s, z) / n).mean())
    auc_o = float((np.cumsum(np.sort(z)[::-1]) / n).mean())
    auc_r = float(z.mean())
    return (auc_s - auc_r) / (auc_o - auc_r) if auc_o > auc_r else float("nan")


def risk_coverage(scores: Sequence[float], z: Sequence[bool], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    s, z = np.asarray(scores, float), np.asarray(z, float)
    order = np.lexsort((np.random.default_rng(seed).random(len(s)), -s))
    n = np.arange(1, len(s) + 1)
    return n / len(s), 1 - np.cumsum(z[order]) / n


# -- paired comparisons across batches -----------------------------------------------

def paired_compare(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                   seed: int = 0) -> dict:
    """Estimation first: mean difference + bootstrap CI over batches; Wilcoxon as
    corroboration."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d)), replace=True).mean(axis=1)
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(st.wilcoxon(d, zero_method="zsplit", alternative="two-sided").pvalue)
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975)), "wilcoxon_p": p,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "n": len(d)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# -- monotonicity (Assumption 1) ----------------------------------------------------

def strata_accuracy(scores: Sequence[float], z: Sequence[bool], n_strata: int = 10,
                    seed: int = 0) -> list[dict]:
    """Equal-count rank strata (most confident first) with Jeffreys intervals."""
    s, z = np.asarray(scores, float), np.asarray(z, bool)
    order = np.lexsort((np.random.default_rng(seed).random(len(s)), -s))
    out = []
    for i, chunk in enumerate(np.array_split(order, n_strata)):
        x, n = int(z[chunk].sum()), len(chunk)
        lo, hi = jeffreys_interval(x, n)
        out.append({"stratum": i + 1, "n": n, "acc": x / n, "lo": lo, "hi": hi})
    return out


def pav(y: Sequence[float]) -> np.ndarray:
    """Pool-adjacent-violators: best NON-INCREASING fit in rank order
    (rank 1 = most confident, so accuracy should not increase with rank)."""
    blocks = [[float(v), 1] for v in y]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:          # violation of non-increasing
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1])
            w = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [v / w, w]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[v] * w for v, w in blocks])


def isotonic_gap(scores: Sequence[float], z: Sequence[bool], n_strata: int = 10,
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Brier cost of imposing monotonicity on stratum accuracies (0 = perfectly
    monotone); bootstrap CI over items."""
    s, z = np.asarray(scores, float), np.asarray(z, float)
    rng = np.random.default_rng(seed)

    def gap(idx):
        acc = np.array([d["acc"] for d in strata_accuracy(s[idx], z[idx] > 0.5, n_strata)])
        return float(((pav(acc) - acc) ** 2).mean())

    g = gap(np.arange(len(s)))
    boots = [gap(rng.integers(0, len(s), len(s))) for _ in range(n_boot)]
    return {"gap": g, "ci_lo": float(np.quantile(boots, 0.025)),
            "ci_hi": float(np.quantile(boots, 0.975))}


def top_strata_test(rank_positions: Sequence[int], z: Sequence[bool],
                    bounds=((1, 10), (11, 30), (31, 50))) -> dict:
    """Local monotonicity where Prop. 1 consumes it: per-batch rank strata,
    pooled over batches; one-sided Mann-Whitney between adjacent strata."""
    r, z = np.asarray(rank_positions), np.asarray(z, float)
    groups = {f"{a}-{b}": z[(r >= a) & (r <= b)] for a, b in bounds}
    out = {k: {"n": int(len(v)), "acc": float(v.mean()) if len(v) else float("nan")}
           for k, v in groups.items()}
    keys = list(groups)
    for hi, lo in zip(keys, keys[1:]):
        if len(groups[hi]) and len(groups[lo]):
            out[f"p_{hi}>{lo}"] = float(st.mannwhitneyu(groups[hi], groups[lo],
                                                        alternative="greater").pvalue)
    return out


# -- robustness ----------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def tie_aware_consistency(selected_b: set, top_group_a: set, k: int) -> float:
    """Fraction of run-B selections inside run A's top tie-group (every case scored
    at or above A's k-th selection). Unlike Jaccard, random tie-breaking among
    equally good cases does not count as instability."""
    return len(selected_b & top_group_a) / k


def jaccard_chance_floor(N: int = 50, k: int = 10) -> float:
    """E[J] for two independent uniform k-of-N selections (overlap ~ hypergeometric)."""
    xs = np.arange(0, k + 1)
    pmf = st.hypergeom.pmf(xs, N, k, k)
    return float(np.sum(pmf * xs / (2 * k - xs)))


def heldout_audit(selected: Sequence[bool], controls: Sequence[bool]) -> dict:
    """Held-out probe survival, selected (ranks 1-10) vs matched controls (11-30);
    one-sided Fisher exact test."""
    a, b = [x for x in selected if x is not None], [x for x in controls if x is not None]
    table = [[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]]
    p = float(st.fisher_exact(table, alternative="greater").pvalue) if a and b else float("nan")
    return {"selected_pass": sum(a) / len(a) if a else float("nan"),
            "control_pass": sum(b) / len(b) if b else float("nan"), "fisher_p": p}


# -- empirical exchange rate (Prop. 3) --------------------------------------------------

def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 100) -> np.ndarray:
    """Ridge-regularised logistic regression by Newton-Raphson; X includes intercept."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def empirical_kappa(agree: Sequence[int], passes: dict[str, Sequence[float]],
                    z: Sequence[bool], n_boot: int = 500, seed: int = 0) -> dict:
    """kappa_hat_g = beta_g / beta_agree from logit P(Z=1) = b0 + b_m m + sum b_g C_g."""
    names = list(passes)
    X = np.column_stack([np.ones(len(agree)), np.asarray(agree, float)] +
                        [np.asarray(passes[g], float) for g in names])
    y = np.asarray(z, float)

    def fit(idx):
        w = logistic_fit(X[idx], y[idx])
        return {g: w[2 + i] / w[1] if abs(w[1]) > 1e-6 else float("nan")
                for i, g in enumerate(names)} | {"beta_agree": w[1]}

    est = fit(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = [fit(rng.integers(0, len(y), len(y))) for _ in range(n_boot)]
    ci = {g: (float(np.nanquantile([b[g] for b in boots], 0.025)),
              float(np.nanquantile([b[g] for b in boots], 0.975))) for g in names}
    return {"kappa_hat": {g: float(est[g]) for g in names}, "ci": ci,
            "beta_agree": float(est["beta_agree"])}
