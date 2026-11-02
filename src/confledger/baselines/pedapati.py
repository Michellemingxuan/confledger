"""Pedapati et al. 2024 ("LLM Confidence Estimation via Black-Box Access"), ported to our
signals: for each perturbation, (i) the number of semantic sets among its answers and (ii)
the mean pairwise ROUGE-L of the full responses (answer + explanation); a logistic
regression of Z on these features, fitted on D_fit.

Perturbations replayed from the collection: stochastic decoding (same-prompt draws) and
paraphrasing (one paraphrase per draw, + 1 generator call). Budget b: 1 generator call,
ceil((b-1)/2) decoding and floor((b-1)/2) paraphrase draws; below b = 5 decoding only.
On yes/no answers the semantic-set count is 1 or 2 (all agree or not) and ROUGE-L over
answers alone would be a function of the agreement count; over full responses it also
reads how similar the explanations are. Not implemented (need new prompts / a local NLI
model): sentence permutation, entity-frequency amplification, stop-word removal,
split-response consistency. Reported as "black-box calibrator (Pedapati et al.)"; the paper states what it omits.
"""
from __future__ import annotations

import math
import re
from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from .. import metrics as M
from ..methods import PARAPHRASE_GEN_CALLS, CaseScore, _perm
from ..sprt_posterior import majority
from ..utility.records import CaseData, Draw


def rouge_l(a: str, b: str) -> float:
    """ROUGE-L F1 on lower-cased word tokens."""
    x, y = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    if not x or not y:
        return float(x == y)
    prev = [0] * (len(y) + 1)
    for xi in x:
        cur = [0]
        for j, yj in enumerate(y):
            cur.append(prev[j] + 1 if xi == yj else max(prev[j + 1], cur[j]))
        prev = cur
    lcs = prev[-1]
    return 0.0 if lcs == 0 else 2 * lcs / (len(x) + len(y))


def _block(draws: Sequence[Draw]) -> list[float]:
    """[number of semantic sets, mean pairwise ROUGE-L]; a failed extraction is its own set."""
    sets = len({d.answer if d.answer is not None else f"fail{i}" for i, d in enumerate(draws)})
    text = [f"{d.answer or ''} {d.explanation}" for d in draws]
    pairs = list(combinations(range(len(draws)), 2))
    sim = float(np.mean([rouge_l(text[i], text[j]) for i, j in pairs])) if pairs else 1.0
    return [float(sets), sim]


def split(b: int) -> tuple[int, int]:
    if b < 5:
        return b, 0
    return math.ceil((b - PARAPHRASE_GEN_CALLS) / 2), (b - PARAPHRASE_GEN_CALLS) // 2


def features(cd: CaseData, b: int, seed: Optional[int] = None) -> tuple[list[float], list[Draw]]:
    n_sd, n_pp = split(b)
    sd = _perm(cd.same, seed, cd.item_id)[:n_sd]
    pp = _perm(cd.fresh, seed, cd.item_id)[:n_pp]
    return _block(sd) + (_block(pp) if n_pp else []), sd + pp


def fit_pedapati(fit: Sequence[CaseData], b: int, l2: float = 1.0) -> dict:
    """Logistic regression on standardised features (one L2 penalty, intercept free)."""
    X = np.array([features(cd, b)[0] for cd in fit])
    z = np.array([majority([d.answer for d in features(cd, b)[1]])[0] == cd.gold for cd in fit], float)
    if z.all() or not z.any():
        return {"note": "D_fit has one class only; no regression", "b": b}
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = M.logistic_fit(Xs, z, l2=np.r_[0.0, np.full(X.shape[1], l2)])
    return {"coef": w.tolist(), "mean": mu.tolist(), "sd": sd.tolist(), "b": b}


def pedapati(cd: CaseData, fit: dict, seed: Optional[int] = None) -> CaseScore:
    x, used = features(cd, fit["b"], seed)
    ans, m, n = majority([d.answer for d in used])
    xs = (np.asarray(x) - fit["mean"]) / fit["sd"]
    p = float(1 / (1 + np.exp(-(fit["coef"][0] + xs @ np.asarray(fit["coef"][1:])))))
    n_pp = split(fit["b"])[1]
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used) + PARAPHRASE_GEN_CALLS * (n_pp > 0),
                     signals={"p_lr": p, "agree": m / n}, key=(p, m / n))
