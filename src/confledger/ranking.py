"""Ranking: cascade keys, Borda aggregation, total order, top-k selection (Sec. 3.5).

Conventions (Sec. 4.5, stated in the paper):
  * stage-1-eliminated cases rank below every survivor, ordered by stage-1
    agreement; ties broken randomly (seeded) for selection, mid-rank for AUROC;
  * cascade strata produce genuine ties, scored as equal.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
from scipy.stats import rankdata

from .methods import CaseScore

# orientation: +1 larger is better, -1 smaller is better
SIGNAL_ORIENT = {"L": 1, "flip": 1, "se": -1, "chal": 1, "verb": 1, "llr": 1}


def key_scores(cases: Sequence[CaseScore]) -> np.ndarray:
    """Dense numeric score from lexicographic keys (equal keys -> equal score)."""
    uniq = sorted({c.key for c in cases})
    pos = {k: i for i, k in enumerate(uniq)}
    return np.array([pos[c.key] for c in cases], dtype=float)


def borda_scores(cases: Sequence[CaseScore], signals: Sequence[str]) -> np.ndarray:
    """Average-rank (Borda) over the given signals among survivors; eliminated
    cases below all survivors, ordered by stage-1 agreement. Empty signal set ->
    all survivors tie (coalition V(empty) is handled by the caller)."""
    surv = [i for i, c in enumerate(cases) if c.survived]
    out = np.zeros(len(cases))
    for i, c in enumerate(cases):
        if not c.survived:
            out[i] = c.s1_agree / 10.0          # in [0, 0.3): strictly below survivors
    if not surv:
        return out
    sigs = [s for s in signals if all(s in cases[i].signals for i in surv)]
    if not sigs:
        out[surv] = 1.0
        return out
    ranks = np.zeros(len(surv))
    for s in sigs:
        vals = np.array([SIGNAL_ORIENT[s] * cases[i].signals[s] for i in surv])
        ranks += rankdata(vals, method="average") / len(surv)
    out[surv] = 1.0 + ranks / len(sigs)
    return out


def select_top_k(scores: np.ndarray, k: int, rng: random.Random) -> list[int]:
    """Top-k indices; ties at the boundary broken uniformly at random."""
    jitter = np.array([rng.random() for _ in range(len(scores))])
    order = np.lexsort((-jitter, -scores))        # primary: score desc; secondary: jitter
    return list(order[:k])


def ranks_within(scores: np.ndarray, rng: Optional[random.Random] = None) -> np.ndarray:
    """1-based rank positions (1 = most confident), random tie-breaking."""
    rng = rng or random.Random(0)
    order = select_top_k(scores, len(scores), rng)
    r = np.empty(len(scores), dtype=int)
    r[order] = np.arange(1, len(scores) + 1)
    return r
