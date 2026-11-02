"""CISC, Taubenfeld et al. 2025 ("Confidence Improves Self-Consistency in LLMs"):

    a_hat = argmax_a sum_i 1{a_i = a} w_i,   w_i = exp(c_i / T) / sum_j exp(c_j / T),

over b samples on the original prompt. T -> inf is self-consistency; T is tuned on held-out
data over 80 log-spaced values in [1e-4, 1e4] for the vote's accuracy (here on D_fit; ties go
to the larger T, i.e. towards self-consistency). CISC defines the answer, not a selection
score; the score here is the winning answer's weight share sum_{a_i = a_hat} w_i (our
adaptation, stated as such). Because the vote can differ from the majority, CISC's own
accuracy can differ from self-consistency's; each method is scored on its own Z.

Confidence sources: 'verbal' (0-100 per draw, black-box) or 'p_answer' (the answer
token's probability from logprobs, a one-token stand-in for their length-normalised response
probability; gray-box). Their best source, P(True), needs a new prompt with logprobs.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..methods import CaseScore, _perm
from ..utility.records import YES, CaseData, Draw

T_GRID = np.logspace(-4, 4, 80)


def draw_conf(d: Draw, source: str) -> Optional[float]:
    if source == "verbal":
        return None if d.confidence is None else d.confidence / 100
    if d.p_yes is None or d.answer is None:
        return None
    return d.p_yes if d.answer == YES else 1 - d.p_yes


def vote(draws: Sequence[Draw], T: float, source: str) -> tuple[Optional[str], float]:
    """Weighted vote; draws with no answer drop out, a missing confidence counts as 0.5."""
    ok = [d for d in draws if d.answer is not None]
    if not ok:
        return None, 0.0
    c = np.array([draw_conf(d, source) if draw_conf(d, source) is not None else 0.5 for d in ok])
    w = np.exp((c - c.max()) / T)
    w /= w.sum()
    tally: dict = {}
    for d, wi in zip(ok, w):
        tally[d.answer] = tally.get(d.answer, 0.0) + wi
    ans = max(sorted(tally), key=tally.get)                      # sorted: deterministic ties
    return ans, float(tally[ans]) * len(ok) / len(draws)        # failures count against the share


def cisc(cd: CaseData, T: float, b: int = 8, source: str = "verbal", seed: Optional[int] = None) -> CaseScore:
    used = _perm(cd.same, seed, cd.item_id)[:b]
    ans, share = vote(used, T, source)
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used), signals={"cisc": share}, key=(share,))


def fit_cisc(fit: Sequence[CaseData], b: int, source: str = "verbal") -> dict:
    """T maximising the vote's accuracy on D_fit (reads D_fit labels only)."""
    acc = [np.mean([vote(cd.same[:b], T, source)[0] == cd.gold for cd in fit]) for T in T_GRID]
    best = max(range(len(T_GRID)), key=lambda i: (round(acc[i], 12), i))
    return {"T": float(T_GRID[best]), "fit_acc": float(acc[best]),
            "sc_acc": float(acc[-1]), "source": source, "b": b}
