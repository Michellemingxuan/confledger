"""Exact Shapley valuation of confidence signals (Sec. 3.8, Eq. 13).

Players are the signals; V(C) is supplied by the caller (batch-averaged
precision@10 of Borda-over-C). With 5 players there are 32 coalitions, so the
computation is exact -- no permutation sampling.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Sequence


def all_coalitions(players: Sequence[str]) -> list[frozenset]:
    return [frozenset(c) for r in range(len(players) + 1) for c in combinations(players, r)]


def exact_shapley(players: Sequence[str], value: Callable[[frozenset], float]) -> dict:
    n = len(players)
    V = {C: value(C) for C in all_coalitions(players)}
    phi = {}
    for j in players:
        others = [p for p in players if p != j]
        total = 0.0
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for C in combinations(others, r):
                C = frozenset(C)
                total += w * (V[C | {j}] - V[C])
        phi[j] = total
    grand, empty = V[frozenset(players)], V[frozenset()]
    return {"phi": phi, "V": {",".join(sorted(C)) or "{}": v for C, v in V.items()},
            "efficiency_check": sum(phi.values()) - (grand - empty),
            "total_gain": grand - empty}
