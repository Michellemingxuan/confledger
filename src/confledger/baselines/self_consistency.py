"""Self-consistency (Wang et al. 2023): b draws on the original prompt, majority answer,
confidence = agreement share m/b."""
from __future__ import annotations

from typing import Optional

from ..methods import CaseScore, _perm
from ..sprt_posterior import majority
from ..utility.records import CaseData


def self_consistency(cd: CaseData, b: int = 8, seed: Optional[int] = None) -> CaseScore:
    used = _perm(cd.same, seed, cd.item_id)[:b]
    ans, m, n = majority([d.answer for d in used])
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used), signals={"agree": m / n},
                     key=(m / n,))


def self_consistency_verbal(cd: CaseData, b: int = 8, seed: Optional[int] = None) -> CaseScore:
    """Self-consistency that also reads the confidence each draw carries (verbal_per_draw):
    rank by agreement, ties by mean confidence. Same b calls; the fair control when
    verbal confidence is free for every method."""
    used = _perm(cd.same, seed, cd.item_id)[:b]
    ans, m, n = majority([d.answer for d in used])
    confs = [d.confidence for d in used if d.confidence is not None]
    conf = sum(confs) / len(confs) if confs else 50.0
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used), signals={"agree": m / n, "verb": conf},
                     key=(m / n, conf))
