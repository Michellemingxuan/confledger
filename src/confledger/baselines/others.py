"""Single-signal baselines: verbalized confidence, semantic entropy, BSDetector-style, random."""
from __future__ import annotations

from typing import Optional

from ..methods import CaseScore, _perm
from ..sprt_posterior import majority, miller_madow_entropy
from ..utility.records import CaseData


def semantic_entropy(cd: CaseData, b: int = 8, seed: Optional[int] = None) -> CaseScore:
    used = _perm(cd.same, seed, cd.item_id)[:b]
    ans, m, n = majority([d.answer for d in used])
    se = miller_madow_entropy([d.cluster for d in used])
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used) + 1, signals={"se": se},
                     key=(-se, m / n))


def verbalized(cd: CaseData, **_) -> CaseScore:
    ans = cd.verbal_answer
    conf = cd.verbal_conf if cd.verbal_conf is not None else 0.0
    return CaseScore(cd.item_id, ans, ans == cd.gold, 1, signals={"verb": conf}, key=(conf,))


def random_selection(cd: CaseData, **_) -> CaseScore:
    ans = cd.verbal_answer
    return CaseScore(cd.item_id, ans, ans == cd.gold, 1, key=(0,))


def bsdetector_like(cd: CaseData, b: int = 7, w: float = 0.7, seed: Optional[int] = None) -> CaseScore:
    """Approximation of BSDetector (Chen & Mueller): observed consistency blended
    with self-reflection certainty. Their self-reflection prompt differs from our
    verbal prompt; labelled 'BSDetector' in figures and tables; the paper text states it is a reconstruction from cached draws."""
    used = _perm(cd.same, seed, cd.item_id)[:b]
    ans, m, n = majority([d.answer for d in used])
    refl = (cd.verbal_conf or 0.0) / 100
    reflection_call = 0 if cd.meta.get("verbal_per_draw") else 1
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used) + reflection_call,
                     signals={"agree": m / n, "refl": refl}, key=(w * m / n + (1 - w) * refl,))
