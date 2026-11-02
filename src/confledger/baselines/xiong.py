"""Xiong et al. 2024 (ICLR), "Can LLMs Express Their Uncertainty?": black-box confidence
= prompting strategy + sampling + aggregation. Their recommended aggregation Avg-Conf:

    C = sum_i 1{Y_i = Y~} c_i / sum_i c_i,

c_i the verbal confidence of sample i and Y~ the answer under evaluation (the paper only
calls it "an associated answer"; here the majority answer, the one returned). Prompt: our
draw prompt with the confidence line (vanilla verbal elicitation plus a short explanation),
so it needs collect.verbal_per_draw. Sampling: self-random (same prompt, their default) or
'prompting' (one paraphrase per sample, + 1 generator call). M = b samples.
Not implemented (need new prompts): misleading hints, Top-K prompting and Pair-Rank (which
with two possible answers reduces to the Top-2 probabilities).
"""
from __future__ import annotations

from typing import Optional

from ..methods import PARAPHRASE_GEN_CALLS, CaseScore, _perm
from ..sprt_posterior import majority
from ..utility.records import CaseData


def xiong(cd: CaseData, b: int = 5, sampling: str = "self_random", seed: Optional[int] = None) -> CaseScore:
    para = sampling == "prompting"
    m_samples = b - PARAPHRASE_GEN_CALLS if para else b
    used = _perm(cd.fresh if para else cd.same, seed, cd.item_id)[:m_samples]
    ans, m, n = majority([d.answer for d in used])
    conf = [(d.answer, d.confidence / 100) for d in used if d.confidence is not None]
    tot = sum(c for _, c in conf)
    avg = sum(c for a, c in conf if a == ans) / tot if tot > 0 else m / n
    return CaseScore(cd.item_id, ans, ans == cd.gold, len(used) + PARAPHRASE_GEN_CALLS * para,
                     signals={"avg_conf": avg, "agree": m / n}, key=(avg, m / n))
