"""Response parsing. Failures return None and are counted, never silently coerced.

Audit step (Sec. 4): hand-check ~50 random extractions per model on the dev
slice before scaling -- extraction noise masquerades as model instability.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .records import NO, YES

_ANS = re.compile(r"answer\s*[:\-]\s*[*_`\"']*\s*(yes|no|a|b)\b", re.I)
_EXPL = re.compile(r"explanation\s*[:\-]\s*(.+)", re.I | re.S)
_CONF = re.compile(r"confidence\s*[:\-]\s*[*_`]*\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.I)
_BARE = re.compile(r"^\s*[*_`\"']*\s*(yes|no)\b", re.I)
_PYES = re.compile(r"\n?\[\[p_yes: ([0-9.]+)\]\]\s*$")
_HEDGES = re.compile(r"\b(might|may|maybe|perhaps|possibly|probably|likely|unlikely|not sure|"
                     r"unsure|uncertain|unclear|i think|i believe|it depends|could be|seems?|"
                     r"appears?|approximately|roughly|arguably)\b", re.I)


def parse_answer(text: str, fmt: str = "yn") -> Optional[str]:
    """Map a response to 'yes'/'no'. fmt='ab' uses the relabel mapping A->no, B->yes."""
    if not text:
        return None
    m = _ANS.search(text)
    tok = m.group(1).lower() if m else None
    if tok is None and fmt == "yn":
        b = _BARE.search(text)
        tok = b.group(1).lower() if b else None
    if tok is None:
        return None
    if fmt == "ab":
        return {"a": NO, "b": YES}.get(tok)
    return {"yes": YES, "no": NO}.get(tok)


def parse_explanation(text: str) -> str:
    m = _EXPL.search(text or "")
    return (m.group(1) if m else (text or "")).strip()[:600]


def parse_confidence(text: str) -> Optional[float]:
    m = _CONF.search(text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 <= v <= 100 else None


def parse_json_list(text: str) -> Optional[list]:
    """Extract the first JSON list from a response (tolerates code fences / prose)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start, end = t.find("["), t.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        out = json.loads(t[start:end + 1])
        return out if isinstance(out, list) else None
    except json.JSONDecodeError:
        return None


# -- gray-box: answer-token probability ----------------------------------------------

def _get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def answer_token_prob(tokens) -> Optional[float]:
    """P(Yes) at the answer token, normalised over Yes/No among its top logprobs.
    `tokens` is the OpenAI `logprobs.content` list (objects or dicts with token, logprob,
    top_logprobs). The answer token is the first yes/no token after 'Answer'."""
    import math
    seen, text = False, ""
    for t in tokens or []:
        tok = (_get(t, "token") or "")
        text += tok
        if not seen:
            seen = "answer" in text.lower()
            continue
        if tok.strip(" :*_`\"'").lower() not in ("yes", "no"):
            continue
        p = {"yes": 0.0, "no": 0.0}
        for alt in (_get(t, "top_logprobs") or [{"token": tok, "logprob": _get(t, "logprob")}]):
            w = (_get(alt, "token") or "").strip(" :*_`\"'").lower()
            if w in p:
                p[w] += math.exp(_get(alt, "logprob"))
        tot = p["yes"] + p["no"]
        return p["yes"] / tot if tot > 0 else None
    return None


def p_yes_trailer(p: Optional[float]) -> str:
    return "" if p is None else f"\n[[p_yes: {p:.6f}]]"


def split_p_yes(text: str) -> tuple[str, Optional[float]]:
    """Remove the logprob trailer a backend appended; return (text, P(Yes) or None)."""
    m = _PYES.search(text or "")
    if not m:
        return text, None
    return text[:m.start()], float(m.group(1))


# -- free text signals of a rationale -------------------------------------------------

def rationale_words(expl: str) -> int:
    return len(re.findall(r"\w+", expl or ""))


def has_hedge(expl: str) -> bool:
    return bool(_HEDGES.search(expl or ""))
