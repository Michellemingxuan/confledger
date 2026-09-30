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
