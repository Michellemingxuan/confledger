"""Core data structures. Everything serialises to JSON for the collection files."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

YES, NO = "yes", "no"


def flip(ans: Optional[str]) -> Optional[str]:
    return None if ans is None else (NO if ans == YES else YES)


@dataclass
class Item:
    """A binary (verification-format) question with its gold answer."""
    item_id: str
    dataset: str
    question: str
    gold: str                     # "yes" or "no"
    meta: dict = field(default_factory=dict)


@dataclass
class Draw:
    """One sampled response: parsed answer (None = extraction failure) + rationale."""
    answer: Optional[str]
    explanation: str = ""
    cluster: Optional[int] = None  # rationale-equivalence class, filled by the judge
    confidence: Optional[float] = None  # verbal confidence 0-100 (collect.verbal_per_draw)
    p_yes: Optional[float] = None       # P(Yes) at the answer token, from logprobs (collect.logprobs; gray-box)


@dataclass
class CaseData:
    """Superset collection for one item. Every method is a replay over this record.

    fresh : draws, each on a distinct paraphrase (Lemma 1: exactly i.i.d.)
    same  : draws, all on the original phrasing (naive self-consistency baseline)
    reuse : reuse[p][r] = sample r on paraphrase p, p<3 (no-spread ablation;
            reuse[p][0] shares its cache key with fresh[p], so it is free)
    probes: temperature-0 answers under label-action transforms
            {"neg": ..., "relabel": ..., "distractor": ..., "heldout": ...}
    """
    item_id: str
    dataset: str
    gold: str
    fresh: list[Draw]
    same: list[Draw]
    reuse: list[list[Draw]]
    probes: dict[str, Optional[str]]
    challenge_answer: Optional[str] = None
    challenge_base: Optional[str] = None   # the answer that was challenged
    verbal_answer: Optional[str] = None
    verbal_conf: Optional[float] = None
    alt_clusters: Optional[dict] = None    # judge-swap sensitivity (optional)
    probe_conf: dict = field(default_factory=dict)  # verbal confidence on a probe, e.g. {"neg": 70.0}
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "CaseData":
        d = dict(d)
        d["fresh"] = [Draw(**x) for x in d["fresh"]]
        d["same"] = [Draw(**x) for x in d["same"]]
        d["reuse"] = [[Draw(**x) for x in row] for row in d["reuse"]]
        return CaseData(**d)
