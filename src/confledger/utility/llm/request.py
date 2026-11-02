"""Backend-agnostic request object. Real providers read only `prompt`;
the simulator also reads `meta` (item id, variant) to generate responses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Rejected(Exception):
    """The provider refused this request for good (HTTP 400 / 403, e.g. SafeChain's input
    redactor or firewall; AgenticSys_v2's FirewallRejection). Retrying the same prompt
    cannot help, so CachedClient raises it at once instead of backing off."""


@dataclass
class Request:
    prompt: str
    temperature: float = 0.0
    kind: str = "answer"          # answer | paraphrase_gen | judge | challenge | verbal | distractor_gen
    sample: int = 0               # distinguishes repeated samples of an identical prompt
    max_tokens: int = 300
    meta: dict = field(default_factory=dict)
    logprobs: bool = False        # ask for token logprobs; the backend appends a [[p_yes: ...]] trailer


class Backend(Protocol):
    name: str

    def generate(self, req: Request) -> str: ...
