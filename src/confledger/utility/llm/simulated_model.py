"""Synthetic black-box model for tests and the end-to-end dry run.

It speaks the same text protocol as a real model, so the full collection ->
parsing -> replay -> evaluation path is exercised without network access.

Latent structure per item (deterministic in item_id and seed):
  concept_confident : modal answer = gold, high stability, one dominant rationale,
                      passes label-action probes w.p. 0.95
  concept_uncertain : modal answer = gold w.p. 0.65, middling stability,
                      diffuse rationales
  cue               : stably follows a surface cue (modal answer = gold w.p. 0.55),
                      HIGH stability, diffuse rationales, passes negation/relabel
                      w.p. 0.5 -- the stable-but-wrong regime of Lemma 4.
Same-prompt draws share one prompt-specific quirk (within-phrasing correlation,
Lemma 2); paraphrase draws each get their own.

SIMULATION RESULTS ARE A PIPELINE CHECK, NOT EVIDENCE: the simulator encodes the
paper's hypotheses, so the method is expected to win here by construction.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from ..answer_parsing import p_yes_trailer
from ..records import NO, YES, flip
from .request import Request

PROFILES = {
    # (P[concept_confident], P[concept_uncertain], P[cue]), verbal informativeness w
    "recall":    ((0.50, 0.20, 0.30), 0.10),   # TriviaQA-like: many stable-but-wrong
    "reasoning": ((0.40, 0.42, 0.18), 0.10),   # StrategyQA-like: many unstable
    "boundary":  ((0.45, 0.35, 0.20), 0.55),   # MMLU-like: verbal carries real signal
}


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


class SimulatedBackend:
    def __init__(self, seed: int = 0, name: str = "simulated", n_reasons: int = 5, **_):
        self.seed = seed
        self.name = f"{name}:{seed}"
        self.n_reasons = n_reasons

    # -- latent state -------------------------------------------------------
    def _latent(self, item_id: str, gold: str, profile: str) -> dict:
        r = _rng(self.seed, "latent", item_id)
        (pc, pu, _), w = PROFILES.get(profile, PROFILES["recall"])
        u = r.random()
        if u < pc:
            kind, mode = "concept_confident", gold
            p = r.betavariate(18, 1)
            dom = 0.85
        elif u < pc + pu:
            kind = "concept_uncertain"
            mode = gold if r.random() < 0.65 else flip(gold)
            p = r.uniform(0.5, 0.85)
            dom = 0.35
        else:
            kind = "cue"
            mode = gold if r.random() < 0.55 else flip(gold)
            p = r.betavariate(15, 1)
            dom = 0.25
        return {"kind": kind, "mode": mode, "p": p, "dom": dom, "w": w,
                "correct": mode == gold, "quirk": r.gauss(0, 0.08)}

    def _p_variant(self, lat: dict, item_id: str, variant: str) -> float:
        if variant == "original":
            jitter = lat["quirk"]
        else:
            jitter = _rng(self.seed, "para", item_id, variant).gauss(0, 0.05)
        return min(0.999, max(0.5, lat["p"] + jitter))

    def _reason(self, lat: dict, r: random.Random) -> int:
        if r.random() < lat["dom"]:
            return 0
        return r.randint(1, self.n_reasons)

    @staticmethod
    def _conf(lat: dict, r: random.Random) -> int:
        """Verbal confidence: RLHF-style pile-up, informative in proportion to w."""
        sig = lat["w"] * float(lat["correct"]) + (1 - lat["w"]) * lat["p"]
        return max(50, min(99, round((50 + 45 * sig + r.gauss(0, 6)) / 5) * 5))

    @staticmethod
    def _fmt(ans: str, reason: int, ab: bool = False, hedge: bool = False, long: bool = False) -> str:
        tok = ("B" if ans == YES else "A") if ab else ("Yes" if ans == YES else "No")
        why = f"The key consideration is {'probably ' if hedge else ''}factor R{reason}."
        if long:
            why += " This follows from the definitions involved and the usual reading of the question."
        return f"Answer: {tok}\nExplanation: {why}"

    @staticmethod
    def _style(lat: dict, r: random.Random) -> dict:
        """Rationale style: unsure models hedge more, confident ones explain more."""
        return {"hedge": r.random() < {"concept_confident": 0.1, "concept_uncertain": 0.6, "cue": 0.25}[lat["kind"]],
                "long": r.random() < {"concept_confident": 0.6, "concept_uncertain": 0.3, "cue": 0.3}[lat["kind"]]}

    # -- protocol -------------------------------------------------------------
    def generate(self, req: Request) -> str:
        m = req.meta
        if req.kind == "paraphrase_gen":
            n = m.get("n", 12)
            return json.dumps([f"(rephrasing {j}) {m.get('question', '')}" for j in range(n)])
        if req.kind == "judge":
            reasons = []
            for i, (_, text) in enumerate(re.findall(r"^\[(\d+)\](.*)$", req.prompt, re.M)):
                hit = re.search(r"R(\d+)", text)
                reasons.append(int(hit.group(1)) if hit else -1 - i)
            first: dict = {}
            return json.dumps([first.setdefault(x, len(first)) for x in reasons])
        if req.kind == "distractor_gen":
            return "Some Other Answer"

        lat = self._latent(m["item_id"], m["gold"], m.get("profile", "recall"))
        variant = m.get("variant", "original")
        r = _rng(self.seed, "draw", m["item_id"], variant, req.sample, req.kind)

        if req.kind == "verbal":
            return f"Answer: {'Yes' if lat['mode'] == YES else 'No'}\nConfidence: {self._conf(lat, r)}"

        if req.kind == "challenge":
            hold = {"concept_confident": 0.85, "concept_uncertain": 0.4, "cue": 0.5}[lat["kind"]]
            base = m["base"]
            return self._fmt(base if r.random() < hold else flip(base), self._reason(lat, r))

        # kind == "answer"
        concept = lat["kind"] != "cue"
        if variant in ("neg", "relabel", "distractor", "heldout"):
            p_pass = {"neg": 0.95 if concept else 0.5,
                      "relabel": 0.95 if concept else 0.5,
                      "distractor": 0.97 if concept else 0.8,
                      "heldout": 0.95 if concept else 0.55}[variant]
            base = lat["mode"]
            passing = flip(base) if variant == "neg" else base
            ans = passing if r.random() < p_pass else flip(passing)
            out = self._fmt(ans, self._reason(lat, r), ab=(variant == "relabel"))
            if m.get("with_conf"):                 # confidence under negation: cue-followers drop
                drop = r.gauss(5, 4) if concept else r.gauss(18, 8)
                out += f"\nConfidence: {max(0, min(100, round(self._conf(lat, r) - drop)))}"
            return out
        p = self._p_variant(lat, m["item_id"], variant)
        ans = lat["mode"] if r.random() < p else flip(lat["mode"])
        if r.random() < 0.01:          # rare malformed output -> extraction failure
            return "I think it depends on how you read it."
        out = self._fmt(ans, self._reason(lat, r), **self._style(lat, r))
        if m.get("with_conf"):
            out += f"\nConfidence: {self._conf(lat, r)}"
        if req.logprobs:                           # gray-box: first-token probability of Yes
            q = min(0.999, max(0.5, p + r.gauss(0, 0.04)))
            out += p_yes_trailer(q if ans == YES else 1 - q)
        return out
