"""Local NLI judge for the semantic-entropy signal (decisions.md #30).

Rationales are grouped by bidirectional entailment, as in Kuhn et al. (2023): two
explanations share a group when each entails the other under an off-the-shelf NLI model
(default microsoft/deberta-large-mnli), each prefixed with the question. Greedy: a new
explanation joins the first existing group whose representative (its first member) it
mutually entails, else opens a new group. The confidence line a draw may carry is not
part of the reasoning and is removed first. Many items are clustered in lockstep so each
model call carries a full batch. Replaces the API judge's clustering call:
deterministic, no API spend. Needs the `nli` extra (torch, transformers).
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

DEFAULT_MODEL = "microsoft/deberta-large-mnli"
_CONF = re.compile(r"^\s*confidence\s*[:=].*$", re.I | re.M)


def reasoning_text(explanation: str) -> str:
    """The explanation without its confidence line, whitespace normalised."""
    return " ".join(_CONF.sub("", explanation or "").split())


class NLIClusterer:
    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 batch_size: int = 32, with_question: bool = True, max_length: int = 192, **_):
        self.model_id, self.device, self.model = model, device, None
        self.name, self.batch_size, self.with_question, self.max_length = f"local_nli:{model}", batch_size, with_question, max_length

    def _load(self) -> None:
        """Load on first use: initialising the GPU before `datasets` loads data (which may
        fork) hangs on macOS, and the configs build the judge before the split."""
        import torch  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415
        self.torch = torch
        # CUDA if present, else CPU. Not MPS: DeBERTa stalls there after many calls (macOS, torch 2.14).
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    def _entails(self, pairs: Sequence[tuple[str, str]]) -> list[bool]:
        if self.model is None:
            self._load()
        out = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation=True,
                           max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out += (self.model(**enc).logits.argmax(-1) == self.entail).tolist()
        return out

    def cluster(self, question: str, explanations: Sequence[str]) -> list[Optional[int]]:
        return self.cluster_many([question], [explanations])[0]

    def cluster_many(self, questions: Sequence[str], pools: Sequence[Sequence[str]]) -> list[list[Optional[int]]]:
        """Group ids per explanation, for many items at once; None for an empty explanation
        (it then counts as its own group in the entropy, the same conservative rule as a
        failed judge call). Items advance in lockstep, one explanation per step, and every
        step's comparisons go to the model as one batch."""
        texts = [[(f"{q} {r}" if self.with_question else r) if (r := reasoning_text(e)) else None for e in pool]
                 for q, pool in zip(questions, pools)]
        reps: list[list[str]] = [[] for _ in texts]
        labels: list[list[Optional[int]]] = [[] for _ in texts]
        for j in range(max((len(t) for t in texts), default=0)):
            todo = []                                    # (item, text) needing the model at this step
            for i, ts in enumerate(texts):
                if j >= len(ts):
                    continue
                t = ts[j]
                if t is None:
                    labels[i].append(None)
                elif t in reps[i]:                       # identical text: same group, no model call
                    labels[i].append(reps[i].index(t))
                elif not reps[i]:
                    reps[i].append(t)
                    labels[i].append(0)
                else:
                    todo.append((i, t))
            pairs = [pr for i, t in todo for r in reps[i] for pr in ((t, r), (r, t))]
            ent = iter(self._entails(pairs))
            for i, t in todo:
                both = [all((next(ent), next(ent))) for _ in reps[i]]   # t => r and r => t
                hit = next((g for g, ok in enumerate(both) if ok), None)
                if hit is None:
                    reps[i].append(t)
                    hit = len(reps[i]) - 1
                labels[i].append(hit)
        return labels
