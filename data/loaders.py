"""Dataset loaders -> verification-format binary items (Sec. 4.1).

Requires the optional `datasets` extra for real data. Dataset IDs and field
names are configurable because HF mirrors differ; verify on first use.

TriviaQA-verification: "Is [candidate] the correct answer to [question]?",
candidates 50/50 gold vs plausible-wrong. Distractors come from a *generator*
model asked for a plausible incorrect answer of the same type, checked against
the gold aliases (fallback: another item's gold answer). The generator is a
separate model from the one under test, so no evaluation labels leak into
the model's behaviour.
"""
from __future__ import annotations

import random
import re
import string
from typing import Callable, Optional

from confledger.utility.records import NO, YES, Item

# (HF id, config, split) per dataset: the loaders' defaults.
HF_SOURCES = {"triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
              "strategyqa": ("ChilleD/StrategyQA", None, "train"),
              "mmlu": ("cais/mmlu", "all", "test")}


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _verif_q(question: str, candidate: str) -> str:
    return f"Question: {question}\nProposed answer: {candidate}\nIs the proposed answer correct?"


def load_triviaqa(n: int, seed: int, distractor_fn: Optional[Callable[[str, str], str]],
                  hf_id: str = HF_SOURCES["triviaqa"][0], config: str = HF_SOURCES["triviaqa"][1],
                  split: str = HF_SOURCES["triviaqa"][2]) -> list[Item]:
    from datasets import load_dataset  # noqa: PLC0415
    ds = load_dataset(hf_id, config, split=split)
    rng = random.Random(seed)
    # rc.nocontext lists most questions twice (web + wiki subsets, same question_id);
    # sample unique ids so no question repeats or straddles dev/eval.
    seen: set = set()
    uniq = [i for i, q in enumerate(ds["question_id"]) if not (q in seen or seen.add(q))]
    idx = rng.sample(uniq, min(n, len(uniq)))
    rows = [ds[i] for i in idx]
    golds = [r["answer"]["value"] for r in rows]
    items = []
    for j, row in enumerate(rows):
        q, gold = row["question"], row["answer"]["value"]
        aliases = {_norm(a) for a in row["answer"].get("aliases", [])} | {_norm(gold)}
        if rng.random() < 0.5:
            cand, label = gold, YES
        else:
            cand = distractor_fn(q, gold) if distractor_fn else None
            if not cand or _norm(cand) in aliases:
                cand = next(g for g in rng.sample(golds, len(golds)) if _norm(g) not in aliases)
            label = NO
        items.append(Item(f"triviaqa-{row['question_id']}", "triviaqa", _verif_q(q, cand),
                          label, {"raw_question": q, "candidate": cand}))
    return items


def load_strategyqa(n: int, seed: int, hf_id: str = HF_SOURCES["strategyqa"][0],
                    split: str = HF_SOURCES["strategyqa"][2], q_field: str = "question",
                    a_field: str = "answer") -> list[Item]:
    from datasets import load_dataset  # noqa: PLC0415
    ds = load_dataset(hf_id, split=split)
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    return [Item(f"strategyqa-{i}", "strategyqa", ds[i][q_field],
                 YES if bool(ds[i][a_field]) else NO) for i in idx]


def load_mmlu(n: int, seed: int, hf_id: str = HF_SOURCES["mmlu"][0], config: str = HF_SOURCES["mmlu"][1],
              split: str = HF_SOURCES["mmlu"][2]) -> list[Item]:
    """Binarized: 'Is option X correct?' with 50/50 correct vs random wrong option."""
    from datasets import load_dataset  # noqa: PLC0415
    ds = load_dataset(hf_id, config, split=split)
    rng = random.Random(seed)
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    items = []
    for i in idx:
        row = ds[i]
        letters = "ABCD"[:len(row["choices"])]
        opts = "\n".join(f"({l}) {c}" for l, c in zip(letters, row["choices"]))
        correct = row["answer"]
        if rng.random() < 0.5:
            pick, label = correct, YES
        else:
            pick = rng.choice([k for k in range(len(letters)) if k != correct])
            label = NO
        q = (f"Question: {row['question']}\n{opts}\n"
             f"Is option ({letters[pick]}) the correct answer?")
        items.append(Item(f"mmlu-{i}", "mmlu", q, label, {"subject": row.get("subject")}))
    return items


def synthetic_items(dataset: str, n: int, seed: int = 0) -> list[Item]:
    """Items for the simulated backend; dataset name selects the latent profile."""
    profile = {"triviaqa": "recall", "strategyqa": "reasoning", "mmlu": "boundary"}.get(dataset, "recall")
    rng = random.Random(seed)
    return [Item(f"{dataset}-sim-{i}", dataset, f"[{dataset} synthetic question {i}] Is this true?",
                 YES if rng.random() < 0.5 else NO, {"sim_profile": profile})
            for i in range(n)]


def load_items(dataset: str, n: int, seed: int = 0, synthetic: bool = False,
               distractor_fn=None, **kw) -> list[Item]:
    if synthetic:
        return synthetic_items(dataset, n, seed)
    if dataset == "triviaqa":
        return load_triviaqa(n, seed, distractor_fn, **kw)
    if dataset == "strategyqa":
        return load_strategyqa(n, seed, **kw)
    if dataset == "mmlu":
        return load_mmlu(n, seed, **kw)
    raise ValueError(dataset)
