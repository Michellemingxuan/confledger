"""Dataset loaders -> verification-format binary items (Sec. 4.1).

Requires the optional `datasets` extra for real data. Dataset IDs and field
names are configurable because HF mirrors differ; verify on first use.

TriviaQA-verification: "Is [candidate] the correct answer to [question]?",
candidates 50/50 gold vs plausible-wrong. Distractors come from a *generator*
model asked for a plausible incorrect answer of the same type, checked against
the gold aliases (fallback: another question's gold answer). Each item draws from its own
seeded randomness, so building a subset gives the same items as building them all. The generator is a
separate model from the one under test, so no evaluation labels leak into
the model's behaviour.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

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


def _item_rng(seed: int, dataset: str, row) -> random.Random:
    """Per-item randomness, so an item is the same whichever other items are built
    (a pilot builds only its own items; decisions.md #35)."""
    return random.Random(f"{seed}:{dataset}:{row}")


# Local copy of the three source splits (python -m data.export_sources), for machines without
# Hugging Face access (the private environment). One gzipped JSONL per dataset, read with the
# standard library only. When a file exists, load_source reads it instead of Hugging Face.
SOURCES_DIR = Path(os.environ.get("CONFLEDGER_SOURCES") or Path(__file__).parent / "sources")
_UNUSED_COLUMNS = {"entity_pages", "search_results"}      # TriviaQA contexts: large, never read


class LocalRows:
    """The part of the `datasets` API the loaders use: len(ds), ds[i] -> row dict, ds["col"] -> list."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, k):
        return self._rows[k] if isinstance(k, int) else [r[k] for r in self._rows]

    @property
    def column_names(self) -> list[str]:
        return list(self._rows[0]) if self._rows else []


def write_rows(rows, path: Path) -> str:
    """Write rows as gzipped JSONL, reproducibly (no timestamp, so the same rows give the same
    bytes); returns the sha256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
        for r in rows:
            gz.write((json.dumps({k: v for k, v in r.items() if k not in _UNUSED_COLUMNS},
                                 ensure_ascii=False) + "\n").encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> LocalRows:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return LocalRows([json.loads(line) for line in f])


def export_sources(datasets=None, out_dir: Optional[Path] = None) -> dict:
    """Export each source split from Hugging Face to `out_dir` (default data/sources) and write
    manifest.json (rows and sha256 per file). Needs the `datasets` extra and the Hugging Face cache."""
    out = Path(out_dir or SOURCES_DIR)
    manifest = {}
    for name in datasets or HF_SOURCES:
        ds = load_source(name, local=False)
        sha = write_rows(ds, out / f"{name}.jsonl.gz")
        manifest[name] = {"rows": len(ds), "sha256": sha, "source": list(HF_SOURCES[name])}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def verify_sources(datasets=None, out_dir: Optional[Path] = None) -> dict:
    """Per dataset: row count and whether the local file matches manifest.json. Raises if a file
    is missing or its hash differs, so a damaged copy is caught before a run."""
    out = Path(out_dir or SOURCES_DIR)
    manifest = json.loads((out / "manifest.json").read_text())
    got = {}
    for name in datasets or HF_SOURCES:
        path = out / f"{name}.jsonl.gz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise ValueError(f"{path}: sha256 differs from manifest.json")
        n = len(read_rows(path))
        if n != manifest[name]["rows"]:
            raise ValueError(f"{path}: {n} rows, manifest says {manifest[name]['rows']}")
        got[name] = n
    return got


def load_source(dataset: str, local: bool = True):
    """The dataset's source split: the local export when present (and `local`), else Hugging Face."""
    path = SOURCES_DIR / f"{dataset}.jsonl.gz"
    if local and path.exists():
        return read_rows(path)
    from datasets import load_dataset  # noqa: PLC0415
    hf_id, config, split = HF_SOURCES[dataset]
    return load_dataset(hf_id, config, split=split) if config else load_dataset(hf_id, split=split)


def usable_rows(dataset: str, ds) -> list[int]:
    """Row indices of distinct questions, in source order. TriviaQA rc.nocontext lists most
    questions twice (web + wiki, same question_id); MMLU repeats some question + choices."""
    if dataset == "triviaqa":
        key = ds["question_id"]
    elif dataset == "mmlu":
        key = [(q, tuple(c)) for q, c in zip(ds["question"], ds["choices"])]
    else:
        return list(range(len(ds)))
    seen: set = set()
    return [i for i, k in enumerate(key) if not (k in seen or seen.add(k))]


def item_id(dataset: str, ds, row: int) -> str:
    return f"triviaqa-{ds[row]['question_id']}" if dataset == "triviaqa" else f"{dataset}-{row}"


def build_items(dataset: str, ds, rows: Sequence[int], seed: int,
                distractor_fn: Optional[Callable[[str, str], str]] = None, workers: int = 8) -> list[Item]:
    """Verification items for the given source rows, each from its own seeded randomness."""
    if dataset == "triviaqa":
        return _triviaqa_items(ds, rows, seed, distractor_fn, workers)
    if dataset == "strategyqa":
        return [Item(f"strategyqa-{i}", "strategyqa", ds[i]["question"],
                     YES if bool(ds[i]["answer"]) else NO) for i in rows]
    return [_mmlu_item(ds[i], i, _item_rng(seed, dataset, i)) for i in rows]


def _triviaqa_items(ds, rows, seed, distractor_fn, workers) -> list[Item]:
    recs = [ds[i] for i in rows]
    rngs = [_item_rng(seed, "triviaqa", r["question_id"]) for r in recs]
    wrong = [rng.random() >= 0.5 for rng in rngs]            # half the candidates are wrong
    asks = [(r["question"], r["answer"]["value"]) for r, w in zip(recs, wrong) if w]
    if distractor_fn and asks:                               # generator calls in parallel, order kept
        with ThreadPoolExecutor(workers) as ex:
            results = ex.map(lambda a: distractor_fn(*a), asks)
            try:                                              # progress bar if tqdm is installed
                from tqdm import tqdm  # noqa: PLC0415
                results = tqdm(results, total=len(asks), desc="triviaqa distractors", unit="call",
                               dynamic_ncols=True, mininterval=2.0, leave=False)
            except ImportError:
                pass
            got = iter(list(results))
    else:
        got = iter([None] * len(asks))
    golds = None
    items = []
    for r, w, rng in zip(recs, wrong, rngs):
        q, gold = r["question"], r["answer"]["value"]
        cand, label = gold, YES
        if w:
            aliases = {_norm(a) for a in r["answer"].get("aliases", [])} | {_norm(gold)}
            cand, label = next(got), NO
            if not cand or _norm(cand) in aliases:            # fallback: another question's gold answer
                golds = golds or sorted({x["value"] for x in ds["answer"]})
                cand = next(g for g in rng.sample(golds, len(golds)) if _norm(g) not in aliases)
        items.append(Item(f"triviaqa-{r['question_id']}", "triviaqa", _verif_q(q, cand),
                          label, {"raw_question": q, "candidate": cand}))
    return items


def _mmlu_item(row, i: int, rng: random.Random) -> Item:
    """Binarized: 'Is option X correct?' with 50/50 correct vs random wrong option."""
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
    return Item(f"mmlu-{i}", "mmlu", q, label, {"subject": row.get("subject")})


def load_triviaqa(n: int, seed: int, distractor_fn: Optional[Callable[[str, str], str]]) -> list[Item]:
    return load_items("triviaqa", n, seed, distractor_fn=distractor_fn)


def synthetic_items(dataset: str, n: int, seed: int = 0) -> list[Item]:
    """Items for the simulated backend; dataset name selects the latent profile."""
    profile = {"triviaqa": "recall", "strategyqa": "reasoning", "mmlu": "boundary"}.get(dataset, "recall")
    rng = random.Random(seed)
    return [Item(f"{dataset}-sim-{i}", dataset, f"[{dataset} synthetic question {i}] Is this true?",
                 YES if rng.random() < 0.5 else NO, {"sim_profile": profile})
            for i in range(n)]


def load_items(dataset: str, n: int, seed: int = 0, synthetic: bool = False,
               distractor_fn=None) -> list[Item]:
    """n items drawn at random from the dataset's distinct questions."""
    if synthetic:
        return synthetic_items(dataset, n, seed)
    if dataset not in HF_SOURCES:
        raise ValueError(dataset)
    ds = load_source(dataset)
    rows = usable_rows(dataset, ds)
    return build_items(dataset, ds, random.Random(seed).sample(rows, min(n, len(rows))), seed, distractor_fn)
