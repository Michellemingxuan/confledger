"""Dev (D_fit) / calibration (D_cal) / evaluation split and batch partition (Sec. 4.1).

Nothing evaluative touches the dev slice; nothing tuning touches the eval items; the
calibration set is touched only after the score is frozen on dev (docs/reference.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from confledger import prompts as P
from confledger.utility.llm import Request, cached_client
from confledger.utility.records import Item

from .loaders import load_items


@dataclass
class Split:
    dev: list[Item]
    eval: list[Item]
    batches: list[list[str]]   # item_ids; one population (the whole eval set) unless batch_size is set
    cal: list[Item] = field(default_factory=list)


def make_split(items: list[Item], n_dev: int = 100, n_eval: int = 500,
               batch_size: int | None = None, seed: int = 0, n_cal: int = 0) -> Split:
    """Shuffle once; dev, then eval, then calibration (so n_cal = 0 reproduces the
    two-way split of the same items). batch_size None (default, decisions.md #29): the
    eval set is one population and the top share is taken over all of it."""
    if len(items) < n_dev + n_eval + n_cal:
        raise ValueError(f"need {n_dev + n_eval + n_cal} items, have {len(items)}")
    pool = list(items)
    random.Random(seed).shuffle(pool)
    dev, ev = pool[:n_dev], pool[n_dev:n_dev + n_eval]
    cal = pool[n_dev + n_eval:n_dev + n_eval + n_cal]
    ids = [it.item_id for it in ev]
    bs = batch_size or len(ids)
    batches = [ids[i:i + bs] for i in range(0, len(ids), bs)]
    return Split(dev=dev, eval=ev, batches=batches, cal=cal)


def load_split(cfg: dict, ds: str, generator=None) -> tuple[list[Item], Split]:
    """Items and dev/eval/calibration split for one dataset of a config. TriviaQA distractors come
    from the generator model (API calls on the first run, cached afterwards)."""
    sp = cfg["split"]
    gen = generator or cached_client(cfg["generator"], cfg["paths"]["cache"], "generator")

    def distractor_fn(q, gold):
        return gen.generate(Request(P.distractor_answer_prompt(q, gold), 0.7,
                                    "distractor_gen", 0, 60)).strip().splitlines()[0]

    n_cal = sp.get("n_cal", 0)
    items = load_items(ds, sp["n_dev"] + sp["n_eval"] + n_cal, cfg["seed"],
                       synthetic=cfg.get("synthetic", False), distractor_fn=distractor_fn)
    return items, make_split(items, sp["n_dev"], sp["n_eval"], sp.get("batch_size"), cfg["seed"], n_cal)
