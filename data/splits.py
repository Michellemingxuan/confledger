"""Dev / evaluation split and batch partition (Sec. 4.1).

Nothing evaluative touches the dev slice; nothing tuning touches the eval items.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from confledger import prompts as P
from confledger.utility.llm import Request, cached_client
from confledger.utility.records import Item

from .loaders import load_items


@dataclass
class Split:
    dev: list[Item]
    eval: list[Item]
    batches: list[list[str]]   # item_ids, n_batches x batch_size


def make_split(items: list[Item], n_dev: int = 100, n_eval: int = 500,
               batch_size: int = 50, seed: int = 0) -> Split:
    if len(items) < n_dev + n_eval:
        raise ValueError(f"need {n_dev + n_eval} items, have {len(items)}")
    pool = list(items)
    random.Random(seed).shuffle(pool)
    dev, ev = pool[:n_dev], pool[n_dev:n_dev + n_eval]
    ids = [it.item_id for it in ev]
    batches = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    return Split(dev=dev, eval=ev, batches=batches)


def load_split(cfg: dict, ds: str, generator=None) -> tuple[list[Item], Split]:
    """Items and dev/eval split for one dataset of a config. TriviaQA distractors come
    from the generator model (API calls on the first run, cached afterwards)."""
    sp = cfg["split"]
    gen = generator or cached_client(cfg["generator"], cfg["paths"]["cache"], "generator")

    def distractor_fn(q, gold):
        return gen.generate(Request(P.distractor_answer_prompt(q, gold), 0.7,
                                    "distractor_gen", 0, 60)).strip().splitlines()[0]

    items = load_items(ds, sp["n_dev"] + sp["n_eval"], cfg["seed"],
                       synthetic=cfg.get("synthetic", False), distractor_fn=distractor_fn)
    return items, make_split(items, sp["n_dev"], sp["n_eval"], sp["batch_size"], cfg["seed"])
