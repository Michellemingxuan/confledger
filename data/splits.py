"""Dev (D_fit) / calibration (D_cal) / evaluation split and batch partition (Sec. 4.1).

Nothing evaluative touches the dev slice; nothing tuning touches the eval items; the
calibration set is touched only after the score is frozen on dev (docs/reference.md).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from confledger import prompts as P
from confledger.utility.llm import CachedClient, Request, cache_path, make_backend
from confledger.utility.records import Item

from .loaders import build_items, item_id, load_items, load_source, usable_rows


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


def set_sizes(sp: dict, pool: int | None = None) -> tuple[dict[str, int], dict[str, int]]:
    """(full, used) item counts per set. `split.frac` (e.g. dev 0.2, cal 0.2, eval 0.6) cuts the
    whole source split of `pool` distinct questions; n_dev / n_eval / n_cal then select the first
    items of each full set (a pilot nested in the full run). Without `frac`, n_* are drawn
    directly, the older form (decisions.md #35)."""
    used = {s: sp.get(f"n_{s}") for s in ("dev", "eval", "cal")}
    if "frac" not in sp:
        used = {s: v or 0 for s, v in used.items()}
        return used, used
    if pool is None:
        raise ValueError("split.frac needs the source size")
    f = sp["frac"]
    full = {"dev": round(f["dev"] * pool), "cal": round(f.get("cal", 0) * pool)}
    full["eval"] = pool - full["dev"] - full["cal"] if f["eval"] + f["dev"] + f.get("cal", 0) >= 1 - 1e-9 \
        else round(f["eval"] * pool)
    used = {s: full[s] if used[s] is None else used[s] for s in full}
    if any(used[s] > full[s] for s in full):
        raise ValueError(f"split uses {used}, more than the full sets {full}")
    return full, used


def _rows(cfg: dict, dataset: str, src) -> dict[str, list]:
    """Source rows of each used set: one seeded shuffle of the distinct questions."""
    sp, seed = cfg["split"], cfg["seed"]
    rows = usable_rows(dataset, src)
    full, used = set_sizes(sp, len(rows))
    if "frac" in sp:
        order = list(rows)
    else:                                       # older form: draw the items, then shuffle them
        order = random.Random(seed).sample(rows, sum(full.values()))
    random.Random(seed).shuffle(order)
    out, at = {}, 0
    for s in ("dev", "eval", "cal"):            # this order keeps older splits unchanged
        out[s] = order[at:at + used[s]]
        at += full[s]
    return out


def _split(parts: dict[str, list], batch_size: int | None) -> Split:
    ids = [it.item_id for it in parts["eval"]]
    bs = batch_size or max(1, len(ids))
    return Split(dev=parts["dev"], eval=parts["eval"], batches=[ids[i:i + bs] for i in range(0, len(ids), bs)],
                 cal=parts["cal"])


def generator_client(cfg: dict):
    """The generator behind its cache. `paths.generator_cache` names one shared file (kept in
    git), so every machine and model sees the same paraphrases and distractors (#35)."""
    path = cfg["paths"].get("generator_cache") or cache_path(cfg["paths"]["cache"], "generator")
    return CachedClient(make_backend(cfg["generator"]), path)


def load_split(cfg: dict, ds: str, generator=None) -> tuple[list[Item], Split]:
    """Items and dev/eval/calibration split for one dataset of a config. Only the items in use
    are built; TriviaQA distractors come from the generator model (API calls on the first
    run, cached afterwards)."""
    sp = cfg["split"]
    if cfg.get("synthetic"):
        n = set_sizes(sp)[0]
        items = load_items(ds, sum(n.values()), cfg["seed"], synthetic=True)
        return items, make_split(items, n["dev"], n["eval"], sp.get("batch_size"), cfg["seed"], n["cal"])
    gen = generator
    if ds == "triviaqa" and gen is None:
        gen = generator_client(cfg)

    def distractor_fn(q, gold):
        return gen.generate(Request(P.distractor_answer_prompt(q, gold), 0.7,
                                    "distractor_gen", 0, 60)).strip().splitlines()[0]

    src = load_source(ds)
    rows = _rows(cfg, ds, src)
    flat = [r for s in ("dev", "eval", "cal") for r in rows[s]]
    built = dict(zip(flat, build_items(ds, src, flat, cfg["seed"], distractor_fn if gen else None)))
    parts = {s: [built[r] for r in rows[s]] for s in rows}
    return list(built.values()), _split(parts, sp.get("batch_size"))


def split_ids(cfg: dict, ds: str) -> tuple[list[str], list[list[str]], list[str]]:
    """(dev ids, eval batches, cal ids) without building any item: no generator calls, so
    collections can be evaluated anywhere."""
    sp = cfg["split"]
    if cfg.get("synthetic"):
        _, split = load_split(cfg, ds)
        return [i.item_id for i in split.dev], split.batches, [i.item_id for i in split.cal]
    src = load_source(ds)
    rows = _rows(cfg, ds, src)
    ids = {s: [item_id(ds, src, r) for r in rows[s]] for s in rows}
    bs = sp.get("batch_size") or max(1, len(ids["eval"]))
    return ids["dev"], [ids["eval"][i:i + bs] for i in range(0, len(ids["eval"]), bs)], ids["cal"]


def source_size(cfg: dict, ds: str) -> int | None:
    """Distinct questions in the source split, read only when split.frac needs it."""
    if "frac" not in cfg["split"] or cfg.get("synthetic"):
        return None
    return len(usable_rows(ds, load_source(ds)))
