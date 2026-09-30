"""One superset collection per (dataset, model)  (Sec. 4.3).

Every method in the paper -- the adaptive pipeline, its ablations, all baselines,
the budget curves, the evidence-equivalence arms and all 32 Shapley coalitions --
is a *replay policy* over this record (methods.py), with the calls that policy
would have made counted explicitly. The collection is the only API spend.

Per item (defaults): 1 paraphrase-gen call (generator model), 12 fresh-paraphrase
draws, 12 same-prompt draws, 6 extra reuse draws, 4 temperature-0 probes,
1 challenge, 1 verbal, 1 judge call  ->  ~38 calls (cheaper than any sweep).

Real backends never see the gold label: it travels only in `meta`, which only
the simulator reads.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from data import load_split

from . import prompts as P
from .utility.answer_parsing import parse_answer, parse_confidence, parse_explanation, parse_json_list
from .methods import cel_from_settings, max_cel_calls
from .utility.config import check_config, model_spec, record_run
from .utility.llm import CachedClient, Request, cached_client
from .sprt_posterior import majority
from .utility.records import CaseData, Draw, Item

log = logging.getLogger(__name__)


@dataclass
class CollectConfig:
    n_paraphrases: int = 12
    n_same: int = 16                 # >= max CEL cost per case, so the matched budget fits
    n_reuse_paras: int = 3
    n_reuse_samples: int = 3
    temperature: float = 0.7
    sample_offset: int = 0          # >0 re-draws everything (Jaccard stability run)
    verbal_per_draw: bool = False   # confidence line on every draw; no separate verbal call
    workers: int = 8


class Collector:
    def __init__(self, target: CachedClient, generator: CachedClient, judge: CachedClient,
                 cfg: CollectConfig, alt_judge: Optional[CachedClient] = None):
        self.t, self.g, self.j, self.cfg, self.alt = target, generator, judge, cfg, alt_judge
        self.failures = {"answer_parse": 0, "paraphrase": 0, "judge": 0}

    # -- helpers ---------------------------------------------------------------
    def _meta(self, item: Item, variant: str, **kw) -> dict:
        return {"item_id": item.item_id, "gold": item.gold,
                "profile": item.meta.get("sim_profile", "recall"), "variant": variant, **kw}

    def _draw(self, item: Item, prompt: str, variant: str, sample: int,
              temperature: float, fmt: str = "yn", kind: str = "answer", **kw) -> Draw:
        txt = self.t.generate(Request(prompt, temperature, kind, sample,
                                      meta=self._meta(item, variant, **kw)))
        ans = parse_answer(txt, fmt)
        if ans is None:
            self.failures["answer_parse"] += 1
        conf = parse_confidence(txt) if kw.get("with_conf") else None
        return Draw(ans, parse_explanation(txt), confidence=conf)

    def paraphrases(self, item: Item) -> list[str]:
        n = self.cfg.n_paraphrases
        for attempt in range(3):
            txt = self.g.generate(Request(P.paraphrase_gen_prompt(item.question, n), 0.7,
                                          "paraphrase_gen", attempt, max_tokens=2000,
                                          meta=self._meta(item, "paraphrase_gen", n=n,
                                                          question=item.question)))
            out = [s for s in (parse_json_list(txt) or []) if isinstance(s, str) and s.strip()]
            if len(out) >= n:
                return out[:n]
        self.failures["paraphrase"] += 1
        log.warning("paraphrase generation short for %s; padding", item.item_id)
        out = out + [f"Please answer the following. {item.question}"] * n
        return out[:n]

    def _cluster(self, client: CachedClient, item: Item, expls: list[str]) -> list[Optional[int]]:
        txt = client.generate(Request(P.judge_prompt(item.question, expls), 0.0, "judge", 0,
                                      max_tokens=600, meta=self._meta(item, "judge")))
        labels = parse_json_list(txt)
        if not labels or len(labels) != len(expls) or not all(isinstance(x, int) for x in labels):
            self.failures["judge"] += 1
            return [None] * len(expls)          # conservative: every draw its own cluster
        return labels

    # -- one item --------------------------------------------------------------
    def collect(self, item: Item) -> CaseData:
        c, off, T = self.cfg, self.cfg.sample_offset, self.cfg.temperature
        paras = self.paraphrases(item)
        vpd = c.verbal_per_draw
        ask = lambda q: P.answer_prompt(q, with_confidence=vpd)
        fresh = [self._draw(item, ask(paras[j]), f"para:{j}", off, T, with_conf=vpd)
                 for j in range(c.n_paraphrases)]
        same = [self._draw(item, ask(item.question), "original", off + s, T, with_conf=vpd)
                for s in range(c.n_same)]
        reuse = [[fresh[p]] + [self._draw(item, ask(paras[p]), f"para:{p}", off + s, T, with_conf=vpd)
                               for s in range(1, c.n_reuse_samples)]
                 for p in range(c.n_reuse_paras)]
        probes = {
            "neg": self._draw(item, P.negation_prompt(item.question), "neg", off, 0.0).answer,
            "relabel": self._draw(item, P.relabel_prompt(item.question), "relabel", off, 0.0,
                                  fmt="ab").answer,
            "distractor": self._draw(item, P.distractor_prompt(item.question, item.item_id),
                                     "distractor", off, 0.0).answer,
            "heldout": self._draw(item, P.heldout_prompt(item.question), "heldout", off, 0.0).answer,
        }
        base = same[0].answer or "yes"            # challenge the original-wording draw's answer
        ch = self._draw(item, P.challenge_prompt(item.question, base), "challenge", off, 0.0,
                        kind="challenge", base=base)
        if vpd:                                   # verbal = the first same-prompt draw
            v_ans, v_conf = same[0].answer, same[0].confidence
        else:
            vtxt = self.t.generate(Request(P.verbal_prompt(item.question), 0.0, "verbal", off,
                                           meta=self._meta(item, "verbal")))
            v_ans, v_conf = parse_answer(vtxt), parse_confidence(vtxt)

        # one judge call clusters every rationale in the record
        pool = fresh + same + [d for row in reuse for d in row[1:]]
        for d, lab in zip(pool, self._cluster(self.j, item, [d.explanation for d in pool])):
            d.cluster = lab
        alt = None
        if self.alt is not None:
            alt = {"labels": self._cluster(self.alt, item, [d.explanation for d in pool])}

        return CaseData(item.item_id, item.dataset, item.gold, fresh, same, reuse, probes,
                        challenge_answer=ch.answer, challenge_base=base,
                        verbal_answer=v_ans, verbal_conf=v_conf,
                        alt_clusters=alt, meta={"sample_offset": off, "verbal_per_draw": vpd})

    # -- many items, resumable JSONL -----------------------------------------------
    def collect_all(self, items: list[Item], out_path: str | Path) -> list[CaseData]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done: dict[str, CaseData] = {}
        if out_path.exists():
            for line in out_path.read_text().splitlines():
                cd = CaseData.from_json(json.loads(line))
                done[cd.item_id] = cd
        todo = [it for it in items if it.item_id not in done]
        with out_path.open("a") as fh, ThreadPoolExecutor(self.cfg.workers) as ex:
            futs = {ex.submit(self.collect, it): it for it in todo}
            for f in as_completed(futs):
                cd = f.result()
                fh.write(json.dumps(cd.to_json()) + "\n")
                fh.flush()
                done[cd.item_id] = cd
        log.info("collection %s: %d items, failures=%s", out_path.name, len(done), self.failures)
        return [done[it.item_id] for it in items]


def target_calls_per_item(collect_cfg: dict) -> int:
    """Model-under-test calls to collect one item: fresh-paraphrase draws, same-prompt
    draws, extra reuse draws (reuse[p][0] is fresh[p], free), 4 probes (negation,
    relabel, distractor, held-out; fixed templates, no generation call), challenge, and the
    verbal call unless verbal confidence rides on every draw (verbal_per_draw)."""
    c = collect_cfg
    return (c["n_paraphrases"] + c["n_same"] + c["n_reuse_paras"] * (c["n_reuse_samples"] - 1)
            + 4 + 1 + (0 if c.get("verbal_per_draw") else 1))


def estimate_calls(cfg: dict, reseed_batches: int = 0) -> list[dict]:
    """Expected API calls of `collect_config`, by model role. Per (dataset, model):
    target draws/probes and 1 judge call per item (dev + eval + re-drawn batches), plus
    the alt judge on the judge-sensitivity dataset. Per dataset (shared by all models,
    since the cache key ignores the target): 1 paraphrase-generation call per question
    and, for TriviaQA, a distractor-generation call for ~half the items. Excludes
    retries after malformed paraphrase output."""
    sp = cfg["split"]
    n = sp["n_dev"] + sp["n_eval"]
    r = reseed_batches * sp["batch_size"]
    per = target_calls_per_item(cfg["collect"])
    rows = []
    for ds in cfg["_datasets"]:
        gen = n + (n // 2 if ds == "triviaqa" and not cfg.get("synthetic") else 0)
        for mk in cfg["_models"]:
            alt = n if ds == cfg.get("judge_sensitivity_dataset") and cfg.get("alt_judge") else 0
            rows.append({"dataset": ds, "model": mk, "items": n + r, "target": per * (n + r),
                         "judge": n + r + alt, "generator": gen})
            gen = 0                             # paraphrases/distractors are cached across models
    return rows


def collection_path(cfg: dict, ds: str, mk: str, suffix: str = "") -> Path:
    """<collections>/<ds>__<mk><suffix>.jsonl; suffix '' (eval), '__dev', '__reseed'."""
    return Path(cfg["paths"]["collections"]) / f"{ds}__{mk}{suffix}.jsonl"


def load_collection(path: str | Path) -> dict[str, CaseData]:
    return {cd.item_id: cd for cd in
            (CaseData.from_json(json.loads(l)) for l in Path(path).read_text().splitlines())}


def collect_config(cfg: dict, reseed_batches: int = 0) -> None:
    """Collect every selected (dataset, model) cell of a config: eval items, the dev
    slice, and optionally the first `reseed_batches` batches re-drawn (Jaccard check).
    The only function that spends API calls; refuses placeholder model ids."""
    check_config(cfg)
    c = cfg["collect"]
    base, _, plan = cel_from_settings(cfg.get("sprt"))
    bound = max_cel_calls(c.get("verbal_per_draw", False), base)
    need_fresh = max(12, plan["depth"] + 3)
    if c["n_same"] < bound or c["n_paraphrases"] < need_fresh:
        log.warning("n_paraphrases=%d, n_same=%d: evaluation needs n_paraphrases >= %d and "
                    "n_same >= %d (CEL's per-case cost bounds the matched budget); this "
                    "collection can test the plumbing but evaluate.py will refuse it",
                    c["n_paraphrases"], c["n_same"], need_fresh, bound)
    record_run(cfg, f"collect --reseed-batches {reseed_batches}")
    cache = cfg["paths"]["cache"]
    gen = cached_client(cfg["generator"], cache, "generator")
    judge = cached_client(cfg["judge"], cache, "judge")
    for ds in cfg["_datasets"]:
        items, split = load_split(cfg, ds, gen)
        by_id = {it.item_id: it for it in items}
        for mk in cfg["_models"]:
            target = cached_client(model_spec(cfg, mk), cache, f"target-{mk}")
            alt = None
            if ds == cfg.get("judge_sensitivity_dataset") and cfg.get("alt_judge"):
                alt = cached_client(cfg["alt_judge"], cache, "judge-alt")
            col = Collector(target, gen, judge, CollectConfig(**cfg["collect"]), alt)
            col.collect_all(split.eval, collection_path(cfg, ds, mk))
            col.collect_all(split.dev, collection_path(cfg, ds, mk, "__dev"))
            if reseed_batches:
                ids = [i for b in split.batches[:reseed_batches] for i in b]
                cc2 = CollectConfig(**{**cfg["collect"], "sample_offset": 1000})
                Collector(target, gen, judge, cc2).collect_all(
                    [by_id[i] for i in ids], collection_path(cfg, ds, mk, "__reseed"))
            log.info("%s/%s: target cache hits=%d misses=%d", ds, mk, target.hits, target.misses)
