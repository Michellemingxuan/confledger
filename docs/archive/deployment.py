# ARCHIVED 2026-09-29: not used by the Sec. 4 experiments. To restore, move back to
# src/confledger/deployment.py (its relative imports assume that location) and re-add
# the test kept at the bottom of this file to tests/test_pipeline.py.

"""Deployment runner: label-free, adaptive, calls the API only when needed.

Implements the 'lean + certify-the-contenders' allocation (Sec. 3 budget
discussion): every case gets the 3-draw screen; survivors get a lean stage 2
(2 draws + probe battery); only the top `n_contenders` per batch are deepened
to the certification depth. Returns the top-k with per-case tail masses and the
Prop. 2 guarantee. No gold labels are used anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts as P
from .ranking import borda_scores, select_top_k
from .collection import Collector, CollectConfig
from .utility.llm import CachedClient
from .methods import LABEL_ACTION, CaseScore
from .sprt_posterior import (SPRT, EvidenceLedger, evidence_llr, k_effective, lower_bound, majority,
                    miller_madow_entropy, selection_guarantee, tail_mass,
                    tail_mass_effective)
from .utility.records import Item

import random


@dataclass
class OnlineConfig:
    s1: int = 3
    lean_draws: int = 2
    deep_draws: int = 9            # total draws for contenders (Cor. 1 anchor)
    n_contenders: int = 15
    probes: tuple = ("neg", "relabel", "distractor")
    k: int = 10
    p0_star: float = 0.8
    temperature: float = 0.7


class OnlineRunner:
    def __init__(self, target: CachedClient, generator: CachedClient, judge: CachedClient,
                 cfg: OnlineConfig = OnlineConfig()):
        self.cfg = cfg
        self.c = Collector(target, generator, judge,
                           CollectConfig(n_paraphrases=cfg.deep_draws, temperature=cfg.temperature))
        self.calls = 0

    def _draws(self, item: Item, paras: list[str], start: int, stop: int):
        self.calls += stop - start
        return [self.c._draw(item, P.answer_prompt(paras[j]), f"para:{j}", 0, self.cfg.temperature)
                for j in range(start, stop)]

    def _probe(self, item: Item, g: str):
        self.calls += 1
        build = {"neg": (P.negation_prompt(item.question), "yn"),
                 "relabel": (P.relabel_prompt(item.question), "ab"),
                 "distractor": (P.distractor_prompt(item.question, item.item_id), "yn")}[g]
        return self.c._draw(item, build[0], g, 0, 0.0, fmt=build[1]).answer

    def run_batch(self, items: list[Item]) -> dict:
        cfg, rng = self.cfg, random.Random(0)
        state = {}
        for it in items:                                    # stage 1: screen
            paras = self.c.paraphrases(it)
            self.calls += 1
            d = self._draws(it, paras, 0, cfg.s1)
            a, m, n = majority([x.answer for x in d])
            state[it.item_id] = {"item": it, "paras": paras, "draws": d,
                                 "alive": a is not None and m == n}
        survivors = [s for s in state.values() if s["alive"]]
        for s in survivors:                                 # lean stage 2
            s["draws"] += self._draws(s["item"], s["paras"], cfg.s1, cfg.s1 + cfg.lean_draws)
            ans, m, k = majority([x.answer for x in s["draws"]])
            s["passes"] = {g: (lambda pa: None if pa is None or ans is None
                               else pa == LABEL_ACTION[g](ans))(self._probe(s["item"], g))
                           for g in cfg.probes}
            s["llr"] = evidence_llr(m, k, s["passes"])
        survivors.sort(key=lambda s: -s["llr"])
        for s in survivors[:cfg.n_contenders]:             # deepen contenders only
            have = len(s["draws"])
            s["draws"] += self._draws(s["item"], s["paras"], have, cfg.deep_draws)
            labels = self.c._cluster(self.c.j, s["item"], [x.explanation for x in s["draws"]])
            self.calls += 1
            for x, lab in zip(s["draws"], labels):
                x.cluster = lab
            ans, m, k = majority([x.answer for x in s["draws"]])
            s.update(ans=ans, m=m, k=k,
                     sig={"L": lower_bound(m, k),
                          "flip": sum(bool(v) for v in s["passes"].values()) / len(cfg.probes),
                          "se": miller_madow_entropy([x.cluster for x in s["draws"]]),
                          "llr": evidence_llr(m, k, s["passes"])})
        contenders = [s for s in survivors[:cfg.n_contenders]]
        cases = [CaseScore(s["item"].item_id, s["ans"], False, 0, signals=s["sig"])
                 for s in contenders]
        scores = borda_scores(cases, ("L", "flip", "se", "llr"))
        top = select_top_k(scores, min(cfg.k, len(cases)), rng)
        selected = []
        for t in top:
            s = contenders[t]
            eps = tail_mass(s["m"], s["k"], cfg.p0_star)
            eps_eff = (tail_mass_effective(k_effective(s["m"], s["passes"]), cfg.p0_star)
                       if s["m"] == s["k"] else eps)
            led = EvidenceLedger()
            for x in s["draws"]:
                led.draw(x.answer is not None and x.answer == s["ans"])
            for g, ok in s["passes"].items():
                led.probe(g, ok)
            selected.append({"item_id": s["item"].item_id, "answer": s["ans"],
                             "ledger": led.statement(),
                             "agree": f"{s['m']}/{s['k']}", "eps": eps, "eps_eff": eps_eff,
                             "signals": s["sig"]})
        return {"selected": selected,
                "guarantee_exact": selection_guarantee(x["eps"] for x in selected),
                "guarantee_model_based": selection_guarantee(x["eps_eff"] for x in selected),
                "calls": self.calls, "calls_per_case": self.calls / max(1, len(items)),
                "survival": len(survivors) / max(1, len(items))}


# --- test (was in tests/test_pipeline.py; needs the `world` fixture there) ---
# def test_online_runner(world):
#     r = OnlineRunner(world["t"], world["g"], world["j"], OnlineConfig(k=5, n_contenders=8))
#     out = r.run_batch(world["split"].eval[:20])
#     assert len(out["selected"]) <= 5
#     assert all(0 <= s["eps"] <= 1 for s in out["selected"])
#     assert out["calls_per_case"] < 15
#     assert all("ledger" in s and "balance_nats" in s["ledger"] for s in out["selected"])
