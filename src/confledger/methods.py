"""Every method as a replay policy over a CaseData record.

A policy returns a CaseScore: the answer it would output, whether that answer is
correct (Z, used only at evaluation), the signals it computed, the ranking key,
and -- crucially -- the number of API calls it *would have spent* online. That
call count is what makes matched-budget comparisons honest.
"""
from __future__ import annotations

import random
from itertools import combinations
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from .sprt_posterior import (SPRT, EvidenceLedger, certification_plan, evidence_llr, lower_bound, majority,
                    miller_madow_entropy)
from .utility.answer_parsing import has_hedge, rationale_words
from .utility.records import YES, CaseData, Draw, flip

# One generator call per case produces the paraphrases every fresh (and reuse) draw is
# asked on; it happens before the screen, so every paraphrase-based policy pays it
# (decisions.md #8). Same-prompt baselines (self-consistency etc.) do not.
PARAPHRASE_GEN_CALLS = 1

LABEL_ACTION = {"neg": flip, "relabel": lambda a: a, "distractor": lambda a: a,
                "heldout": lambda a: a}


@dataclass
class CaseScore:
    item_id: str
    answer: Optional[str]
    correct: bool
    calls: int
    survived: bool = True
    s1_agree: int = 0
    signals: dict = field(default_factory=dict)
    key: tuple = ()
    heldout_pass: Optional[bool] = None
    probe_passes: dict = field(default_factory=dict)


def _perm(draws: list[Draw], seed: Optional[int], item_id: str) -> list[Draw]:
    if seed is None:
        return list(draws)
    out = list(draws)
    random.Random(f"{seed}|{item_id}").shuffle(out)
    return out


def _passes(cd: CaseData, ans: Optional[str], names) -> dict:
    res = {}
    for g in names:
        pa = cd.probes.get(g)
        res[g] = None if (pa is None or ans is None) else (pa == LABEL_ACTION[g](ans))
    return res


def free_signals(cd: CaseData, used: Sequence[Draw], ans: Optional[str], sig: dict, passes: dict) -> None:
    """Signals that cost no extra call, added to `sig` in place:
    length    mean words of the draws' rationales
    hedge     share of rationales with a hedge word ("might", "not sure", ...)
    neg_drop  verbal confidence of the draws minus confidence under negation (needs the
              negation probe bought and confidence on both; collect.verbal_per_draw)
    p_first   mean P(answer) at the answer token (collect.logprobs; GRAY-BOX, used only
              with scoring.gray_box)."""
    expl = [d.explanation for d in used if d.answer is not None]
    if expl:
        sig["length"] = sum(rationale_words(e) for e in expl) / len(expl)
        sig["hedge"] = sum(has_hedge(e) for e in expl) / len(expl)
    ps = [d.p_yes for d in used if d.p_yes is not None]
    if ps and ans is not None:
        sig["p_first"] = sum(p if ans == YES else 1 - p for p in ps) / len(ps)
    nc = cd.probe_conf.get("neg")
    if passes.get("neg") is not None and nc is not None and "verb" in sig:
        sig["neg_drop"] = sig["verb"] - nc


# ---------------------------------------------------------------------------
# The method (Sec. 3) and its ablations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CELConfig:
    s1: int = 3                                   # stage-1 screen draws
    s2: int = 6                                   # stage-2 additional draws
    spread: bool = True                           # fresh paraphrase per draw (Lemma 1)
    screen: bool = True                           # truncated SPRT triage (Lemma 3)
    probes: tuple = ("neg", "relabel", "distractor")
    use_se: bool = True                           # rationale entropy (1 judge call)
    use_chal: bool = True
    use_verb: bool = True
    use_L: bool = True                            # lower-bound tie-break; the draws stay (SPRT needs them)
    alpha: float = 0.05                           # lower-bound level for signal (1)


def _reuse_order(cd: CaseData) -> list[Draw]:
    """No-spread ablation: 3 paraphrases x 3 samples, one per paraphrase first."""
    rows = cd.reuse
    return [rows[p][r] for r in range(len(rows[0])) for p in range(len(rows))]


def cel(cd: CaseData, cfg: CELConfig = CELConfig(), seed: Optional[int] = None,
         sprt: SPRT = SPRT()) -> CaseScore:
    pool = _perm(cd.fresh, seed, cd.item_id) if cfg.spread else _reuse_order(cd)
    s1 = [d.answer for d in pool[:cfg.s1]]
    a1, m1, n1 = majority(s1)
    calls = PARAPHRASE_GEN_CALLS + cfg.s1
    unanimous1 = a1 is not None and m1 == n1
    if cfg.screen and not unanimous1:
        return CaseScore(cd.item_id, a1, a1 == cd.gold, calls, survived=False, s1_agree=m1,
                         signals={"m": m1, "k": n1}, key=(0, m1, 0, 0, 0, 0, 0),
                         heldout_pass=None if a1 is None else cd.probes.get("heldout") == a1)
    used = pool[:cfg.s1 + cfg.s2]
    calls += max(0, len(used) - cfg.s1)
    ans, m, k = majority([d.answer for d in used])
    passes = _passes(cd, ans, cfg.probes)
    calls += len(cfg.probes)
    sig = {"m": m, "k": k}
    if cfg.use_L:
        sig["L"] = lower_bound(m, k, cfg.alpha)
    if cfg.probes:
        vals = [v for v in passes.values() if v is not None]
        sig["flip"] = sum(vals) / len(vals) if vals else 0.0
    if cfg.use_se:
        sig["se"] = miller_madow_entropy([d.cluster for d in used])
        calls += 1
    if cfg.use_chal:
        sig["chal"] = float(cd.challenge_answer is not None and cd.challenge_base == ans
                            and cd.challenge_answer == cd.challenge_base)
        calls += 1
    if cfg.use_verb:
        if cd.meta.get("verbal_per_draw"):        # mean confidence of the draws used: no call
            confs = [d.confidence for d in used if d.confidence is not None]
            sig["verb"] = sum(confs) / len(confs) if confs else 50.0
        else:                                     # separate verbal call
            sig["verb"] = cd.verbal_conf if cd.verbal_conf is not None else 50.0
            calls += 1
    sig["llr"] = evidence_llr(m, k, passes, sprt)
    key = (1, int(m == k),
           int(sig.get("flip", 1.0) >= 2 / 3),
           -sig.get("se", 0.0), sig.get("chal", 0.0), sig.get("verb", 0.0), sig.get("L", 0.0))
    return CaseScore(cd.item_id, ans, ans == cd.gold, calls, True, m1, sig, key,
                     heldout_pass=None if ans is None else cd.probes.get("heldout") == ans,
                     probe_passes=passes)



def ledger_statement(cd: CaseData, cfg: CELConfig = CELConfig()) -> dict:
    """Line-by-line evidence account for one collected case, as the pipeline would
    build it online: the s1+s2 draws against
    the majority answer, then each probe. Eliminated cases stop after the screen."""
    sc = cel(cd, cfg)
    led = EvidenceLedger()
    for d in cd.fresh[:cfg.s1 + (cfg.s2 if sc.survived else 0)]:
        led.draw(d.answer is not None and d.answer == sc.answer)
    for g, ok in sc.probe_passes.items():
        led.probe(g, ok)
    return {"answer": sc.answer, "survived": sc.survived, **led.statement()}

def ablations(base: CELConfig = CELConfig()) -> dict[str, CELConfig]:
    """CEL and its ablations, derived from `base` (whose s2 comes from the SPRT settings)."""
    return {
        "cel": base,
        "cel_no_spread": replace(base, spread=False),
        "cel_no_sprt": replace(base, screen=False, s1=2, s2=0),  # flat: paraphrase-gen + 2 draws + all probes = 9 calls
        "cel_no_flip": replace(base, probes=()),
        "cel_no_se": replace(base, use_se=False),
        "cel_no_chal": replace(base, use_chal=False),
        "cel_no_verb": replace(base, use_verb=False),
        "cel_no_L": replace(base, use_L=False),
    }


ABLATIONS = ablations()

def max_cel_calls(verbal_per_draw: bool = False, cfg: CELConfig = CELConfig()) -> int:
    """Calls a CEL case costs if it survives the screen: the upper bound of CEL's mean
    spend, hence of the matched budget. n_same must be at least this for a fair match."""
    return (PARAPHRASE_GEN_CALLS + cfg.s1 + cfg.s2 + len(cfg.probes) + cfg.use_se + cfg.use_chal
            + (cfg.use_verb and not verbal_per_draw))


# Leave-one-signal-out: the ablation that removes each of the five signals.
SIGNAL_ABLATIONS = {"L": "cel_no_L", "flip": "cel_no_flip", "se": "cel_no_se",
                    "chal": "cel_no_chal", "verb": "cel_no_verb"}

# ---------------------------------------------------------------------------
# Budget-allocated CEL ("CEL-k"): the k-call budget is spread, per batch on average,
# over a screen and the survivors' probes. An Allocation is chosen per budget on the DEV
# slice (evaluation_protocol.select_allocations) and compared with self-consistency with
# exactly k draws on EXPECTED calls per case. No certificate: the ranking mode.
# ---------------------------------------------------------------------------

FLIPS = ("neg", "relabel", "distractor")


@dataclass(frozen=True)
class Allocation:
    k1: int                        # screen draws; any disagreement eliminates the case
    para: bool = False             # draws on fresh paraphrases (+1 generator call per case)
    extra: int = 0                 # more draws for survivors
    flips: tuple = ()              # flip probes for survivors (subset of FLIPS), 1 call each
    chal: bool = False             # challenge for survivors, 1 call
    se: bool = False               # rationale entropy for survivors: 1 judge call, >= 2 draws
    verb: bool = False             # separate verbal call; free (always used) with verbal_per_draw

    def label(self) -> str:
        parts = [f"{self.k1} {'para' if self.para else 'orig'} draw{'s' * (self.k1 > 1)}"]
        if self.extra:
            parts.append(f"+{self.extra} draw{'s' * (self.extra > 1)}")
        parts += list(self.flips) + ["chal"] * self.chal + ["SE"] * self.se + ["verbal call"] * self.verb
        return parts[0] + (" → " + ", ".join(parts[1:]) if parts[1:] else "")


def cel_alloc(cd: CaseData, a: Allocation, seed: Optional[int] = None) -> CaseScore:
    """Replay one Allocation on a case. Ranked with Borda over its signals; eliminated
    cases sit below every survivor (ranking.borda_scores)."""
    pool = cd.fresh if a.para else cd.same
    calls = int(a.para) + a.k1
    a1, m1, n1 = majority([d.answer for d in pool[:a.k1]])
    if a1 is None or m1 < n1:                                   # screen: any disagreement
        return CaseScore(cd.item_id, a1, a1 == cd.gold, calls, survived=False, s1_agree=m1,
                         signals={"m": m1, "k": n1})
    used = pool[:a.k1 + a.extra]
    calls += len(used) - a.k1
    ans, m, n = majority([d.answer for d in used])
    sig = {"L": lower_bound(m, n), "m": m, "k": n}
    if cd.meta.get("verbal_per_draw"):                          # free: rides on every draw
        confs = [d.confidence for d in used if d.confidence is not None]
        sig["verb"] = sum(confs) / len(confs) if confs else 50.0
    elif a.verb:
        sig["verb"] = cd.verbal_conf if cd.verbal_conf is not None else 50.0
        calls += 1
    passes = _passes(cd, ans, a.flips)
    if a.flips:
        vals = [v for v in passes.values() if v is not None]
        sig["flip"] = sum(vals) / len(vals) if vals else 0.0
        calls += len(a.flips)
    if a.chal:
        sig["chal"] = float(cd.challenge_answer is not None and cd.challenge_base == ans
                            and cd.challenge_answer == cd.challenge_base)
        calls += 1
    if a.se and n >= 2:
        sig["se"] = miller_madow_entropy([d.cluster for d in used])
        calls += 1
    free_signals(cd, used, ans, sig, passes)
    return CaseScore(cd.item_id, ans, ans == cd.gold, calls, True, m1, sig, probe_passes=passes)


def cel_as_allocation(base: CELConfig = CELConfig(), verbal_per_draw: bool = True) -> Allocation:
    """Adaptive CEL as one point of CEL-k's allocation space (Sec. 3.9, nesting): same
    screen, same stage-2 draws and probes, same call count. The two differ only in the
    combiner (cascade key vs. Borda)."""
    return Allocation(k1=base.s1, para=base.spread, extra=base.s2, flips=tuple(base.probes),
                      chal=base.use_chal, se=base.use_se, verb=base.use_verb and not verbal_per_draw)


def evidence_depth(a: Allocation, kappas: dict) -> tuple[int, float]:
    """(draws, k_eff) of a unanimous survivor that passes every probe: the exact depth of
    Prop. 2 and the model-based depth of Prop. 3 (Eq. 12). Challenge and SE carry no
    derived exchange rate, so they add nothing here."""
    d = a.k1 + a.extra
    return d, d + sum(kappas[g] for g in a.flips)


def allocation_space(max_screen: int = 4, max_extra: int = 6, verbal_per_draw: bool = True,
                     max_draws: int = 12) -> list[Allocation]:
    """Every allocation the collection can replay: screen size, draw type, extra draws,
    any subset of the three flips, challenge, SE (only with >= 2 draws), verbal call.
    Odd total draws only, so a survivor's majority cannot be a tie (decisions.md #34)."""
    out = []
    for para in (False, True):
        for k1 in range(1, max_screen + 1):
            for extra in range(0, max_extra + 1):
                if k1 + extra > max_draws or (k1 + extra) % 2 == 0:
                    continue
                for r in range(len(FLIPS) + 1):
                    for flips in combinations(FLIPS, r):
                        for chal in (False, True):
                            for se in ((False, True) if k1 + extra >= 2 else (False,)):
                                for verb in ((False,) if verbal_per_draw else (False, True)):
                                    out.append(Allocation(k1, para, extra, flips, chal, se, verb))
    return out


# Budget-curve configurations (x-axis = realised mean calls per case).
def budget_configs(base: CELConfig = CELConfig()) -> dict[str, CELConfig]:
    """CEL at four spend levels (budget curve), around `base`'s certification depth."""
    return {
        "cel_b3": replace(base, s2=0, probes=(), use_se=False, use_chal=False, use_verb=False),
        "cel_lean": replace(base, s2=min(2, base.s2), use_chal=False, use_verb=False),
        "cel": base,
        "cel_rich": replace(base, s2=base.s2 + 3),
    }


BUDGET_CONFIGS = budget_configs()


def cel_from_settings(sprt_cfg: Optional[dict] = None) -> tuple[CELConfig, SPRT, dict]:
    """CEL's base config and SPRT from a config's `sprt:` block (paper defaults if absent).
    The stage-2 depth is derived by certification_plan, never set by hand."""
    c = {"p0": 0.7, "p1": 0.95, "alpha": 0.1, "beta": 0.1, "eps0": 0.05, "p0_star": 0.8, **(sprt_cfg or {})}
    sprt = SPRT(c["p0"], c["p1"], c["alpha"], c["beta"])
    plan = certification_plan(sprt, c["eps0"], c["p0_star"], CELConfig().s1)
    return replace(CELConfig(), s2=plan["s2"]), sprt, {**plan, "settings": c}


# ---------------------------------------------------------------------------
# Evidence-equivalence arms (Prop. 3)
# ---------------------------------------------------------------------------

def evidence_arm(cd: CaseData, n_draws: int, probes: tuple = (), shrink: float = 1.0,
                 seed: Optional[int] = None, sprt: SPRT = SPRT()) -> CaseScore:
    used = _perm(cd.fresh, seed, cd.item_id)[:n_draws]
    ans, m, k = majority([d.answer for d in used])
    passes = _passes(cd, ans, probes)
    lam = evidence_llr(m, k, passes, sprt, shrink=shrink)
    return CaseScore(cd.item_id, ans, ans == cd.gold, PARAPHRASE_GEN_CALLS + len(used) + len(probes),
                     signals={"llr": lam, "m": m, "k": k}, key=(lam,), probe_passes=passes)


ARMS = {
    "A_12draws": lambda cd, seed=None: evidence_arm(cd, 12, seed=seed),
    "B_6draws_battery": lambda cd, seed=None: evidence_arm(
        cd, 6, ("neg", "relabel", "distractor"), seed=seed),
    "B_half_kappa": lambda cd, seed=None: evidence_arm(
        cd, 6, ("neg", "relabel", "distractor"), shrink=0.5, seed=seed),
    "C_9draws": lambda cd, seed=None: evidence_arm(cd, 9, seed=seed),
}


def registry(budget: int = 8, deploy_sc_draws: int = 3, base: CELConfig = CELConfig(),
             sprt: SPRT = SPRT(), verbal_per_draw: bool = False) -> dict[str, Callable]:
    """Main comparison set at matched budget (Sec. 4.4), plus self-consistency at the
    cheap deployment setting (`deploy_sc_draws` draws; not budget-matched, reported with
    its own call count). Baselines live in baselines/; the fitted ones (CISC, Pedapati)
    are added by run_protocol once D_fit is known."""
    from .baselines import (bsdetector_draws, bsdetector_like, random_selection, self_consistency, semantic_entropy,
                            verbalized, xiong)
    methods: dict[str, Callable] = {
        "random": random_selection,
        "verbalized": verbalized,
        "self_consistency": lambda cd, seed=None: self_consistency(cd, budget, seed),
        "self_consistency_deploy": lambda cd, seed=None: self_consistency(cd, deploy_sc_draws, seed),
        "semantic_entropy": lambda cd, seed=None: semantic_entropy(cd, budget - 1, seed),
        "bsdetector_like": lambda cd, seed=None: bsdetector_like(cd, bsdetector_draws(budget, verbal_per_draw), seed=seed),
    }
    if verbal_per_draw:                            # Avg-Conf needs a confidence on every draw
        methods["xiong_avgconf"] = lambda cd, seed=None: xiong(cd, budget, "self_random", seed)
        methods["xiong_avgconf_para"] = lambda cd, seed=None: xiong(cd, budget, "prompting", seed)
    for name, cfg in ablations(base).items():
        methods[name] = (lambda c: (lambda cd, seed=None: cel(cd, c, seed, sprt)))(cfg)
    return methods

