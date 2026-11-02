"""Baselines: formulas on hand-built cases, call accounting, D_fit fitting."""
import numpy as np
import pytest

from confledger.baselines import bsdetector_draws, bsdetector_like, cisc, fit_cisc, fit_pedapati, pedapati, self_consistency, xiong
from confledger.baselines.cisc import vote
from confledger.baselines.pedapati import rouge_l, split
from confledger.utility.records import NO, YES, CaseData, Draw


def _cd(answers, confs, gold=YES, iid="q"):
    same = [Draw(a, f"because reason {i % 2}", confidence=c) for i, (a, c) in enumerate(zip(answers, confs))]
    return CaseData(iid, "test", gold, list(same), same, [], {}, meta={"verbal_per_draw": True})


def test_xiong_avg_conf():
    cd = _cd([YES, YES, NO], [90, 60, 50])
    s = xiong(cd, 3)
    assert s.answer == YES and s.calls == 3
    assert s.signals["avg_conf"] == pytest.approx((0.9 + 0.6) / 2.0)
    assert xiong(cd, 3, "prompting").calls == 3                 # 1 generator + 2 paraphrase draws


def test_cisc_vote():
    cd = _cd([YES, NO, NO], [99, 40, 40])
    assert vote(cd.same, 1e4, "verbal")[0] == NO                 # T -> inf: majority (self-consistency)
    assert vote(cd.same, 1e-3, "verbal")[0] == YES               # T -> 0: most confident draw wins
    assert cisc(cd, 1e4, 3).signals["cisc"] == pytest.approx(2 / 3, abs=1e-3)
    fit = [_cd([YES, NO, NO], [99, 40, 40], iid=f"a{i}") for i in range(5)]
    assert fit_cisc(fit, 3)["fit_acc"] == 1.0                    # a small T recovers the gold answer


def test_pedapati_features_and_fit():
    assert rouge_l("the cat sat", "the cat sat") == 1.0 and rouge_l("a b", "c d") == 0.0
    assert split(3) == (3, 0) and split(8) == (4, 3)
    fit = [_cd([YES] * 8, [80] * 8, iid=f"r{i}") for i in range(6)] + \
          [_cd([YES, NO] * 4, [60] * 8, gold=NO if i % 2 else YES, iid=f"w{i}") for i in range(6)]
    f = fit_pedapati(fit, 8)
    s = pedapati(fit[0], f)
    assert s.calls == 8 and 0 < s.signals["p_lr"] < 1
    assert s.signals["p_lr"] > pedapati(fit[-1], f).signals["p_lr"]     # unanimous ranks above split
    assert self_consistency(fit[0], 8).calls == 8


def test_bsdetector_spends_exactly_the_budget():
    """Decisions.md #37: with confidence on every draw BSDetector's reflection certainty is free,
    so a budget of B buys B draws; without it, B - 1 draws plus one verbal call."""
    from confledger.methods import registry
    cd = _cd([YES, YES, NO, YES, YES], [90, 80, 60, 85, 95])
    for B in (2, 3, 5):
        assert bsdetector_like(cd, bsdetector_draws(B, True)).calls == B
        assert registry(B, verbal_per_draw=True)["bsdetector_like"](cd).calls == B
    cd_free = _cd([YES, YES, NO, YES, YES], [90, 80, 60, 85, 95])
    cd_free.meta["verbal_per_draw"] = False
    cd_free.verbal_conf = 70.0
    for B in (2, 3, 5):
        assert bsdetector_like(cd_free, bsdetector_draws(B, False)).calls == B
