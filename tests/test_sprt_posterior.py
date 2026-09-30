"""Every constant quoted in the paper, checked against the implementation."""
import math

import pytest

from confledger.sprt_posterior import (SPRT, certification_depth, design_effect, k_effective, kappa,
                             lower_bound, majority, miller_madow_entropy,
                             screen_pass_probability, selection_guarantee, tail_mass,
                             tail_mass_effective, truncated_oc, certification_plan)


def test_sprt_increments_and_boundaries():
    s = SPRT()
    assert s.up == pytest.approx(0.305, abs=1e-3)
    assert s.down == pytest.approx(-1.792, abs=1e-3)
    assert s.A == pytest.approx(2.197, abs=1e-3) and s.B == pytest.approx(-2.197, abs=1e-3)
    assert s.decision(3, 3) == "continue"            # stage 1 never certifies (Sec. 3.2)
    assert s.decision(8, 8) == "certify"
    assert s.decision(1, 3) == "eliminate"


def test_lemma3_wald_expected_n():
    e = SPRT().expected_n()
    assert e["H0"] == pytest.approx(5.4, abs=0.05)
    assert e["H1"] == pytest.approx(8.8, abs=0.05)


def test_truncated_sprt_operating_characteristic():
    # README "things the dry run surfaced": 3+6 draws is conservative, not nominal
    s = SPRT()
    assert truncated_oc(s.p0)["certify"] == pytest.approx(0.040, abs=1e-3)        # nominal 0.10
    assert 1 - truncated_oc(s.p1)["certify"] == pytest.approx(0.370, abs=1e-3)    # nominal 0.10
    assert truncated_oc(s.p1, s2=2)["certify"] == 0                                 # 3+2 cannot reach A
    o = truncated_oc(0.8)
    assert o["eliminate"] + o["certify"] + o["undecided"] == pytest.approx(1)


def test_lemma3a_screen():
    assert screen_pass_probability(0.7) == pytest.approx(0.343)
    assert screen_pass_probability(0.95) == pytest.approx(0.857, abs=1e-3)


def test_lemma2_design_effect():
    assert design_effect(1, 0.3) == 1.0
    assert design_effect(3, 0.3) == pytest.approx(1.6)


def test_posterior_constants():
    assert tail_mass(9, 9) == pytest.approx(0.042, abs=1e-3)
    assert tail_mass(12, 12) == pytest.approx(0.019, abs=1e-3)
    assert tail_mass(8, 9) == pytest.approx(0.27, abs=5e-3)
    assert lower_bound(9, 9) == pytest.approx(0.813, abs=1e-3)


def test_corollary1_certification_depth():
    assert certification_depth(0.05) == 9
    assert certification_depth(0.02) == 12
    assert certification_depth(0.05, prior="uniform") == 13
    assert certification_depth(0.02, prior="uniform") == 17


def test_prop3_exchange_rates():
    assert kappa(0.5) == pytest.approx(2.10, abs=0.01)
    assert kappa(0.8) == pytest.approx(0.56, abs=0.01)
    passes = {"neg": True, "relabel": True, "distractor": True}
    assert k_effective(6, passes) == pytest.approx(10.77, abs=0.01)
    assert tail_mass_effective(k_effective(6, passes)) == pytest.approx(0.027, abs=1e-3)


def test_prop2_guarantee():
    g = selection_guarantee([0.019] * 10)
    assert g["expected_substandard"] == pytest.approx(0.19)


def test_majority_conventions():
    assert majority(["yes", "yes", "no"]) == ("yes", 2, 3)
    assert majority(["yes", None, "yes"]) == ("yes", 2, 3)   # None never agrees
    assert majority([None, None]) == (None, 0, 2)
    assert majority(["no", "yes"])[0] == "no"                # tie -> earliest


def test_miller_madow():
    assert miller_madow_entropy([0, 0, 0, 0]) == 0.0
    h = miller_madow_entropy([0, 0, 1, 1])
    assert h == pytest.approx(math.log(2) + 1 / 8)
    assert miller_madow_entropy([None, None]) > 0            # missing labels are singletons


def test_evidence_ledger_matches_prop3():
    from confledger.sprt_posterior import EvidenceLedger
    led = EvidenceLedger()
    for _ in range(6):
        led.draw(True)
    assert not led.certified                      # 6 draws: 1.83 nats < A = 2.20
    for g in ("neg", "relabel", "distractor"):
        led.probe(g, True)
    assert led.k_eff == pytest.approx(10.77, abs=0.01)
    assert led.certified and led.calls == 9
    led2 = EvidenceLedger()
    led2.draw(True); led2.draw(False)
    assert not led2.closed                        # one disagreement: -1.49, near but above B
    led2.draw(False)
    assert led2.closed                            # a second one closes the account


def test_certification_plan_depth():
    # paper defaults: SPRT alone certifies at 8/8, Cor. 1 (eps <= 0.05 at 0.8) needs 9 -> 3 + 6
    p = certification_plan()
    assert (p["sprt_depth"], p["cor1_depth"], p["depth"], p["s2"]) == (8, 9, 9, 6)
    # widening H0 alone saves nothing: Cor. 1 still binds
    assert certification_plan(SPRT(p0=0.6))["depth"] == 9
    # wider gap + relaxed eps0 -> 5/5 (stage 2 of 2 draws)
    p = certification_plan(SPRT(p0=0.6), eps0=0.13)
    assert (p["depth"], p["s2"], p["binding"]) == (5, 2, "SPRT")


def test_certification_budget_constants():
    """Sec. 3.9 / Cor. 2: every number quoted there."""
    from itertools import combinations
    from scipy.optimize import brentq
    from confledger.sprt_posterior import DEFAULT_Q0, kappa, tail_mass, tail_mass_effective
    assert [round(tail_mass(d, d), 2) for d in (1, 2, 3, 5)] == [0.45, 0.31, 0.23, 0.13]
    assert round(brentq(lambda x: tail_mass_effective(x) - 0.05, 1, 20), 2) == 8.35
    K = {g: kappa(DEFAULT_Q0[g]) for g in ("neg", "relabel", "distractor")}

    def least_calls(shrink):
        return min(d + r for d in range(1, 13) for r in range(4) for G in combinations(K, r)
                   if tail_mass_effective(d + shrink * sum(K[g] for g in G)) <= 0.05)
    assert least_calls(1.0) == 7 and least_calls(0.5) == 9            # kappa/2: no saving over 9 draws
    ke = 4 + sum(K.values())
    assert round(ke, 1) == 8.8 and round(tail_mass_effective(ke), 3) == 0.045
