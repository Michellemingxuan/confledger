"""Fitted stability prior, GLR stopping, SHAP values and the Prop. K knapsack."""
import numpy as np
import pytest

from confledger.methods import CaseScore
from confledger.score_model import fit_score_model
from confledger.signal_value import shap_importance
from confledger.stability_prior import (fit_beta_binomial, fitted_prior, glr_decision, glr_depth,
                                        prior_depth)


def test_prior_depths_match_the_method_notes():
    # Jeffreys 9 (= Cor. 1), uniform 13, Beta(8,1) 6, Beta(4,1) 10, Beta(2,1) 12
    assert [prior_depth(*ab) for ab in [(0.5, 0.5), (1, 1), (8, 1), (4, 1), (2, 1)]] == [9, 13, 6, 10, 12]


def test_glr_depths():
    assert (glr_depth(0.1), glr_depth(0.05)) == (25, 29)
    assert (glr_depth(0.1, disagreements=1), glr_depth(0.05, disagreements=1)) == (43, 46)
    assert glr_decision(24, 24) == "continue" and glr_decision(25, 25) == "certify"
    assert glr_decision(0, 30) == "reject"


def test_beta_binomial_fit_and_pessimistic_depth():
    rng = np.random.default_rng(0)
    m = rng.binomial(5, rng.beta(8, 1, 2000))
    a, b = fit_beta_binomial(m, 5)
    assert a / (a + b) == pytest.approx(8 / 9, abs=0.02)
    fp = fitted_prior(m[:200], 5, n_boot=200)
    assert fp["depth"] >= fp["depth_point"] and fp["depth"] <= 9   # between the point estimate and Jeffreys here
    assert fit_beta_binomial(np.full(50, 5), 5)[1] < 0.01           # all unanimous: finite, b -> 0


def test_shap_closed_form_sums_to_logit_deviation():
    rng = np.random.default_rng(0)
    cases = []
    for i in range(300):
        z = bool(rng.random() < 0.7)
        m = int((rng.random(4) < (0.95 if z else 0.7)).sum())
        cases.append(CaseScore(f"c{i}", "yes", z, 5, True, signals={"m": m, "k": 4},
                               probe_passes={"neg": bool(rng.random() < (0.95 if z else 0.5))}))
    model = fit_score_model(cases, [c.correct for c in cases])
    imp = shap_importance(model, cases)
    assert imp["eta"] > 0 and imp["neg"] > 0 and imp["chal"] == 0     # chal never bought: no importance


def test_knapsack_buys_in_value_order_within_budget():
    from confledger.signal_value import knapsack_allocation
    from confledger.utility.records import CaseData, Draw
    fit = [CaseData(f"c{i}", "x", "yes", [Draw("yes")] * 12, [Draw("yes")] * 12, [], {}) for i in range(10)]
    v = {"draw": 0.1, "neg": 0.5, "relabel": 0.4, "distractor": 0.05, "chal": 0.0, "se": 0.9}
    a = knapsack_allocation(v, fit, 3, paras=(False,))["allocation"]
    # every screen survives (sigma = 1): k1 = 1 -> neg, relabel (SE excluded with 1 draw);
    # k1 = 2 -> no plan with 3 calls has an odd total of draws and beats it (decisions.md #34)
    assert (a.k1, a.se, a.flips) == (1, False, ("neg", "relabel"))
    a = knapsack_allocation(v, fit, 2, paras=(False,))["allocation"]
    assert (a.k1, a.flips) == (1, ("neg",))
