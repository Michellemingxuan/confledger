"""Score model (pessimistic score) and certified selection (conformal BH, e-BH)."""
import numpy as np
import pytest

from confledger.certified_selection import (bh, conformal_pvalues, ebh, evalue, pass_rates,
                                            selection_outcome)
from confledger.methods import CaseScore
from confledger.score_model import choose_lambda, features, fit_score_model


def _case(m, n, correct=True, survived=True, probes=None, i=0):
    return CaseScore(f"c{i}", "yes", correct, n, survived, signals={"m": m, "k": n},
                     probe_passes=probes or {})


def test_conformal_pvalue_formula():
    s_cal = np.array([0.1, 0.5, 0.9, 0.5])
    z_cal = np.array([False, False, True, True])
    p = conformal_pvalues(np.array([0.0, 0.5, 1.0]), s_cal, z_cal)
    assert p.tolist() == pytest.approx([3 / 5, 2 / 5, 1 / 5])     # (1 + #incorrect with s >= s_j) / 5


def test_bh_and_ebh_toy():
    assert sorted(bh(np.array([0.01, 0.02, 0.5, 0.04]), 0.2).tolist()) == [0, 1, 3]
    assert bh(np.array([0.9, 0.8]), 0.05).size == 0
    # m = 4, alpha = 0.5: e_(r) >= 8 / r -> r = 2 (e = 5 >= 4), not 3 (2 < 8/3)
    assert sorted(ebh(np.array([10.0, 5.0, 2.0, 0.5]), 0.5).tolist()) == [0, 1]


def test_conformal_bh_controls_fdr():
    """Exchangeable calibration and test cases, an arbitrary (weak) score: E[FDP] <= alpha."""
    rng = np.random.default_rng(1)
    alpha, fdp, power = 0.1, [], []
    for _ in range(400):
        x = rng.normal(size=250)
        z = rng.random(250) < 1 / (1 + np.exp(-(1.0 + 1.5 * x)))
        s = x + rng.normal(scale=0.7, size=250)
        sel = bh(conformal_pvalues(s[200:], s[:200], z[:200]), alpha)
        o = selection_outcome(sel, z[200:])
        fdp.append(o["fdp"])
        power.append(o["power"])
    assert np.mean(fdp) <= alpha + 0.01 and np.mean(power) > 0.05


def test_evalue_is_supermartingale_under_h0():
    """Incorrect answers that pass every signal at exactly q0: mean e-value <= 1."""
    rates = {"draw": {"q0": 0.8, "q1": 0.97}, "neg": {"q0": 0.5, "q1": 0.95}}
    rng = np.random.default_rng(0)
    es = []
    for _ in range(20000):
        m = int((rng.random(6) < 0.8).sum())
        es.append(evalue(_case(m, 6, False, probes={"neg": bool(rng.random() < 0.5)}), rates))
    assert np.mean(es) == pytest.approx(1.0, abs=0.05)


def test_pass_rates_respect_theory_anchor():
    cases = [_case(6, 6, False, probes={"neg": False}, i=i) for i in range(30)] + \
            [_case(6, 6, True, probes={"neg": True}, i=100 + i) for i in range(30)]
    r = pass_rates(cases, [c.correct for c in cases])
    assert r["neg"]["q0"] == 0.5                       # data say ~0, the anchor keeps 0.5
    assert r["draw"]["q0"] > 0.9                       # stable wrong answers: draws are weak evidence


def _world(rng, n_cases, depths):
    """Cases with latent accuracy p; correct ones agree more often across draws."""
    out = []
    for i in range(n_cases):
        n = int(rng.choice(depths))
        z = bool(rng.random() < 0.7)
        m = int((rng.random(n) < (0.95 if z else 0.6)).sum())
        out.append(_case(m, n, z, i=i))
    return out


def test_pessimism_is_inert_under_equal_evidence():
    """Prop. P1: equal depth n, stability only -> among cases with mu >= 1/2 (where a top-k
    cut falls), mu - lambda*sigma orders cases as mu does, so the top-k cannot change.
    Below 1/2 it can reorder, which never matters for the top-k."""
    rng = np.random.default_rng(0)
    fit = _world(rng, 300, [6])
    model = fit_score_model(fit, [c.correct for c in fit])
    mu, sd = model.mu_sigma([_case(m, 6) for m in range(0, 7)])
    assert np.all(np.diff(mu) > 0)
    top = mu >= 0.5                                     # where a top-k cut falls
    assert top.sum() >= 2
    for lam in (1.0, 2.0, 3.0):
        assert np.all(np.diff((mu - lam * sd)[top]) > 0)


def test_sigma_shrinks_with_evidence():
    """Prop. P2's mechanism: unanimous 2/2 is less certain than unanimous 8/8."""
    rng = np.random.default_rng(0)
    fit = _world(rng, 300, [2, 8])
    model = fit_score_model(fit, [c.correct for c in fit])
    _, sd = model.mu_sigma([_case(2, 2), _case(8, 8)])
    assert sd[0] > sd[1]
    assert features(_case(2, 2))[1] > features(_case(8, 8))[1]


def test_choose_lambda_prefers_zero_on_ties():
    rng = np.random.default_rng(0)
    fit = _world(rng, 100, [6])
    lam, table = choose_lambda(fit, [c.correct for c in fit], [0, 1, 2], top_frac=0.2)
    assert set(table) == {0.0, 1.0, 2.0}
    assert table[lam] == max(table.values()) and (lam == 0 or table[lam] > table[0.0])


def test_unbought_signal_widens_sigma():
    """Step B4's third term: a probe not bought adds beta^2 * Var(x) to sigma^2."""
    rng = np.random.default_rng(0)
    fit = []
    for i in range(400):
        z = bool(rng.random() < 0.7)
        m = int((rng.random(4) < (0.95 if z else 0.7)).sum())
        probes = {"neg": bool(rng.random() < (0.95 if z else 0.5))} if rng.random() < 0.5 else {}
        fit.append(_case(m, 4, z, probes=probes, i=i))
    model = fit_score_model(fit, [c.correct for c in fit])
    assert model.coef[3] > 0 and model.var[3] > 0             # neg is informative and varies
    _, sd = model.mu_sigma([_case(4, 4), _case(4, 4, probes={"neg": True})])
    assert sd[0] > sd[1]


def test_stability_coefficient_is_never_negative():
    """Disagreement cases mostly correct, unanimous ones mostly wrong: the raw fit is negative."""
    cases = [_case(1, 2, i=i) for i in range(40)] + [_case(2, 2, i=i) for i in range(40, 80)]
    z = [i % 10 != 0 for i in range(40)] + [i % 2 == 0 for i in range(40)]
    m = fit_score_model(cases, z)
    assert abs(m.coef[1]) < 1e-6
    mu, _ = m.mu_sigma([_case(1, 2), _case(2, 2)])
    assert mu[1] >= mu[0] - 1e-6


def test_allocation_space_has_odd_draws_and_keeps_adaptive_cel():
    from confledger.methods import allocation_space, cel_as_allocation
    space = allocation_space(verbal_per_draw=True)
    assert len(space) == 864 and all((a.k1 + a.extra) % 2 == 1 for a in space)
    assert cel_as_allocation() in space
