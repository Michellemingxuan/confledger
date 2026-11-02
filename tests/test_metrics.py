import random

import numpy as np
import pytest

from confledger import metrics as M
from confledger.signal_shapley import exact_shapley


def test_auroc_worked_example():
    # A..E ranks 5..1 with Z = 1,1,0,1,0 -> 5/6 (the example from the design notes)
    assert M.auroc([5, 4, 3, 2, 1], [1, 1, 0, 1, 0]) == pytest.approx(5 / 6)
    assert M.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == pytest.approx(0.5)   # ties count half


def test_prr_bounds():
    z = [1, 1, 1, 0, 0, 1, 0, 1]
    assert M.prr(z, z) == pytest.approx(1.0)
    assert M.prr([0] * len(z), z) == pytest.approx(0.0, abs=1e-12)


def test_hypergeom_null():
    assert M.hypergeom_expected_precision(40, 50, 10) == pytest.approx(0.8)
    assert M.hypergeom_pvalue(10, 40, 50, 10) == pytest.approx(0.0826, abs=1e-3)


def test_jaccard_floor():
    assert M.jaccard_chance_floor(50, 10) == pytest.approx(0.116, abs=0.005)


def test_pav_monotone():
    fit = M.pav([0.9, 0.95, 0.7, 0.8, 0.5])
    assert all(a >= b - 1e-12 for a, b in zip(fit, fit[1:]))


def test_paired_compare_and_holm():
    rng = np.random.default_rng(0)
    s, z = rng.random(400), rng.random(400) < 0.7
    good = M.prec_rep(s + 2 * z, z, 0.1)          # score that knows the answer
    rand = M.prec_rep(s, z, 0.1)                  # score unrelated to correctness
    r = M.paired_compare(good, rand)
    assert good[0] == pytest.approx(1.0) and r["mean_diff"] > 0.2 and r["p_boot"] < 0.01
    assert r["ci_lo"] < r["mean_diff"] < r["ci_hi"]
    assert M.paired_compare(rand, rand)["p_boot"] == 1.0
    assert M.holm({"a": 0.01, "b": 0.04})["b"] == pytest.approx(0.04)


def test_empirical_kappa_recovers_rate():
    rng = np.random.default_rng(0)
    n = 4000
    m = rng.integers(3, 10, n)
    c = rng.integers(0, 2, n)
    logit = -4 + 0.5 * m + 1.0 * c          # true kappa = 1.0 / 0.5 = 2
    z = rng.random(n) < 1 / (1 + np.exp(-logit))
    out = M.empirical_kappa(m, {"neg": c}, z, n_boot=50)
    assert out["kappa_hat"]["neg"] == pytest.approx(2.0, abs=0.4)


def test_shapley_worked_example_and_efficiency():
    V = {frozenset(): 0.60, frozenset("L"): 0.80, frozenset(["S"]): 0.78, frozenset("F"): 0.70,
         frozenset(["L", "S"]): 0.82, frozenset(["L", "F"]): 0.88, frozenset(["S", "F"]): 0.86,
         frozenset(["L", "S", "F"]): 0.90}
    out = exact_shapley(["L", "S", "F"], lambda C: V[C])
    assert out["phi"]["L"] == pytest.approx(0.1167, abs=1e-3)
    assert out["phi"]["S"] == pytest.approx(0.0967, abs=1e-3)
    assert out["phi"]["F"] == pytest.approx(0.0867, abs=1e-3)
    assert abs(out["efficiency_check"]) < 1e-12


def test_horizontal_gap():
    curve = {"cel:cel": {"calls": 5.0, "prec": 0.8},
             "sc:3": {"calls": 3.0, "prec": 0.6}, "sc:9": {"calls": 9.0, "prec": 0.9}}
    assert M.horizontal_gap(curve) == pytest.approx((3 + 0.2 / 0.3 * 6) / 5)   # 7 calls / 5
    assert M.horizontal_gap({**curve, "cel:cel": {"calls": 5.0, "prec": 0.95}}) == float("inf")


def test_calibration_metrics():
    z = [1, 0, 1, 1, 0, 1, 0, 1]
    assert M.ece([1, 0, 1, 1, 0, 1, 0, 1], z) == 0 and M.brier(z, z) == 0
    # all p = 0.5 in one bin, accuracy 5/8 -> ECE = 1/8, Brier = 1/4
    assert M.ece([0.5] * 8, z) == pytest.approx(1 / 8) and M.brier([0.5] * 8, z) == pytest.approx(0.25)
    # AUARC: perfect ranking -> mean of retained accuracies; constant score -> base rate
    s = [8, 7, 6, 5, 4, 3, 2, 1]
    zz = [1, 1, 1, 1, 1, 0, 0, 0]
    assert M.auarc(s, zz) == pytest.approx(np.mean([min(n, 5) / n for n in range(1, 9)]))
    assert M.auarc([0] * 8, zz) == pytest.approx(5 / 8)
    c = M.calibration(np.linspace(0.05, 0.95, 40), [i % 2 for i in range(40)], n_bins=4)
    assert len(c["reliability"]) == 4 and sum(r["n"] for r in c["reliability"]) == 40


def test_wqd():
    # one question: correct confs 0.9, 0.6; incorrect 0.7 -> 1 win of 2 pairs
    assert M.wqd([([0.9, 0.6, 0.7], [1, 1, 0]), ([0.5, 0.5], [1, 1])]) == pytest.approx(0.5)
    assert np.isnan(M.wqd([([0.5], [1])]))


def test_top_shares_reads_each_cut():
    from types import SimpleNamespace
    from confledger.evaluation_protocol import _top_shares
    z = [True] * 10 + [False] + [True] * 89
    run = {"_scores": np.arange(100, 0, -1, dtype=float), "_cases": [SimpleNamespace(correct=c) for c in z]}
    t = _top_shares([run], [0.05, 0.2])
    assert t["prec_0.05"] == 1.0 and t["prec_0.2"] == pytest.approx(0.95)
    assert t["lo_0.2"] <= t["prec_0.2"] <= t["hi_0.2"]
