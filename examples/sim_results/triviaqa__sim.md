# Results: triviaqa__sim

## Baseline ladder and ablations

| Method | prec@10 | calls/case | AUROC | PRR | Δ vs ours [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| random | 0.804 | 1.0 | 0.500 | 0.000 | -0.196 [-0.218, -0.174] | 0.002 |
| verbalized | 0.910 | 1.0 | 0.708 | 0.457 | -0.090 [-0.150, -0.040] | 0.031 |
| self_consistency | 0.880 | 12.0 | 0.601 | 0.221 | -0.120 [-0.180, -0.060] | 0.016 |
| semantic_entropy | 0.980 | 12.0 | 0.788 | 0.718 | -0.020 [-0.050, -0.000] | 0.500 |
| bsdetector_like | 0.920 | 12.0 | 0.653 | 0.381 | -0.080 [-0.120, -0.040] | 0.016 |
| ours | 1.000 | 11.6 | 0.644 | 0.479 | — | — |
| ours_no_spread | 0.990 | 11.6 | 0.674 | 0.543 | -0.010 [-0.030, -0.000] | 1.000 |
| ours_no_sprt | 0.960 | 8.0 | 0.792 | 0.704 | -0.040 [-0.100, -0.000] | 0.500 |
| ours_no_flip | 0.990 | 9.4 | 0.643 | 0.480 | -0.010 [-0.030, -0.000] | 1.000 |
| ours_borda | 1.000 | 11.6 | 0.718 | 0.630 | -0.000 [-0.000, -0.000] | 1.000 |

Confirmatory (ours − self-consistency): +0.120 [+0.060, +0.180], wins 7/10, p=0.016

## Budget curve

| Config | calls | prec@10 | AUROC |
|---|---|---|---|
| verbalized | 1.0 | 0.910 | 0.708 |
| ours:ours_b3 | 3.0 | 0.807 | 0.562 |
| sc:3 | 3.0 | 0.823 | 0.601 |
| sc:6 | 6.0 | 0.839 | 0.612 |
| ours:ours_lean | 7.4 | 0.994 | 0.679 |
| sc:9 | 9.0 | 0.845 | 0.606 |
| ours:ours | 11.7 | 0.988 | 0.649 |
| sc:12 | 12.0 | 0.850 | 0.595 |
| ours:ours_rich | 13.9 | 0.970 | 0.635 |

## Evidence-equivalence arms (Prop. 3)

| Arm | calls | prec@10 |
|---|---|---|
| A_12draws | 12.0 | 0.760 |
| B_6draws_battery | 9.0 | 0.970 |
| B_half_kappa | 9.0 | 0.970 |
| C_9draws | 9.0 | 0.780 |

B_vs_A: +0.210 [+0.120, +0.310]

B_vs_C: +0.190 [+0.090, +0.290]

Exchange rates, derived vs empirical: neg: κ=2.10 vs κ̂=2.95 [1.32, 5.87], relabel: κ=2.10 vs κ̂=1.76 [0.23, 4.59], distractor: κ=0.56 vs κ̂=0.93 [-1.41, 4.08]

## Monotonicity

Strata accuracy: 1.00, 1.00, 0.82, 0.56, 0.74, 0.92, 0.68, 0.72, 0.74, 0.66

Isotonic gap: 0.0067 [0.0028, 0.0185]

Top strata: {"1-10": {"n": 100, "acc": 1.0}, "11-30": {"n": 200, "acc": 0.73}, "31-50": {"n": 200, "acc": 0.73}, "p_1-10>11-30": 5.094332575893396e-09, "p_11-30>31-50": 0.5002243691753007}

## Exact Shapley (screen fixed)

| Signal | φ | φ per call |
|---|---|---|
| L | -0.0032 | -0.0005 |
| flip | +0.0555 | +0.0185 |
| se | +0.0852 | +0.0852 |
| chal | +0.0176 | +0.0176 |
| verb | +0.0325 | +0.0325 |

Random → screen: +0.0084; signals total: +0.1876; efficiency check: 0.0e+00; flip increment V(L,flip)−V(L): +0.1462

## Label-free audits

Held-out probe survival: selected 0.940 vs controls 0.705 (Fisher p=0.000, 300 probe executions)

Jaccard stability: 0.303 (chance floor 0.116)
