# Results: strategyqa__sim

## Baseline ladder and ablations

| Method | prec@10 | calls/case | AUROC | PRR | Δ vs ours [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| random | 0.756 | 1.0 | 0.500 | 0.000 | -0.224 [-0.280, -0.162] | 0.002 |
| verbalized | 0.930 | 1.0 | 0.758 | 0.594 | -0.050 [-0.090, -0.010] | 0.125 |
| self_consistency | 0.740 | 11.0 | 0.595 | 0.169 | -0.240 [-0.320, -0.150] | 0.008 |
| semantic_entropy | 0.990 | 11.0 | 0.758 | 0.687 | +0.010 [-0.020, +0.040] | 1.000 |
| bsdetector_like | 0.890 | 11.0 | 0.655 | 0.416 | -0.090 [-0.170, -0.020] | 0.094 |
| ours | 0.980 | 10.6 | 0.683 | 0.548 | — | — |
| ours_no_spread | 0.970 | 10.6 | 0.672 | 0.530 | -0.010 [-0.060, +0.030] | 1.000 |
| ours_no_sprt | 0.950 | 8.0 | 0.764 | 0.636 | -0.030 [-0.060, -0.000] | 0.250 |
| ours_no_flip | 0.990 | 8.7 | 0.680 | 0.543 | +0.010 [-0.000, +0.030] | 1.000 |
| ours_borda | 0.990 | 10.6 | 0.714 | 0.617 | +0.010 [-0.000, +0.030] | 1.000 |

Confirmatory (ours − self-consistency): +0.240 [+0.150, +0.320], wins 8/10, p=0.008

## Budget curve

| Config | calls | prec@10 | AUROC |
|---|---|---|---|
| verbalized | 1.0 | 0.930 | 0.758 |
| ours:ours_b3 | 3.0 | 0.797 | 0.598 |
| sc:3 | 3.0 | 0.769 | 0.582 |
| sc:6 | 6.0 | 0.783 | 0.595 |
| ours:ours_lean | 6.8 | 0.987 | 0.683 |
| sc:9 | 9.0 | 0.787 | 0.586 |
| ours:ours | 10.5 | 0.974 | 0.683 |
| sc:12 | 12.0 | 0.779 | 0.588 |
| ours:ours_rich | 12.4 | 0.960 | 0.688 |

## Evidence-equivalence arms (Prop. 3)

| Arm | calls | prec@10 |
|---|---|---|
| A_12draws | 12.0 | 0.870 |
| B_6draws_battery | 9.0 | 0.920 |
| B_half_kappa | 9.0 | 0.920 |
| C_9draws | 9.0 | 0.840 |

B_vs_A: +0.050 [+0.000, +0.100]

B_vs_C: +0.080 [+0.020, +0.140]

Exchange rates, derived vs empirical: neg: κ=2.10 vs κ̂=5.18 [0.27, 21.66], relabel: κ=2.10 vs κ̂=2.81 [-1.83, 12.15], distractor: κ=0.56 vs κ̂=2.90 [-1.46, 13.69]

## Monotonicity

Strata accuracy: 1.00, 1.00, 0.66, 0.82, 0.76, 0.60, 0.60, 0.52, 0.60, 0.70

Isotonic gap: 0.0029 [0.0007, 0.0110]

Top strata: {"1-10": {"n": 100, "acc": 0.98}, "11-30": {"n": 200, "acc": 0.715}, "31-50": {"n": 200, "acc": 0.61}, "p_1-10>11-30": 2.76290268134378e-08, "p_11-30>31-50": 0.013303382507874441}

## Exact Shapley (screen fixed)

| Signal | φ | φ per call |
|---|---|---|
| L | +0.0233 | +0.0039 |
| flip | +0.0336 | +0.0112 |
| se | +0.0757 | +0.0757 |
| chal | +0.0163 | +0.0163 |
| verb | +0.0515 | +0.0515 |

Random → screen: +0.0336; signals total: +0.2004; efficiency check: 0.0e+00; flip increment V(L,flip)−V(L): +0.0790

## Label-free audits

Held-out probe survival: selected 0.910 vs controls 0.795 (Fisher p=0.007, 300 probe executions)

Jaccard stability: 0.538 (chance floor 0.116)
