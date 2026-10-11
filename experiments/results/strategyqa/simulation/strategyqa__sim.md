# Results: strategyqa__sim

## Baseline ladder and ablations

| Method | prec@10 | calls/case | AUROC | PRR | Δ vs CEL [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| random | 0.684 | 1.0 | 0.500 | -0.000 | -0.296 [-0.346, -0.244] | 0.002 |
| verbalized | 0.790 | 1.0 | 0.663 | 0.362 | -0.190 [-0.230, -0.150] | 0.002 |
| self_consistency | 0.740 | 11.0 | 0.595 | 0.169 | -0.240 [-0.320, -0.150] | 0.008 |
| self_consistency_deploy | 0.740 | 3.0 | 0.579 | 0.161 | -0.240 [-0.340, -0.140] | 0.004 |
| semantic_entropy | 0.990 | 11.0 | 0.758 | 0.687 | +0.010 [-0.020, +0.040] | 1.000 |
| bsdetector_like | 0.820 | 10.0 | 0.633 | 0.302 | -0.160 [-0.220, -0.100] | 0.008 |
| cel | 0.980 | 11.0 | 0.684 | 0.549 | — | — |
| cel_no_spread | 0.970 | 11.0 | 0.672 | 0.530 | -0.010 [-0.060, +0.030] | 1.000 |
| cel_no_sprt | 0.950 | 8.0 | 0.765 | 0.649 | -0.030 [-0.060, -0.000] | 0.250 |
| cel_no_flip | 0.990 | 9.1 | 0.681 | 0.544 | +0.010 [-0.000, +0.030] | 1.000 |
| cel_no_se | 0.920 | 10.3 | 0.679 | 0.521 | -0.060 [-0.090, -0.030] | 0.031 |
| cel_no_chal | 0.980 | 10.3 | 0.684 | 0.550 | -0.000 [-0.000, -0.000] | 1.000 |
| cel_no_verb | 0.980 | 11.0 | 0.683 | 0.548 | -0.000 [-0.000, -0.000] | 1.000 |
| cel_no_L | 0.980 | 11.0 | 0.684 | 0.549 | -0.000 [-0.000, -0.000] | 1.000 |
| cel_borda | 0.980 | 11.0 | 0.716 | 0.625 | -0.000 [-0.030, +0.030] | 1.000 |

Confirmatory (CEL − self-consistency): +0.240 [+0.150, +0.320], wins 8/10, p=0.008

## Budget curve

| Config | calls | prec@10 | AUROC |
|---|---|---|---|
| verbalized | 1.0 | 0.790 | 0.663 |
| celk:2 | 2.0 | 0.920 | 0.754 |
| celk:3 | 3.0 | 0.960 | 0.747 |
| celk:4 | 3.0 | 0.960 | 0.747 |
| celk:5 | 3.0 | 0.960 | 0.747 |
| celk:6 | 3.0 | 0.960 | 0.747 |
| celk:8 | 3.0 | 0.960 | 0.747 |
| sc:3 | 3.0 | 0.772 | 0.577 |
| cel:cel_b3 | 4.0 | 0.797 | 0.598 |
| sc:6 | 6.0 | 0.798 | 0.595 |
| cel:cel_lean | 7.8 | 0.987 | 0.683 |
| sc:9 | 9.0 | 0.789 | 0.587 |
| cel:cel | 10.9 | 0.973 | 0.683 |
| sc:12 | 12.0 | 0.786 | 0.580 |
| cel:cel_rich | 12.8 | 0.960 | 0.688 |

## Evidence-equivalence arms (Prop. 3)

| Arm | calls | prec@10 |
|---|---|---|
| A_12draws | 13.0 | 0.870 |
| B_6draws_battery | 10.0 | 0.920 |
| B_half_kappa | 10.0 | 0.920 |
| C_9draws | 10.0 | 0.840 |

B_vs_A: +0.050 [+0.000, +0.100]

B_vs_C: +0.080 [+0.020, +0.140]

Exchange rates, derived vs empirical: neg: κ=2.10 vs κ̂=5.18 [0.53, 26.56], relabel: κ=2.10 vs κ̂=2.81 [-1.08, 13.55], distractor: κ=0.56 vs κ̂=2.90 [-2.54, 17.64]

## Monotonicity

Strata accuracy: 1.00, 1.00, 0.66, 0.82, 0.76, 0.60, 0.60, 0.52, 0.60, 0.70

Isotonic gap: 0.0029 [0.0006, 0.0110]

Top strata: {"1-10": {"n": 100, "acc": 0.98}, "11-30": {"n": 200, "acc": 0.715}, "31-50": {"n": 200, "acc": 0.61}, "p_1-10>11-30": 2.7629026813437808e-08, "p_11-30>31-50": 0.013303382507874441}

## Budget-allocated CEL-k vs self-consistency with k draws

Allocation chosen on 100 dev items by prec@k among 1760 candidates; CEL-k on expected calls per case, SC on exactly k.

| k | CEL-k allocation | E[calls] dev / eval | CEL-k prec@10 | SC-k prec@10 | SC-k+verbal | Δ vs SC [95% CI] | wins | Δ vs SC+verbal [95% CI] |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 orig draw → distractor | 2.0 / 2.0 | 0.920 | 0.740 | 0.880 | +0.180 [+0.100, +0.240] | 9/10 | +0.040 [-0.010, +0.090] |
| 3 | 1 orig draw → distractor, chal | 3.0 / 3.0 | 0.960 | 0.740 | 0.900 | +0.220 [+0.120, +0.330] | 8/10 | +0.060 [+0.010, +0.110] |
| 4 | 1 orig draw → distractor, chal | 3.0 / 3.0 | 0.960 | 0.750 | 0.930 | +0.210 [+0.110, +0.320] | 8/10 | +0.030 [-0.020, +0.080] |
| 5 | 1 orig draw → distractor, chal | 3.0 / 3.0 | 0.960 | 0.760 | 0.930 | +0.200 [+0.110, +0.300] | 9/10 | +0.030 [-0.020, +0.080] |
| 6 | 1 orig draw → distractor, chal | 3.0 / 3.0 | 0.960 | 0.790 | 0.960 | +0.170 [+0.070, +0.280] | 7/10 | +0.000 [-0.040, +0.040] |
| 8 | 1 orig draw → distractor, chal | 3.0 / 3.0 | 0.960 | 0.780 | 0.950 | +0.180 [+0.090, +0.290] | 8/10 | +0.010 [-0.040, +0.060] |

## Signal effects on confidence estimation

Leave-one-out = CEL minus CEL without the signal (Δprec@10 paired over batches). Shapley = fair share of the prec@10 gain over all signal coalitions (Borda, screen fixed).

| Signal | LOO Δprec@10 [95% CI] | LOO ΔAUROC | LOO ΔPRR | calls saved | φ prec@10 [95% CI] | φ per call |
|---|---|---|---|---|---|---|
| L | +0.000 [+0.000, +0.000] | +0.000 | +0.000 | 0.00 | +0.0150 [-0.0010, +0.0322] | +0.0025 |
| flip | -0.010 [-0.030, +0.000] | +0.003 | +0.005 | 1.90 | +0.0218 [+0.0062, +0.0363] | +0.0073 |
| se | +0.060 [+0.030, +0.090] | +0.005 | +0.028 | 0.63 | +0.0681 [+0.0522, +0.0846] | +0.0681 |
| chal | +0.000 [+0.000, +0.000] | -0.000 | -0.001 | 0.63 | +0.0120 [-0.0007, +0.0234] | +0.0120 |
| verb | +0.000 [+0.000, +0.000] | +0.000 | +0.001 | 0.00 | +0.0734 [+0.0588, +0.0881] | +0.0734 |

Random → screen: +0.1056; signals total: +0.1904; efficiency check: 1.4e-17; flip increment V(L,flip)−V(L): +0.0790

## Label-free audits

Held-out probe survival: selected 0.910 vs controls 0.795 (Fisher p=0.007, 300 probe executions)

Jaccard stability: 0.538 (chance floor 0.116)
