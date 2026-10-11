# Results: mmlu__sim

## Baseline ladder and ablations

| Method | prec@10 | calls/case | AUROC | PRR | Δ vs ours [95% CI] | Wilcoxon p |
|---|---|---|---|---|---|---|
| random | 0.776 | 1.0 | 0.500 | 0.000 | -0.184 [-0.222, -0.142] | 0.002 |
| verbalized | 1.000 | 1.0 | 0.994 | 0.992 | +0.040 [-0.000, +0.090] | 0.250 |
| self_consistency | 0.860 | 11.0 | 0.642 | 0.317 | -0.100 [-0.150, -0.040] | 0.027 |
| semantic_entropy | 1.000 | 11.0 | 0.805 | 0.763 | +0.040 [-0.000, +0.090] | 0.250 |
| bsdetector_like | 1.000 | 11.0 | 0.808 | 0.745 | +0.040 [-0.000, +0.090] | 0.250 |
| ours | 0.960 | 10.9 | 0.697 | 0.557 | — | — |
| ours_no_spread | 1.000 | 10.9 | 0.713 | 0.604 | +0.040 [-0.000, +0.090] | 0.250 |
| ours_no_sprt | 1.000 | 8.0 | 0.805 | 0.747 | +0.040 [-0.000, +0.090] | 0.250 |
| ours_no_flip | 0.950 | 9.0 | 0.696 | 0.557 | -0.010 [-0.030, -0.000] | 1.000 |
| ours_borda | 1.000 | 10.9 | 0.765 | 0.706 | +0.040 [-0.000, +0.090] | 0.250 |

Confirmatory (ours − self-consistency): +0.100 [+0.040, +0.150], wins 8/10, p=0.027

## Budget curve

| Config | calls | prec@10 | AUROC |
|---|---|---|---|
| verbalized | 1.0 | 1.000 | 0.994 |
| ours:ours_b3 | 3.0 | 0.785 | 0.585 |
| sc:3 | 3.0 | 0.785 | 0.609 |
| sc:6 | 6.0 | 0.825 | 0.636 |
| ours:ours_lean | 7.0 | 0.988 | 0.687 |
| sc:9 | 9.0 | 0.845 | 0.638 |
| ours:ours | 10.9 | 0.979 | 0.674 |
| sc:12 | 12.0 | 0.851 | 0.636 |
| ours:ours_rich | 12.9 | 0.920 | 0.664 |

## Evidence-equivalence arms (Prop. 3)

| Arm | calls | prec@10 |
|---|---|---|
| A_12draws | 12.0 | 0.800 |
| B_6draws_battery | 9.0 | 0.880 |
| B_half_kappa | 9.0 | 0.880 |
| C_9draws | 9.0 | 0.790 |

B_vs_A: +0.080 [+0.020, +0.140]

B_vs_C: +0.090 [+0.010, +0.180]

Exchange rates, derived vs empirical: neg: κ=2.10 vs κ̂=2.78 [0.29, 9.94], relabel: κ=2.10 vs κ̂=4.16 [1.62, 12.98], distractor: κ=0.56 vs κ̂=0.48 [-2.60, 5.50]

## Monotonicity

Strata accuracy: 1.00, 1.00, 0.62, 0.76, 0.98, 0.64, 0.58, 0.60, 0.64, 0.54

Isotonic gap: 0.0068 [0.0037, 0.0176]

Top strata: {"1-10": {"n": 100, "acc": 0.96}, "11-30": {"n": 200, "acc": 0.765}, "31-50": {"n": 200, "acc": 0.595}, "p_1-10>11-30": 1.1659949871988868e-05, "p_11-30>31-50": 0.00013671588618789708}

## Exact Shapley (screen fixed)

| Signal | φ | φ per call |
|---|---|---|
| L | +0.0037 | +0.0006 |
| flip | +0.0242 | +0.0081 |
| se | +0.0711 | +0.0711 |
| chal | +0.0188 | +0.0188 |
| verb | +0.0720 | +0.0720 |

Random → screen: +0.0342; signals total: +0.1898; efficiency check: 0.0e+00; flip increment V(L,flip)−V(L): +0.0958

## Label-free audits

Held-out probe survival: selected 0.950 vs controls 0.860 (Fisher p=0.012, 300 probe executions)

Jaccard stability: 0.339 (chance floor 0.116)
