# Leaderboard replication

20 independent draws of the full benchmark (layout seed and rollout seeds resampled together). Intervals are 95% percentile bootstrap over replicates.

## Access-level ladder

| Level | best-detector AUROC | 95% CI | mean over detectors | 95% CI |
|---|---|---|---|---|
| L0 | 0.548 | [0.538, 0.557] | 0.497 | [0.493, 0.500] |
| L1 | 0.942 | [0.940, 0.943] | 0.559 | [0.547, 0.570] |
| L2 | 0.943 | [0.938, 0.948] | 0.574 | [0.569, 0.579] |
| L3 | 0.975 | [0.971, 0.978] | 0.975 | [0.971, 0.978] |

## Supervised vs unsupervised

Five detectors expose `fit()` and are scored by 5-fold cross-validation on labels: `Reward MLP`, `State Coverage Rate`, `State Divergence`, `Trajectory MLP`, `Visitation Entropy Trend`.
Removing them and re-deriving the ladder from the same draws:

| Level | best AUROC (all) | 95% CI | best AUROC (unsupervised) | 95% CI | best unsupervised detector |
|---|---|---|---|---|---|
| L0 | 0.5477 | [0.5385, 0.5574] | 0.5437 | [0.5332, 0.5543] | Spectral Reward |
| L1 | 0.9416 | [0.9399, 0.9427] | 0.5340 | [0.5147, 0.5543] | Bimodal Occupancy |
| L2 | 0.9430 | [0.9380, 0.9478] | 0.5872 | [0.5757, 0.5999] | Reward-Feature Correlation |
| L3 | 0.9746 | [0.9712, 0.9778] | 0.9746 | [0.9711, 0.9778] | True Reward Oracle |

Unsupervised L1 minus L0 (max, paired): **-0.0097 [-0.0283, +0.0084]**, 5/20 draws favouring L1 -- does NOT separate.

The published L0->L1 climb is carried by a single label-fitted detector
(`State Divergence`). Without the label-fitted detectors the step reverses sign
and stops separating. Quote the partition alongside any rung difference.

## Do adjacent rungs separate?

Paired within each draw, under BOTH statistics. Where the two disagree, the rung's
status depends on the statistic and neither answer may be quoted alone. Pairing does
not equalise denominators: L1 aggregates 35 cells over 8 families, L0 93 over 27,
L2/L3 123 over 33.

| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|---|
| L1 minus L0 | max | +0.3939 | [+0.3844, +0.4033] | yes | 20/20 |
| L2 minus L1 | max | +0.0014 | [-0.0039, +0.0063] | **NO** | 13/20 |
| L3 minus L2 | max | +0.0316 | [+0.0266, +0.0366] | yes | 20/20 |
| L1 minus L0 | mean | +0.0623 | [+0.0517, +0.0722] | yes | 20/20 |
| L2 minus L1 | mean | +0.0150 | [+0.0049, +0.0257] | yes | 13/20 |
| L3 minus L2 | mean | +0.4007 | [+0.3951, +0.4066] | yes | 20/20 |

**Statistic-dependent: L2_minus_L1.** Separates under one statistic and not the other.

## Zero-variance detectors

0 of 30 detectors returned an identical score on every draw. That is saturation, not precision; their intervals are points by construction.

