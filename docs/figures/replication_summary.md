# Leaderboard replication

20 independent draws of the full benchmark (layout seed and rollout seeds resampled together). Intervals are 95% percentile bootstrap over replicates.

## Access-level ladder

| Level | best-detector AUROC | 95% CI | mean over detectors | 95% CI |
|---|---|---|---|---|
| L0 | 0.571 | [0.560, 0.582] | 0.495 | [0.490, 0.500] |
| L1 | 0.998 | [0.996, 1.000] | 0.567 | [0.553, 0.579] |
| L2 | 0.943 | [0.938, 0.948] | 0.578 | [0.573, 0.584] |
| L3 | 0.975 | [0.971, 0.978] | 0.975 | [0.971, 0.978] |

## Supervised vs unsupervised

Five detectors expose `fit()` and are scored by 5-fold cross-validation on labels: `Reward MLP`, `State Coverage Rate`, `State Divergence`, `Trajectory MLP`, `Visitation Entropy Trend`.
Removing them and re-deriving the ladder from the same draws:

| Level | best AUROC (all) | 95% CI | best AUROC (unsupervised) | 95% CI | best unsupervised detector |
|---|---|---|---|---|---|
| L0 | 0.5713 | [0.5598, 0.5825] | 0.5677 | [0.5550, 0.5796] | Reward Skewness |
| L1 | 0.9984 | [0.9965, 0.9996] | 0.5384 | [0.5166, 0.5613] | Bimodal Occupancy |
| L2 | 0.9430 | [0.9380, 0.9478] | 0.5990 | [0.5892, 0.6096] | Reward-Feature Correlation |
| L3 | 0.9746 | [0.9712, 0.9778] | 0.9746 | [0.9711, 0.9778] | True Reward Oracle |

Unsupervised L1 minus L0 (max, paired): **-0.0293 [-0.0537, -0.0051]**, 5/20 draws favouring L1 -- separates.

The L0->L1 climb under the max (+0.4272) is carried by a label-fitted detector, `State Divergence`, the best L1 detector on 20/20 draws. Without the label-fitted detectors the step reverses sign and separates the other way: unsupervised L1 sits below L0. Quote the partition alongside any rung difference.

## Do adjacent rungs separate?

Paired within each draw, under BOTH statistics. Where the two disagree, the rung's
status depends on the statistic and neither answer may be quoted alone. Pairing does
not equalise denominators. The families each detector is scored on (a family is left out where it lacks the detector's channel or runs fewer episodes than its min_episodes, and the ledger-degenerate ones are left out of L0): L0 on 8 to 27, L1 on 7, L2 on 14 to 33, L3 on 33.

| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|---|
| L1 minus L0 | max | +0.4272 | [+0.4166, +0.4385] | yes | 20/20 |
| L2 minus L1 | max | -0.0555 | [-0.0609, -0.0505] | yes | 0/20 |
| L3 minus L2 | max | +0.0316 | [+0.0266, +0.0366] | yes | 20/20 |
| L1 minus L0 | mean | +0.0715 | [+0.0592, +0.0829] | yes | 20/20 |
| L2 minus L1 | mean | +0.0118 | [+0.0008, +0.0235] | yes | 13/20 |
| L3 minus L2 | mean | +0.3963 | [+0.3904, +0.4022] | yes | 20/20 |

**Statistic-dependent: L2 minus L1.** Both statistics separate it, in opposite directions: the max puts L1 above L2 (-0.0555 [-0.0609, -0.0505], 0/20 draws favouring L2); the mean puts L2 above L1 (+0.0118 [+0.0008, +0.0235], 13/20 draws favouring L2). Quote both.

## Zero-variance detectors

0 of 30 detectors returned an identical score on every draw. That is saturation, not precision; their intervals are points by construction.

