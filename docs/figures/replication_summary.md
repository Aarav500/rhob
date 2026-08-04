# Leaderboard replication

20 independent draws of the full benchmark (layout seed and rollout seeds resampled together). Intervals are 95% percentile bootstrap over replicates.

## Access-level ladder

| Level | best-detector AUROC | 95% CI | mean over detectors | 95% CI |
|---|---|---|---|---|
| L0 | 0.548 | [0.538, 0.557] | 0.497 | [0.493, 0.500] |
| L1 | 0.942 | [0.940, 0.943] | 0.559 | [0.547, 0.570] |
| L2 | 0.943 | [0.938, 0.948] | 0.574 | [0.569, 0.579] |
| L3 | 0.975 | [0.971, 0.978] | 0.975 | [0.971, 0.978] |

## Do adjacent rungs separate?

Paired within each draw, under BOTH statistics. Where the two disagree, the rung's
status depends on the statistic and neither answer may be quoted alone. Pairing does
not equalise denominators: L1 aggregates 35 cells over 8 families, L0 93 over 27,
L2/L3 123 over 33.

| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|---|
| L1 minus L0 | max | +0.3939 | [+0.3843, +0.4035] | yes | 20/20 |
| L2 minus L1 | max | +0.0014 | [-0.0039, +0.0063] | **NO** | 13/20 |
| L3 minus L2 | max | +0.0316 | [+0.0268, +0.0367] | yes | 20/20 |
| L1 minus L0 | mean | +0.0623 | [+0.0515, +0.0726] | yes | 20/20 |
| L2 minus L1 | mean | +0.0150 | [+0.0050, +0.0255] | yes | 13/20 |
| L3 minus L2 | mean | +0.4007 | [+0.3949, +0.4066] | yes | 20/20 |

**Statistic-dependent: L2_minus_L1.** Separates under one statistic and not the other.

## Zero-variance detectors

0 of 30 detectors returned an identical score on every draw. That is saturation, not precision; their intervals are points by construction.

