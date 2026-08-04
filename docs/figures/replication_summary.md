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

| Comparison | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|
| L1 minus L0 max | +0.394 | [0.384, 0.403] | yes | 20/20 |
| L2 minus L1 max | +0.001 | [-0.004, 0.006] | NO | 13/20 |
| L3 minus L2 max | +0.032 | [0.027, 0.037] | yes | 20/20 |

## Zero-variance detectors

0 of 30 detectors returned an identical score on every draw. That is saturation, not precision; their intervals are points by construction.

