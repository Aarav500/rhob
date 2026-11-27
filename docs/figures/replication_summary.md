# Leaderboard replication

20 independent draws of the full benchmark (layout seed and rollout seeds resampled together). Intervals are 95% percentile bootstrap over replicates.

## Access-level ladder

| Level | best-detector AUROC | 95% CI | mean over detectors | 95% CI |
|---|---|---|---|---|
| L0 | 0.575 | [0.565, 0.586] | 0.498 | [0.492, 0.504] |
| L1 | 0.998 | [0.996, 1.000] | 0.567 | [0.553, 0.579] |
| L2 | 0.947 | [0.941, 0.952] | 0.593 | [0.583, 0.604] |
| L3 | 0.979 | [0.976, 0.981] | 0.979 | [0.976, 0.981] |

## Supervised vs unsupervised

Five detectors expose `fit()` and are scored by 5-fold cross-validation on labels: `Reward MLP`, `State Coverage Rate`, `State Divergence`, `Trajectory MLP`, `Visitation Entropy Trend`.
Removing them and re-deriving the ladder from the same draws:

| Level | best AUROC (all) | 95% CI | best AUROC (unsupervised) | 95% CI | best unsupervised detector |
|---|---|---|---|---|---|
| L0 | 0.5755 | [0.5650, 0.5859] | 0.5699 | [0.5598, 0.5795] | Reward Skewness |
| L1 | 0.9984 | [0.9965, 0.9996] | 0.5384 | [0.5166, 0.5613] | Bimodal Occupancy |
| L2 | 0.9466 | [0.9408, 0.9525] | 0.6082 | [0.5950, 0.6243] | Reward-Feature Correlation |
| L3 | 0.9789 | [0.9761, 0.9814] | 0.9789 | [0.9760, 0.9815] | True Reward Oracle |

Adjacent rungs without the label-fitted detectors, paired within each draw under both statistics:

| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|---|
| L1 minus L0 | max | -0.0315 | [-0.0548, -0.0087] | yes | 4/20 |
| L2 minus L1 | max | +0.0698 | [+0.0426, +0.0976] | yes | 17/20 |
| L3 minus L2 | max | +0.3707 | [+0.3525, +0.3853] | yes | 20/20 |
| L1 minus L0 | mean | -0.0050 | [-0.0242, +0.0142] | **NO** | 10/20 |
| L2 minus L1 | mean | +0.0419 | [+0.0186, +0.0671] | yes | 13/20 |
| L3 minus L2 | mean | +0.4448 | [+0.4316, +0.4571] | yes | 20/20 |

**Statistic-dependent without the label-fitted detectors: L1 minus L0.** One statistic separates it and the other does not: the max puts L0 above L1 (-0.0315 [-0.0548, -0.0087], 4/20 draws favouring L1); the mean does not separate L1 from L0 (-0.0050 [-0.0242, +0.0142], 10/20 draws favouring L1). Quote both.

The L0->L1 climb under the max (+0.4230) is carried by a label-fitted detector, `State Divergence`, the best L1 detector on 20/20 draws. Without the label-fitted detectors the step reverses sign under the max, and whether it separates depends on the statistic: the max puts L0 above L1 and the mean does not separate L1 from L0, so quote both. Quote the partition alongside any rung difference.

## Do adjacent rungs separate?

Paired within each draw, under BOTH statistics. Where the two disagree, the rung's
status depends on the statistic and neither answer may be quoted alone. Pairing does
not equalise denominators. The families each detector is scored on (a family is left out where it lacks the detector's channel or runs fewer episodes than its min_episodes, and the ledger-degenerate ones are left out of L0): L0 on 8 to 27, L1 on 7, L2 on 14 to 33, L3 on 33.

| Comparison | statistic | difference | 95% CI | excludes 0? | draws favouring higher |
|---|---|---|---|---|---|
| L1 minus L0 | max | +0.4230 | [+0.4131, +0.4329] | yes | 20/20 |
| L2 minus L1 | max | -0.0518 | [-0.0577, -0.0460] | yes | 0/20 |
| L3 minus L2 | max | +0.0322 | [+0.0253, +0.0390] | yes | 19/20 |
| L1 minus L0 | mean | +0.0689 | [+0.0574, +0.0803] | yes | 20/20 |
| L2 minus L1 | mean | +0.0264 | [+0.0103, +0.0440] | yes | 13/20 |
| L3 minus L2 | mean | +0.3859 | [+0.3740, +0.3969] | yes | 20/20 |

**Statistic-dependent: L2 minus L1.** Both statistics separate it, in opposite directions: the max puts L1 above L2 (-0.0518 [-0.0577, -0.0460], 0/20 draws favouring L2); the mean puts L2 above L1 (+0.0264 [+0.0103, +0.0440], 13/20 draws favouring L2). Quote both.

## Zero-variance detectors

0 of 30 detectors returned an identical score on every draw. That is saturation, not precision; their intervals are points by construction.

## Provenance of the draws

Read from each replicate file's own provenance (`replicates` in `leaderboard/v5_replicated.json`).

- Draw schema: 2 (layout seed forwarded to the families), on all 20.
- Source commit: `54cd63cfac06`, on all 20.
- Tree: clean tree, on all 20.
- Python: CPython 3.12.16, on all 20.
- Packages: numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, torch 2.15.0, rhob 1.8.0, mujoco 3.17.0, mujoco-core (not installed), gymnasium 1.4.0, pettingzoo 1.27.0, mpe2 1.1.1, on all 20.
- Platform: Linux-7.0.0-1018-aws-x86_64-with-glibc2.39 on replicates 0, 2, 4, 8, 10-13, 15, 17-18; Linux-7.0.0-1020-azure-x86_64-with-glibc2.39 on replicates 1, 3, 6-7, 9, 14, 16, 19; Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 on replicate 5.
- Numerics: OPENBLAS_CORETYPE=Haswell, OPENBLAS_NUM_THREADS=1, MKL_CBWR=COMPATIBLE, MKL_NUM_THREADS=1, OMP_NUM_THREADS=1, ATEN_CPU_CAPABILITY=avx2; ATen AVX2; openblas Haswell; openblas Haswell; openmp; openmp, on all 20.
- CPU: GenuineIntel Intel(R) Xeon(R) 6975P-C on replicates 0, 2, 4-5, 8, 10-13, 15, 17-18; AuthenticAMD AMD EPYC 7763 64-Core Processor on replicates 1, 3, 7, 19; AuthenticAMD AMD EPYC 9V45 96-Core Processor on replicates 6, 9, 14; AuthenticAMD AMD EPYC 9V74 80-Core Processor on replicate 16.
- Rows re-scored after the draw: none, on all 20.
- Single source: **yes** -- one commit, clean trees, one Python, one package set, one numerics signature, one draw schema, no re-scored rows.
