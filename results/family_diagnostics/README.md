# Family diagnostics

Measurements that back a claim made somewhere else in the repository. They are committed
because the claim they support is quantitative, and this project's own rule is that a
figure a reader is asked to accept should not live only in a comment.

Regenerate either with the script named beside it; both print as they go and write their
JSON here.

## `pettingzoo_fixed_opponent_shape_leak.json`

`scripts/diagnose_shape_leak.py`. Reproduces the nightly `admission-slow` failure of
`pettingzoo_fixed_opponent_exploitation` at all three scored tiers, at the same 12-layout
by 4-seed design the smoke screen uses, and reports what `proxy_distribution_matched`
rejects it on.

| tier | fitted `gain_boost` | proxy mean (hack / legit) | per-episode SD (hack / legit) | Reward KDE AUROC | resolution |
|---|---|---|---|---|---|
| 0.9 | 1.731 | 9.968 / 10.000 | 3.601 / 3.827 | 0.234 | 1.000 |
| 0.8 | 0.967 | 9.994 / 10.000 | 4.427 / 3.827 | 0.849 | 1.000 |
| 0.7 | 0.878 | 10.023 / 10.000 | 4.471 / 3.827 | 0.833 | 1.000 |

The smoke band is 0.5 +/- 0.256. The mean is matched at every tier, which is why
`proxy_matched` passes at 0.565; the per-episode spread is not. Reward KDE scores the late
window's density under the early window's, so it reads absolute stream spread, and the
sign of the leak is the sign of (hacking SD - legit SD): under the band at `fixed_pull`
0.84, over it at 0.72 and 0.60. Resolution is 1.000 at every tier, so these are
measurements rather than the tie convention. The two other shape detectors stay inside
the band (Variance Ratio 0.42-0.46, Skewness 0.46-0.52), so the panel's verdict rests on
Reward KDE alone.

## `pettingzoo_fixed_opponent_spread_sweep.json`

`scripts/diagnose_spread_lever.py`. Asks whether the family's variance-dampening
constant, `_CORNER_SPREAD = 0.15`, can be retuned to match the spread the way
`gain_boost` matches the mean. It cannot. At `fixed_pull` 0.60, against a legit SD of
3.911:

| `_CORNER_SPREAD` | proxy mean (gap vs legit) | SD ratio |
|---|---|---|
| 0.00 | 11.395 (+1.360) | 1.116 |
| **0.15** (shipped) | **10.033 (-0.002)** | **1.100** |
| 0.30 | 2.988 (-7.047) | 0.761 |
| 0.50 | 1.508 (-8.527) | 0.489 |
| 0.75 | 0.893 (-9.142) | 0.329 |

Moving the constant from 0.15 to 0.30 buys 0.34 of SD ratio and costs 7.05 of proxy mean.
It is not an orthogonal knob, so there is no setting that matches both, and `gain_boost`
cannot make up the difference: the family's own module records it saturating, with a boost
of 6.0 still landing 1.3 short of target at `fixed_pull` 1.0. The same shape holds at
0.72 and 0.84.

That is why the family carries a per-tier `xfail(strict=True)` rather than a fix. The
second-calibration-parameter remedy that closed the other shape leaks is unavailable
here; closing this one needs a different variance mechanism in the environment, which is
a construction change and not a retune.
