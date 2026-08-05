# The L0 negative control, with the forced cells removed

RHOB's L0 rung is a negative control: reward-only detectors should sit at chance on a
proxy calibrated to be indistinguishable between variants. The replication states that
control as **all 13 L0 detectors have a 95% interval containing 0.500**.

## The objection

That claim is computed over all 33 families. Six of them are marked DEGENERATE in the
admission ledger because their proxy reward is *constant*, and on a constant proxy every
L0 detector returns exactly 0.500 -- verified: all six score 0.500 for every L0 detector,
in every draw. Those 30 cells cannot come out any other way, so including them pulls each
detector's mean toward 0.500 by construction. A control padded with cells that are forced
to agree with it is the failure mode this project keeps finding.

## The check

Recomputed over the **27 non-degenerate families only**, family-weighted, bootstrapped over
the 20 replicates exactly as the headline is.

| L0 detector | all 33 families | 95% CI | contains 0.5 | 27 families | 95% CI | contains 0.5 |
|---|---|---|---|---|---|---|
| Gradient Reversal | 0.5027 | [0.4956, 0.5101] | yes | 0.5035 | [0.4948, 0.5123] | yes |
| Max Plateau | 0.4928 | [0.4841, 0.5027] | yes | 0.4910 | [0.4803, 0.5031] | yes |
| Reward Autocorrelation | 0.4947 | [0.4865, 0.5018] | yes | 0.4937 | [0.4841, 0.5022] | yes |
| Reward CUSUM | 0.4983 | [0.4905, 0.5061] | yes | 0.4982 | [0.4884, 0.5078] | yes |
| Reward KDE | 0.4923 | [0.4822, 0.5031] | yes | 0.4899 | [0.4772, 0.5029] | yes |
| Reward MLP | 0.5025 | [0.4901, 0.5146] | yes | 0.5030 | [0.4879, 0.5178] | yes |
| Reward Peak | 0.4987 | [0.4931, 0.5043] | yes | 0.4985 | [0.4916, 0.5052] | yes |
| Reward Skewness | 0.5025 | [0.4986, 0.5065] | yes | 0.5033 | [0.4983, 0.5082] | yes |
| Reward Threshold | 0.4905 | [0.4774, 0.5043] | yes | 0.4882 | [0.4724, 0.5050] | yes |
| Reward Trend | 0.4966 | [0.4896, 0.5040] | yes | 0.4960 | [0.4876, 0.5049] | yes |
| Reward Variance Ratio | 0.4854 | [0.4702, 0.5011] | yes | 0.4822 | [0.4632, 0.5014] | yes |
| Spectral Reward | 0.5062 | [0.4926, 0.5192] | yes | 0.5079 | [0.4913, 0.5239] | yes |
| Variance Window | 0.5005 | [0.4964, 0.5045] | yes | 0.5008 | [0.4958, 0.5056] | yes |

**Zero verdicts flip.** All 13 detectors contain 0.500 under both denominators. The negative
control does not depend on the forced cells; removing them moves individual means by at most
0.0032 — and the largest movers move *away* from 0.500, not toward it.

## What this does and does not license

It licenses: the L0 control is not an artifact of averaging in cells that could not have
come out otherwise.

It does not license: any statement about the *suite mean*. That statistic's interval endpoint
sits within the Monte-Carlo resolution of its own bootstrap (across 300 alternative bootstrap
seeds at 10,000 resamples, its upper bound falls above 0.500 in 18 and below in 282), so no
direction may be read from it. The control is carried detector by detector, which is where it
is robust.

Regenerate: `scripts/aggregate_replication.py` for the headline; this table is recomputed from
`results/replication/replicate_*.json` per-family values with the ledger's degenerate set.
