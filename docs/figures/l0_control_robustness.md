# The L0 negative control, with the forced cells removed

RHOB's L0 rung is a negative control: reward-only detectors should sit at chance on a
proxy calibrated to be indistinguishable between variants. The replication states that
control as **all 13 L0 detectors have a 95% interval containing 0.500**.

## The objection

That claim is computed over all 33 families -- each detector's `overall_auroc` is the
cell-weighted mean over all 123 cells, and `scripts/aggregate_replication.py` applies the
ledger exclusion only inside `summarize_access_levels`, i.e. only to the level aggregate.
Six families are marked DEGENERATE in the admission ledger because their proxy is
*constant* (three families) or identical, or nearly, in every run (three), so an L0
statistic can order almost none of their run pairs, and there an L0 detector returns 0.500
for a reason that has nothing to do with proxy matching. Those 30 cells cannot say anything
else, so including them pulls each detector's mean toward 0.500 by construction. A control padded with cells
that are forced to agree with it is the failure mode this project keeps finding.

Verified, 13 detectors x 20 draws = 260 values per family: **four of the six return
exactly 0.500 every time** (`distributional_shift`, `monitored_sandbagging`,
`rlhf_reward_model_overopt`, `shortcut_exploitation`; 0 of 260 depart). The other two are
constant only to numerical dust -- `orbit_chirality` has relative dispersion 1.36e-16
(1.1e-16 in the ledger regenerated on 2026-11-19) and `physics_exploitation`
strictly orders 0.35% of cross-variant pairs -- so a detector does
occasionally order a pair on rounding error: 27 of 260 and 37 of 260 respectively. That is
degeneracy of the second kind rather than a measurement, so the argument for holding all
six out is unchanged. An earlier version of this file said all six were exactly 0.500 in
every draw; that was wrong and is corrected here. The check below is unaffected -- it
removes all six families whichever way their cells came out.

## The check

Recomputed over the **27 non-degenerate families only**, family-weighted, bootstrapped over
the 20 replicates exactly as the headline is. Both columns are family-weighted so the
comparison is like for like; the published per-detector figure is the cell-weighted one and
differs from the 33-family column below by at most 0.0024 (Reward Variance Ratio), which is
not what this check is about.

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

It does not license: any statement about which side of chance the **suite mean** sits on.
The published interval is `access_levels.L0.mean_auroc` = 0.4966217948717948,
[0.4933695869, 0.4999211859], and its upper endpoint is not resolvable from 0.500 by the
bootstrap that produced it. Replaying the published procedure (10,000 resamples of the same
20 draws) at generator seeds 0..299 instead of the script's fixed `BOOTSTRAP_SEED = 20260804`
moves that endpoint over **[0.499823, 0.500060]**: it lands at or above 0.500 in **18 of the
300** and below it in the other **282**. The endpoint's own Monte-Carlo scatter straddles
0.500, so the interval places the suite mean on neither side of chance and no direction may
be read from it. Subtracting the published endpoint from 0.500 and reporting the remainder is
the false-precision error itself, not a smaller version of it; earlier versions of this file
did exactly that, and the arithmetic is deliberately not repeated here.

The endpoint has also moved twice across regenerations, with no measurement changed and no
replicate re-run. `aggregate_replication.py` used to draw every interval from one
`default_rng(BOOTSTRAP_SEED)` stream, so a bootstrap inserted *ahead* of a published one
advanced the stream underneath it: `access_levels.L0.mean_auroc.ci_hi` went 0.4999495905 ->
0.4998890919 as the suite-mean column and then the unsupervised ladder were added, and ->
0.4999211859 when the mechanism was repaired. `mean` is 0.4966217948717948 at all three.
The script now seeds each interval from a digest of its own name (commit 3f080de), so adding
a reported quantity can no longer move a published one. That closes one source of movement;
it does not make this endpoint resolvable, which is what the 300-seed replay above measures.

The per-detector verdicts are not seed-sensitive in that way. Over the same 300 seeds, all
13 L0 intervals contain 0.500 at every seed; the narrowest margin from 0.500 at any seed is
0.0011 (Reward Variance Ratio), about five times the scatter that decides the suite mean's
endpoint. The control is carried detector by detector, which is where it is robust.

**What the 27 families are (2026-11-19).** Nor does the check license a claim that the
proxy is matched on those 27 families. Since its 2026-11-19 regeneration (`d4b700d`) the
admission ledger covers all 33 families, 123 cells. It admits 42 of them, certifying a
matched proxy on each; a 43rd, `proxy_correlation_gaming` @ 0.60, passes both proxy
criteria and fails only `behavioral_separated`. It finds the same six families DEGENERATE
(30 cells) and measures 51 NOT ADMITTED, 50 of them failing a proxy criterion: 24 fail
`proxy_matched`, 40 fail `proxy_distribution_matched` (14 fail both), and 1 fails
`behavioral_separated` alone. The 27-family column above holds out only the DEGENERATE
families, as the L0 aggregate does, so the NOT ADMITTED cells are in it: all 51, for every
detector. Whether they should be is an open question for the author
([ROADMAP.md](../../ROADMAP.md)). The degenerate set is the same six families, so the
regeneration changes nothing in the table.

The table, the suite-mean interval and the seed replay above date from 2026-08-06
(`97ac575`), before the re-score of `253c7ec` (2026-09-03) recorded Reward Skewness not
applicable on the 19 families that run 40 or 60 episodes. Its Skewness row therefore still
averages that detector's fallback 0.500 on those families, NOT ADMITTED cells included.
Since the re-score the replicated board's Skewness row covers 14 families and 9 of the 51
NOT ADMITTED cells, and `leaderboard/v5_replicated.json` gives the L0 suite mean as 0.4972
[0.4939, 0.5005] rather than 0.4966 [0.4934, 0.4999]. This file has not been recomputed
since.

Skewness is the only L0 detector that has been re-scored this way, because it is the only
one that declares its horizon. Eight other rows still average a fallback 0.500 that is not
a measurement, in both columns and in every draw: Reward Peak, Reward Autocorrelation,
Reward Trend and Variance Window on the 19 families of 40 or 60 episodes, and Max Plateau,
Reward MLP, Gradient Reversal and Reward CUSUM on the 7 of 40. Like the degenerate cells
above, those cells could not have come out otherwise, and they pull each of those rows
toward 0.500. Declaring each detector's horizon and re-scoring the boards is an open item
in [ROADMAP.md](../../ROADMAP.md); this table would then have to be recomputed.

Regenerate: `scripts/aggregate_replication.py` for the headline; this table is recomputed from
`results/replication/replicate_*.json` per-family values with the ledger's degenerate set.
The seed replay re-runs `bootstrap_ci`'s body -- 10,000 resamples of the 20 per-replicate
values -- under `default_rng(s)` for s in 0..299, on the L0 suite mean and on each of the 13
L0 detectors.
