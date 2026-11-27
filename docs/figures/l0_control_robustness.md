# The L0 negative control, with the forced cells removed

RHOB's L0 rung is a negative control: reward-only detectors should sit at chance on a
proxy calibrated to be indistinguishable between variants. The replication states that
control as **all 13 L0 detectors have a 95% interval containing 0.500**.
*(2026-11-27: that is true of the 20 draws made on 2026-08-04, which this file analyses.
On the re-draw that replaced them it is not: 12 of the 13 do, and Reward CUSUM's,
0.486 [0.475, 0.497], lies wholly below 0.500. See the end of this file.)*

## The objection

That claim is computed over every family a detector is scored on -- each detector's
`overall_auroc` is the cell-weighted mean over its measured cells (all 123 for four of the
13, 101 for four and 64 for five, which are not applicable on the families shorter than the
`min_episodes` they declare), and `scripts/aggregate_replication.py` applies the
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
occasionally order a pair on rounding error: 27 of 260 and 37 of 260 respectively.
*(2026-11-27: these are the counts of the 2026-08-04 draws. On the re-draw
`rlhf_reward_model_overopt` departs in 6 of 260 and `orbit_chirality` in 21; see the end
of this file.)* That is
degeneracy of the second kind rather than a measurement, so the argument for holding all
six out is unchanged. An earlier version of this file said all six were exactly 0.500 in
every draw; that was wrong and is corrected here. The check below is unaffected -- it
removes all six families whichever way their cells came out.

## The check

Recomputed over the **non-degenerate families only**, family-weighted, bootstrapped over
the 20 replicates exactly as the headline is. A detector contributes only the families it is
scored on: all 33 (27 without the degenerate six) for Reward KDE, Reward Threshold, Reward
Variance Ratio and Spectral Reward; 26 (20) for Gradient Reversal, Max Plateau, Reward CUSUM
and Reward MLP, which are not applicable on the 7 families of 40 episodes; and 14 (8) for
Reward Autocorrelation, Reward Peak, Reward Skewness, Reward Trend and Variance Window,
which are not applicable on the 19 of 40 or 60. Every detector is scored on all six
degenerate families. Both columns are family-weighted so the comparison is like for like;
the published per-detector figure is the cell-weighted one and differs from the
all-families column below by at most 0.0032 (Reward Autocorrelation, 0.00315; Max Plateau
0.00314), which is not what this check is
about.

| L0 detector | families | all scored families | 95% CI | contains 0.5 | without the 6 degenerate | 95% CI | contains 0.5 |
|---|---|---|---|---|---|---|---|
| Gradient Reversal | 26 / 20 | 0.5034 | [0.4944, 0.5128] | yes | 0.5047 | [0.4930, 0.5166] | yes |
| Max Plateau | 26 / 20 | 0.4908 | [0.4798, 0.5034] | yes | 0.4878 | [0.4734, 0.5043] | yes |
| Reward Autocorrelation | 14 / 8 | 0.4875 | [0.4682, 0.5042] | yes | 0.4787 | [0.4462, 0.5076] | yes |
| Reward CUSUM | 26 / 20 | 0.4979 | [0.4879, 0.5077] | yes | 0.4975 | [0.4843, 0.5105] | yes |
| Reward KDE | 33 / 27 | 0.4923 | [0.4822, 0.5031] | yes | 0.4899 | [0.4772, 0.5029] | yes |
| Reward MLP | 26 / 20 | 0.5032 | [0.4874, 0.5186] | yes | 0.5041 | [0.4837, 0.5240] | yes |
| Reward Peak | 14 / 8 | 0.4970 | [0.4838, 0.5100] | yes | 0.4948 | [0.4717, 0.5176] | yes |
| Reward Skewness | 14 / 8 | 0.5060 | [0.4966, 0.5152] | yes | 0.5110 | [0.4941, 0.5277] | yes |
| Reward Threshold | 33 / 27 | 0.4905 | [0.4774, 0.5043] | yes | 0.4882 | [0.4724, 0.5050] | yes |
| Reward Trend | 14 / 8 | 0.4919 | [0.4754, 0.5094] | yes | 0.4865 | [0.4581, 0.5166] | yes |
| Reward Variance Ratio | 33 / 27 | 0.4854 | [0.4702, 0.5011] | yes | 0.4822 | [0.4632, 0.5014] | yes |
| Spectral Reward | 33 / 27 | 0.5062 | [0.4926, 0.5192] | yes | 0.5079 | [0.4913, 0.5239] | yes |
| Variance Window | 14 / 8 | 0.5012 | [0.4914, 0.5107] | yes | 0.5026 | [0.4859, 0.5191] | yes |

**Zero verdicts flip.** All 13 detectors contain 0.500 under both denominators. The negative
control does not depend on the forced cells; removing them moves individual means by at most
0.0088 (Reward Autocorrelation) — and every one of the 13 moves *away* from 0.500, not
toward it. The three largest moves (Reward Autocorrelation 0.0088, Reward Trend 0.0054,
Reward Skewness 0.0051) are among the five detectors scored on 14 families, where the six
degenerate ones are close to half the denominator, and so are the three widest
degenerate-free intervals (Reward Autocorrelation, Reward Trend, Reward Peak), which rest
on 8 families. Family count does not order the rest: Reward Peak (0.0022) and Variance
Window (0.0015) move less than Reward Variance Ratio (0.0032), which is scored on 33.
*(2026-11-27: the table and these verdicts are of the 2026-08-04 draws and have not been
recomputed on the re-draw, where Reward CUSUM's published interval excludes 0.500; see
the end of this file.)*

## What this does and does not license

It licenses: the L0 control is not an artifact of averaging in cells that could not have
come out otherwise.

It does not license: any statement about which side of chance the **suite mean** sits on.
The published interval is `access_levels.L0.mean_auroc` = 0.49507437321937314,
[0.4898966901, 0.5000149501]. It contains 0.500 only at its upper end (0.50001 at five
decimals; it prints as 0.5000 at four), and that endpoint is not resolvable from 0.500 by
the bootstrap that produced it. *(2026-11-27: that is the interval of the 2026-08-04
draws. On the re-draw it is 0.4976617058404559, [0.4917808965, 0.5035671936], which
contains 0.500 with room; the replay below is of the 2026-08-04 draws.)* Replaying the
published procedure (10,000 resamples of the same 20 draws) at generator seeds 0..299
instead of the script's fixed
`BOOTSTRAP_SEED = 20260804` moves that endpoint over **[0.499879, 0.500216]**: it lands at or
above 0.500 in **225 of the 300** and below it in the other **75**. The endpoint's own
Monte-Carlo scatter straddles 0.500, so the interval places the suite mean on neither side
of chance and no direction may be read from it. Subtracting the published endpoint from
0.500 and reporting the remainder is the false-precision error itself, not a smaller version
of it; earlier versions of this file did exactly that, and the arithmetic is deliberately
not repeated here. The reading is the one this file gave the interval of 2026-08-06,
0.4966 [0.4934, 0.4999], whose endpoint the same replay put at or above 0.500 in 18 of 300:
the endpoint sat on the other side of 0.500 then, and in neither case does its side mean
anything.

On the board this file was first written against, the endpoint also moved twice across
regenerations, with no measurement changed and no replicate re-run. `aggregate_replication.py` used to draw every interval from one
`default_rng(BOOTSTRAP_SEED)` stream, so a bootstrap inserted *ahead* of a published one
advanced the stream underneath it: `access_levels.L0.mean_auroc.ci_hi` went 0.4999495905 ->
0.4998890919 as the suite-mean column and then the unsupervised ladder were added, and ->
0.4999211859 when the mechanism was repaired. `mean` is 0.4966217948717948 at all three.
The script now seeds each interval from a digest of its own name (commit 3f080de), so adding
a reported quantity can no longer move a published one. That closes one source of movement;
it does not make this endpoint resolvable, which is what the 300-seed replay above measures.

The per-detector verdicts are not seed-sensitive in that way. Over the same 300 seeds, all
13 L0 intervals contain 0.500 at every seed; the narrowest margin from 0.500 at any seed is
0.0011 (Reward Variance Ratio), about three times the scatter that decides the suite mean's
endpoint. The control is carried detector by detector, which is where it is robust.
*(2026-11-27: this is the replay of the 2026-08-04 draws. On the re-draw one of the 13,
Reward CUSUM, excludes 0.500 at the published seed; the replay has not been run on the
re-draw.)*

**What the 27 families are (2026-11-19).** Nor does the check license a claim that the
proxy is matched on those 27 families. Since its 2026-11-19 regeneration (`d4b700d`) the
admission ledger covers all 33 families, 123 cells. It admits 42 of them, certifying a
matched proxy on each; a 43rd, `proxy_correlation_gaming` @ 0.60, passes both proxy
criteria and fails only `behavioral_separated`. It finds the same six families DEGENERATE
(30 cells) and measures 51 NOT ADMITTED, 50 of them failing a proxy criterion: 24 fail
`proxy_matched`, 40 fail `proxy_distribution_matched` (14 fail both), and 1 fails
`behavioral_separated` alone. The degenerate-free column above holds out only the
DEGENERATE families, as the L0 aggregate does, so the NOT ADMITTED cells a detector is
scored on are in it: all 51 for the four detectors scored on every family, 33 for the four
not applicable on the 7 families of 40 episodes, and 9 for the five not applicable on the
19 of 40 or 60. Whether they should be is an open question for the author
([ROADMAP.md](../../ROADMAP.md)). The degenerate set is the same six families, so the
regeneration changes nothing in the table.

**Recomputed on 2026-11-26.** The table, the suite-mean interval and the seed replay above
are recomputed from the re-scored replicates (CI runs 45747922473 and 45776113470,
`827ae57`; the merge kept each draw's values and applied only the new not-applicable
cells). The previous version,
dated 2026-08-06 (`97ac575`), predated both re-scores of the L0 rows: `253c7ec`
(2026-09-03), which recorded Reward Skewness not applicable on the 19 families that run 40
or 60 episodes, and `f83a152` (2026-11-26), after which every other L0 detector whose score
was a constant below some run length declares that length too and is not applicable below
it: Reward Peak, Reward Autocorrelation, Reward Trend and Variance Window (100 episodes) on
the 19 families of 40 or 60 episodes, and Gradient Reversal (60), Reward CUSUM (51), Max
Plateau and Reward MLP (50) on the 7 of 40. Reward KDE, Reward Variance Ratio and Spectral
Reward declare 22, 21 and 24, below every family's run length, so their rows are unchanged.
Until then nine rows averaged a fallback 0.500 that was not a measurement, in both columns
and in every draw, and like the degenerate cells it pulled each of them toward 0.500. With
it gone no verdict flips, the denominator effect is larger (at most 0.0032 then, 0.0088
now), and the intervals of the five 100-episode detectors are wider, because they rest on
14 families (8 without the degenerate ones) rather than 33 (27). The L0 suite mean went
from 0.4966 [0.4934, 0.4999] on 2026-08-06, through 0.4972 [0.4939, 0.5005] after the
Skewness re-score, to 0.4951 [0.4899, 0.5000] now. *(2026-11-27: "now" in this paragraph
is 2026-11-26. On the re-draw the suite mean is 0.4977 [0.4918, 0.5036]; see the end of
this file.)*

**The draws varied rollout seeds only (2026-11-27).** Every interval in this file, except
those it quotes from the re-draw, is over the 20 draws made on 2026-08-04, and all 20 were
built at layout 0: until `827aec9`,
`FamilyRegistry.generate_suite` dropped its `layout_seed`, so draws 1 to 19 resampled
only their rollout seeds, whatever layout seed they record
(`leaderboard/v5_replicated.json` read `layouts_resampled: false`). The intervals are
over rollout seeds at one layout. For these L0 rows the behavioural sign a layout also
sets does not matter, since `restrict()` hides `behav_trace` below L2, and the layout
seed builds a different environment in two families only, `proxy_correlation_gaming` and
`rlhf_reward_model_overopt`: 12 families never read it, and 19 use it only to offset
their rollout seeds.

**Re-drawn on 2026-11-27; not recomputed here.** Those 20 draws have been replaced by a
re-draw at one clean commit under the CI lock and the six numerics pins, each draw at its
own layout (CI run 46011019246 at `54cd63c`, `b93c351`; the `replicates` block reads
`draw_schema` 2 in all 20, `layouts_resampled: true` and `single_source: true`). This
file remains the analysis of the 2026-08-04 draws: its table and its seed replay have not
been redone on the re-draw, and only the departure counts at the end of this paragraph
have. The control it examines did change. On the re-draw
12 of the 13 L0 detectors' published intervals contain 0.500, and Reward CUSUM's,
0.486 [0.475, 0.497] over its 101 cells, lies wholly below it: Reward CUSUM separates the
variants weakly, in the direction that would not flag hacking. On the 2026-08-04 draws
its interval was 0.498 [0.490, 0.507], and all 13 contained 0.500. Whether that interval
matters for the control is the author's decision. The level's best-detector maximum still
excludes 0.500, as the maximum of 13 detectors does by selection, at 0.575
[0.565, 0.586]; the suite mean, 0.4977 [0.4918, 0.5036] against 0.4951 [0.4899, 0.5000]
on the 2026-08-04 draws as re-scored, now contains 0.500 with room rather than at its
endpoint. Among the six DEGENERATE families, `rlhf_reward_model_overopt`
no longer returns exactly 0.500 in every draw: 6 of its 260 L0 values depart, against 0
on the 2026-08-04 draws. `distributional_shift`, `monitored_sandbagging` and
`shortcut_exploitation` still depart in none, `orbit_chirality` in 21 (27 before) and
`physics_exploitation` in 37 (37).

Regenerate: `scripts/aggregate_replication.py` for the headline; this table is recomputed from
`results/replication/replicate_*.json` per-family values with the ledger's degenerate set:
per draw, the mean of the detector's non-null per-family values (a not-applicable family is
null and left out), then a percentile bootstrap of those 20 values, 10,000 resamples under
`default_rng(20260804)`.
The seed replay re-runs `bootstrap_ci`'s body -- 10,000 resamples of the 20 per-replicate
values -- under `default_rng(s)` for s in 0..299, on the L0 suite mean and on each of the 13
L0 detectors.
*(2026-11-27: the table, the suite-mean interval and the replay are of the 2026-08-04
draws as last written, re-scored, in `827ae57`. To reproduce them, read the replicates at
that commit (`git show 827ae57:results/replication/replicate_000.json`, and so on to
`replicate_019.json`). Run on the committed files, the recipe gives the re-draw's figures
instead; this file does not tabulate them, and quotes only the headline's, in the
re-draw paragraph above.)*
