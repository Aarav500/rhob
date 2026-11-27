# RHOB roadmap

## What changed

This file used to aim RHOB at becoming "the shared, default benchmark for reward-hacking
detection — the way ImageNet became the default benchmark for image classification", with
the RHOB Transfer Score (RTS) as its headline number and a twice-yearly "RHOB Challenge" to
give the field a reason to keep coming back. That plan was withdrawn on 2026-09-02
(`838a8d7`), and its reasons were restated more exactly on 2026-10-31. The audit behind
those reasons became the paper this repository is now the case study for.

- **The headline RTS did not measure transfer.** The top-5 ensemble's 0.994 on held-out
  families fell to 0.508, chance, once each family's behavioural feature stopped carrying
  the label in its sign. That is one run of three trials, and in it the label-fitted
  Trajectory MLP kept 0.706 ([docs/figures/sign_randomization_impact.json](docs/figures/sign_randomization_impact.json)),
  so what survives is a label-fitted detector's transfer, not the headline. Even before
  that, the held-out families were written by one author under one construction recipe, so
  the most RTS could show was transfer within a single authored convention.
- **RHOB cannot be the default measure of detection.** In 27 of its 33 families the hack
  itself is authored, and in all 33 so is the flaw it exploits. A benchmark built that way
  can say what a signal contains. It cannot say how well a detector catches hacking an
  optimizer found, which is the question the ImageNet framing promised to answer.
- **The onset column the acronym refers to could not fail.** Every one of the 50 cells in
  the 2026-08-03 admission ledger had `onset_std` exactly 0.0, because those onsets are
  authored, so a constant prediction equal to the label scored a perfect error of 0.000. Two
  things follow. `584ba13` removed the internal onset column from the leaderboard: `CellResult` no
  longer carries it, and it is still computed only for `external/hvta`, where the label was
  not written by us. And `rhob.audit` coded `onset_localizable` as degenerate (`e9a1ace`).
  The full ledger (2026-11-19, 123 cells) still has 0.0 in 118 cells. In the other 5, the
  tiers of `continuous_camping` and `gridworld_camping`, it is nonzero, and
  `onset_localizable` passes there too. Measured over the whole shipped design, which the
  ledger now covers, the statistic takes six values, so the self-coding no longer codes the
  criterion degenerate (see Milestones).

The withdrawn text is in git history rather than here: last at `06df9f3` on the branch
that carried this rewrite, and at `9352529`, the last version both lines of history share,
on main's. A plan whose headline metric is retracted should not sit
beside one that is current.

## The goal now

What the repository is: a 33-family matched-pair benchmark with an admission gate, a
20-draw replicated leaderboard, and the audit record of the checks in it that could not
have failed. The benchmark is the evidence. The paper is the claim.

What it is for: making the failure class findable in other people's benchmarks. The
contribution is the class — checks that cannot return a negative verdict under the design
that ships — and the diagnostics that find it. RHOB is the case study, and the main evidence
that the diagnostics find real defects. The paper counts seven instances in it, three of
them introduced by the fix to an earlier one. Two more were found in the released artifact
after the paper (`584ba13`): Reward Skewness returned a constant 0.5 below its 100-episode
horizon, on 19 families, which padded the L0 negative control, and `true_reward_diverges`
reported FAIL on arms tied by arithmetic, with no tie guard; it now returns DEGENERATE.
Eight other L0 detectors returned a constant below their own windows, which padded the
same control, as did every L1 detector on `continuous_camping` and three L2 detectors on
the short-horizon families. That was found on 2026-11-19 and fixed on 2026-11-26
(`f83a152`): each now declares `min_episodes`, and both boards were re-scored with those
cells N/A (Milestones). An external baseline that is not on the board had a constant
score too: Bayesian Online Changepoint Detection scored `max_t P(r_t = 0)`, which a
constant hazard pins to the hazard, so every HVTA run scored 0.004 and its onset rule
could never fire. It now scores `1 - P(no changepoint over the run)`, and HVTA's primary
draw was re-scored with it (`827aec9`, `f192797`). Whether either is a further instance
in the paper's count is the author's decision.

## Retired

- The RHOB Transfer Score as a headline metric, and any leaderboard ranked on it.
- The challenge cadence and the workshop shared task built around it.
  [docs/WORKSHOP_PITCH.md](docs/WORKSHOP_PITCH.md) is kept as a withdrawn record.
- The internal onset column, for the reason above.
- The v3.2 tables in `leaderboard/` (`leaderboard.json`, the standings in its `README.md`,
  `by_mechanism.md`, `by_difficulty.md`) and the two scripts that wrote them,
  `scripts/populate_leaderboard_v32.py` and `scripts/populate_v3_leaderboard.py`
  (`827aec9`). Nothing read them, they predated the 2026-08-03 audit and the
  not-applicable rule, and every detector they ranked except
  `Ensemble(BehavioralThreshold x2)` is on both v5 boards. They remain in the history
  ([leaderboard/README.md](leaderboard/README.md) gives the commands).
  `python -m rhob submit` now records self-reported submissions in `leaderboard/community/`.
- Any claim that RHOB detects reward hacking. The replicated result is that the three
  sub-oracle rungs sit within 0.10 of chance for unsupervised detectors: max AUROC L0 0.568
  [0.555, 0.580], L1 0.538 [0.517, 0.561] and L2 0.599 [0.589, 0.610], against 0.975
  [0.971, 0.978] for the L3 oracle, over 20 draws
  ([leaderboard/v5_replicated.json](leaderboard/v5_replicated.json)).
- The old v1.5, v1.6 and v2.0 milestones — RTS for every detector, a multi-agent
  extension, 25 or more families. Each would enlarge a benchmark whose central construction
  is now the thing in question.

## Milestones

- **`cannotfail` 0.1 — done (2026-10-31); its PyPI release prepared, not published
  (2026-11-15).** D1 and D2 as a standalone package in
  [`tools/cannotfail`](tools/cannotfail), installable without RHOB and held to RHOB's
  admission gate by [`tests/test_cannotfail_parity.py`](tests/test_cannotfail_parity.py).
  Since `827aec9` that test also holds the third copy of the gate's tie rule, the Inspect
  eval's (`inspect_eval/hvtb_hack_detection/metrics.py`), to the gate: its tolerances and
  its rule, read with `ast`, so `inspect_ai` need not be installed. It is packaged for
  PyPI, so `pip install cannotfail` and `uvx cannotfail` work once 0.1.0 is uploaded;
  [docs/RELEASING_CANNOTFAIL.md](docs/RELEASING_CANNOTFAIL.md) holds the steps. Until then
  it installs only from this repository, by the `git+https` line in its
  [README](tools/cannotfail/README.md#install). Its own CI,
  [`.github/workflows/cannotfail.yml`](.github/workflows/cannotfail.yml), installs the built
  wheel the way an outside maintainer would and runs the README's examples from it. Green
  says the tool's tests pass and its examples print what its README shows, nothing about the
  checks it is run on. A run on another benchmark reaches the author through
  [an issue form](https://github.com/Aarav500/rhob/issues/new?template=cannotfail_run.yml),
  or by email to the address in [SECURITY.md](SECURITY.md#private-reports-that-are-not-security-issues) when it should not be public.
- **One convention for every published number — done on both boards for the missing
  channel on the L1 rows, for the Reward Skewness row (the 20-draw Skewness re-score
  2026-09-03, the single-draw board 2026-11-19) and for every other detector's short horizon
  (2026-11-26); for the cross-family transfer and
  `docs/figures/sign_randomization_impact.json`, code done (`827aec9`, `f0caa3d`), CI runs
  pending (below); the v3 tables in `leaderboard/` were retired instead (`827aec9`).** A
  cell a detector could not have scored is N/A, never a 0.5. The L1 row was fixed in the
  2026-08 correction. The L0 Reward Skewness row had the same defect from a different cause
  (its windows need 100 episodes; 19 families run 40 or 60) and was re-scored on all 20
  draws in `9da6b21`: those 19 families are now N/A, leaving 0.504 [0.494, 0.514] over 64
  cells on 14 families. Five of the 14 are still at exactly 0.500, from a near-constant
  proxy rather than the horizon. That is disclosed and not fixed. The single-draw board
  (`leaderboard/v5_leaderboard.json`) was regenerated on CI on 2026-11-19 (run 44447220658,
  `52656cb`) under the shipped configuration, with the behavioural sign randomized, and
  re-scored on 2026-11-26 with every detector's horizon declared (CI run 45747902322,
  `03d978e`): its 8 L1 rows are `null` on 92 of 123 cells (26 families), its Reward
  Skewness, Reward Peak, Reward Autocorrelation, Reward Trend, Variance Window and Feature
  Consistency rows on 59, and its Reward CUSUM, Max Plateau, Gradient Reversal, Reward MLP,
  Feature Magnitude and Reward-Feature Correlation rows on 22. Its ladder of means is L0
  0.4834, L1 0.5789, L2 0.5771, L3 0.9830, against 0.4866, 0.5200, 0.7213 and 0.9830 on the
  2026-07-17 board, and of maxima 0.5474, 0.9820, 0.9430 and 0.9830: L1's best, over 7
  families and 31 cells, is above L2's and within 0.001 of L3's
  ([CHANGELOG.md](CHANGELOG.md)). Still open: the transfer measurement, whose only
  committed artifact, `leaderboard/cross_family_transfer.json`, is the 2026-07-09 record
  and predates the provenance block, the N/A schema and the horizon declarations; and
  `docs/figures/sign_randomization_impact.json`, scored before the declarations (below).
  The v3 tables in `leaderboard/`, which predated the N/A schema, were retired
  (`827aec9`; see Retired).
- **Every detector declares the run length its score needs — done (2026-11-26); it changed
  the L0 ladder.** Until `f83a152`, Reward Skewness was the only L0 detector that declared
  `min_episodes`. Below their windows, Reward Peak, Reward Autocorrelation, Reward Trend and
  Variance Window (100 episodes), Max Plateau and Reward MLP (50) and Gradient Reversal (60)
  returned 0.5, and Reward CUSUM (51: at exactly its 50-episode baseline the accumulation
  loop never runs) scored every run 0, which ties at 0.5. Both boards recorded that constant
  as a measured 0.5: the first four on the 19 families that run 40 or 60 episodes, the other
  four on the 7 of those that run 40. On the single-draw board that was 104 of the
  per-family values behind the L0 mean of 0.4876, and in all 20 draws every one of those
  cells was 0.5. It is the defect `584ba13` fixed for Reward Skewness. The admission gate is
  not affected: it reads none of the eight. The same defect sat outside L0, in 41 more
  per-family values: three L2 detectors (Feature Consistency below 100 episodes, on the 19
  families; Feature Magnitude and Reward-Feature Correlation below 50, on the 7), and all
  eight L1 detectors on `continuous_camping`, whose 40-episode runs are below every L1
  window. Each detector now declares its minimum horizon, derived from its own attributes,
  and a cell below it is N/A (`f83a152`); `EnsembleDetector` needs the largest of its
  members' minimums, and Perfect Feature Oracle declares the `behav_trace` channel it reads.
  The five external baselines, scored on HVTA and not on the board, declare theirs since
  `827aec9`: Page-Hinkley 3, Isolation Forest 2, AR(p) Residual `order + 1` (6 by
  default), PCA Reconstruction its 100-episode `steady_window`, and Bayesian Online
  Changepoint Detection 5. `tests/test_detectors/test_min_episodes.py` holds every shipped
  detector to its declaration: constant one episode below it, varying at it.
  Both boards were re-scored on CI: the single draw in run 45747902322 (`03d978e`), which is
  the 2026-11-19 board with exactly those 145 values set to `null`, and the 20 draws in run
  45747922473, merged by run 45776113470 (`827ae57`) with
  `scripts/merge_rescored_detector.py --keep-old-values`: every draw keeps the value it was
  scored with and only the new not-applicable cells are applied, because 414 of the 23,480
  re-scored cells did not reproduce on the CI lock (319 on `sequence_keyword_stuffing`; see
  the next item), after which `scripts/aggregate_replication.py`
  rebuilt `leaderboard/v5_replicated.json` and `docs/figures/replication_summary.md`. The
  summary was rendered again by the updated aggregator, whose sentences on the label-fitted
  climb and on each level's denominators are now derived from the data. The HVTA
  channels were re-scored too (`1e38a27`, 132 entries now N/A). Re-scored, the single-draw
  L0 mean is 0.4834 and the best 0.5474 (Reward CUSUM, over 20 families), against 0.4876
  and 0.5351, and Reward Peak moves from third to second (0.5375). On 20 draws the L0
  control mean is 0.4951 [0.4899, 0.5000], against 0.4972 [0.4939, 0.5005]; its upper end
  is 0.50001 to five decimals, so 0.5 is only just inside. L2 minus L1 on maxima is −0.0555
  [−0.0609, −0.0505] and now separates, with L1 above L2; on means L2 is still above L1
  (+0.0118 [+0.0008, +0.0235]). So on the all-detector board both statistics separate every
  rung, and they disagree on the direction of that one. In the unsupervised partition,
  which the top-level `statistic_disagreement` does not cover (since `827aec9` the
  aggregator applies the same rule to it, in
  `supervision.statistic_disagreement_unsupervised_only`, which lists this step), the L1
  minus L0 step now depends on the statistic: on maxima −0.0293 [−0.0537, −0.0051]
  (positive in 5 of 20 draws) separates, with L1 below L0; on means −0.0022
  [−0.0217, +0.0167] (11 of 20) does not. Before the re-score neither statistic separated
  it. L1's best, State Divergence, is within 0.001 of the L3 oracle on the single draw (0.982 against 0.983) and above it on 20 draws (0.998
  against 0.975), over 31 cells in 7 families against the oracle's 123 in 33, so that is
  not a like-for-like comparison. The figures the repository's docs quote from either board were
  updated with the re-score; the paper's are under the venue milestone below.
- **Re-draw the 20 replicates on today's code — code done (`827aec9`, `d36d60f`,
  `320b7b7`); CI run pending.** The 20 draws behind `leaderboard/v5_replicated.json` were
  made on 2026-08-04, under Python 3.11 and numpy 2.4.6 with no git commit recorded, and
  every later fix has been merged into them detector by detector. `e7bb3f3` (2026-09-19)
  rewrote `sequence_keyword_stuffing`, whose keyword branch had been dead code, so every
  detector's 20-draw value on that family still measures the old version: the 2026-11-26
  re-score could not reproduce 319 of its cells and kept the drawn values (see the item
  above). Nor did the draws vary the layout. Until `827aec9`,
  `FamilyRegistry.generate_suite` dropped its `layout_seed`, so draws 1 to 19, which
  record layout seeds 1 to 19, were built and oriented at layout 0 and resampled only
  their rollout seeds: the committed intervals are over rollout seeds at one layout, with
  every family under one behavioural orientation. A re-draw adds the orientation in every
  family, since each family's behavioural sign is drawn per layout, and a new environment
  layout in two: 12 of the 33 families never read the layout seed, 19 use it only as an
  offset to rollout seeds the draws already resample, and `proxy_correlation_gaming` and
  `rlhf_reward_model_overopt` build a real layout from it. The tooling is in place.
  Replicates record `draw_schema` 2, and a draw without it is refused by
  `scripts/merge_rescored_detector.py` and, for `artifacts=replicated`, by
  `regenerate.yml`'s plan job (`320b7b7`). `regenerate.yml`'s `redraw` mode draws the
  board and all 20 replicates, every detector, at one commit under the CI lock and the six
  numerics pins, after one `calibrate` job derives the calibrations for all of them.
  `scripts/check_redraw.py` refuses the set unless it is complete, at draw schema 2, from
  one clean commit under the pins themselves (`d36d60f`), and replicate 0 equals the board
  cell for cell; `scripts/aggregate_replication.py --require-single-source` then rebuilds
  the intervals. The aggregator now gives `v5_replicated.json` its own `provenance` block and
  a `replicates` block that tallies the inputs' sources (`827aec9`). Re-aggregating the
  committed draws with it changed no number, and their `replicates` block reads
  `layouts_resampled: false` and `single_source: false`. The re-draw moves every 20-draw
  figure, so the docs that quote them move with it.
- **Re-run the artifacts scored before the horizon declarations — code done (`827aec9`,
  `f0caa3d`); CI runs pending.**
  `leaderboard/cross_family_transfer.json` and
  [docs/figures/sign_randomization_impact.json](docs/figures/sign_randomization_impact.json)
  were not re-run with `f83a152`. The held-out families of both transfer runs all run at
  least 100 episodes, and `continuous_camping` (40) is among their six training families,
  though the Reward MLP and State Divergence fits, it turned out, already dropped its runs.
  Whether the ensemble's 0.994 to 0.508 or Trajectory MLP's 0.706 (What changed) moves
  has not been measured. The sign-randomization run scores all 33 families, so its L2
  ladder (mean 0.7229 before, 0.5629 after) still averages in the short-horizon fallbacks
  of Feature Consistency, Feature Magnitude and Reward-Feature Correlation, and its
  invariance check (123 of 123 cells identical before and after) counts the cells below
  the declared horizons of those three and of Reward CUSUM (L0) and State Coverage Rate
  (L1) too; its before-and-after differences for Behavioral Threshold, Centroid Tracker and
  Trajectory MLP do not read them. The transfer script now applies the benchmark's
  not-applicable rule, `rhob.v3.benchmark.not_applicable_reason`, outside the harness,
  and the sign-randomization script measures transfer through the transfer script's own
  code (`827aec9`, `f0caa3d`). `scripts/cross_family_transfer.py` applies the rule to the
  pooled fit and to every held-out cell, records a cell whose frozen model cannot consume the family's
  representation as N/A with its reason instead of scoring it 0.5 (State Divergence's
  published transfer row was 1200 such imputed scores), cross-validates the Top-5
  ensemble's train AUROC with a fold-fitted Trajectory MLP, and writes
  `leaderboard/cross_family_transfer_v5.json`; it refuses to overwrite the 2026-07-09 file,
  which is kept as the record. `scripts/measure_sign_randomization.py`, which had crashed
  on its first board cell since the onset column was removed (2026-09-02), runs again: its
  invariance check compares only the cells scored in both modes, and its `ci` subcommand
  runs the published design end to end. `regenerate.yml` runs the two as its `transfer`
  (5 trials) and `signrand` modes. The sign-randomization measurement's `transfer` stage
  keeps `--n-trials` at 3, the published design's (it was 5 until 2026-11-27); whether
  that default stays is open, the author's decision. `scripts/populate_v3_leaderboard.py`
  and the v3 tables it wrote to `leaderboard/` (`leaderboard.json`, `README.md`,
  `by_mechanism.md`, `by_difficulty.md`, last committed on 2026-07-07) predated the N/A
  schema as well. They were retired (`827aec9`): nothing read them, and every detector
  they ranked except `Ensemble(BehavioralThreshold x2)` is on both v5 boards.
- **Extend the admission ledger to the short-horizon families — done (2026-11-19).** The
  nightly smoke screen has been re-measured with Reward Skewness declaring its horizon: 19
  short-horizon cells admit and 14 fail on a real late-window density leak (`9da6b21`; the
  14 are marked `xfail(strict=True)` per tier in `tests/test_v3/test_family_*.py`). The
  certification ledger now covers all 33 families and 123 cells (CI run 44447220658,
  `d4b700d`): 42 ADMITTED, 30 DEGENERATE and 51 NOT ADMITTED, against 15, 30 and 5 over the
  10 families and 50 cells of 2026-08-03. The 10 families already in it keep every verdict
  in all 50 of their cells, and the DEGENERATE families are the same six, so the 30
  calibration cells below are the same 30, still DEGENERATE. `proxy_matched` was measured
  on 93 cells and failed 24, where it had failed none of 20; `proxy_distribution_matched`
  failed 40 and `behavioral_separated` 1. Of the 73 new cells, 27 admit and 46 do not,
  among them all 13 MuJoCo cells and 14 of the 15 PettingZoo cells.
- **The 46 NOT ADMITTED cells in the 23 families new to the ledger — open.** For each
  family, fix its proxy matching or document the failure as a known leak. They are in 20 of
  the 23 families. 24 fail `proxy_matched`: in 8 the point estimate is outside the band,
  and 16 are underpowered, with the point estimate inside it and the 90% interval not, so
  the equivalence test could not establish matching at this design; in 15 of the 16 the
  interval also excludes 0.5. 35 fail
  `proxy_distribution_matched`, all but one on Reward KDE, and 14 fail both. One,
  `proxy_correlation_gaming` @0.6, fails only `behavioral_separated`. Each cell's binding
  measurement is in [admission/ADMISSION_LEDGER.md](admission/ADMISSION_LEDGER.md).
- **Whether the L0 aggregate should also hold out NOT ADMITTED cells — open, the author's
  decision.** The code holds out only the six DEGENERATE families
  (`summarize_access_levels`), so the 51 NOT ADMITTED cells are still in the L0 mean; 22 of
  the 27 families that mean is over have at least one. On 50 of the 51 the gate measured
  the proxy and could not certify it matched; the 51st, `proxy_correlation_gaming` @0.6,
  passes both proxy criteria and fails only `behavioral_separated`.
- **RHOB's self-coding on the full ledger — changed (2026-11-19); the paper's wording is
  the author's decision.** `docs/SURVEY_PROTOCOL.md` codes a criterion degenerate only when
  its statistic took one value across the whole shipped design. Measured over the 123-cell
  ledger, `onset_std` takes six values, so `onset_localizable` codes `not_applicable` and
  the post-audit coding leaves no finding ([audit/SELF_CODING.md](audit/SELF_CODING.md)).
  On the 50-cell ledger the row was first coded against (`e9a1ace`) it was degenerate, and
  the statistic is still 0.0 in the 118 cells of the 31 families whose onset label is the
  configured schedule. Wherever the paper states RHOB's self-audit result, it has to give
  the full-ledger coding or scope the earlier one to the 50-cell ledger before submission.
- **One maintainer outside the author's network runs it on their own score vectors — by
  2026-12-15.** A run of the diagnostics on a benchmark the author did not build.
- **The paper at a venue.** A submission to TMLR, after the paper's claims are made to
  match the repository's. The arXiv route has produced no endorsement; a journal needs
  none. Among those claims: App. G's thirty-second screen calls a design `CANNOT_FAIL`
  when a minimum detectable effect, computed with the null SD for the alternative too,
  exceeds the attainable range. That is a normal-approximation screen, not a power bound,
  and it does not show that no data can reject. Its own 5-vs-5 cell can reject: an exact
  two-sided test does at every complete separation (p = 0.0079), and the power
  computation the axis-A plan pins first reaches 0.80 at an AUROC of 0.944, inside the
  range ([docs/SURVEY_AXIS_A_PREREG.md](docs/SURVEY_AXIS_A_PREREG.md), "What is already
  known"). Also among them, and open: any figure the paper quotes from the single-draw
  board, the 20-draw board or the admission ledger must be updated before submission. The
  single-draw board and the ledger were regenerated on 2026-11-19, and both boards were
  re-scored on 2026-11-26 with every detector's horizon declared. The changes include the
  L0 control on 20 draws, now 0.4951 [0.4899, 0.5000] (was 0.4972 [0.4939, 0.5005]); L2
  minus L1 on maxima, which now separates with L1 above L2 (−0.0555 [−0.0609, −0.0505])
  where it was the one rung that separated under one statistic and not the other; and the
  unsupervised L1 minus L0 step on maxima, now −0.0293 [−0.0537, −0.0051], which separates
  where −0.0174 [−0.0391, +0.0017] did not, while on means it still does not (−0.0022
  [−0.0217, +0.0167]), so that step's status now depends on the statistic. HVTA's figures
  moved too, with the re-score of Bayesian Online Changepoint Detection (`827aec9`,
  `f192797`): on the primary draw it is now the strongest external baseline, 0.739
  against Page-Hinkley's 0.690, and the best onset correlation among detectors that fire
  on most runs is its r = 0.44 (primary draw only; its other three draws are pre-fix),
  where it was Reward Peak's 0.43.
- **The survey: the port, then axis A — open; a pre-registration of axis A's power re-run
  written (2026-11-16), awaiting its sealed record and then a deposit dated by a third
  party.** The analysis plan for the re-run on the 15 sub-class A claims is written before
  any row is filled, and before the computation it fixes has been run on them
  ([docs/SURVEY_AXIS_A_PREREG.md](docs/SURVEY_AXIS_A_PREREG.md)). It is not the first power
  figure on the survey: the paper already reports 15 of 20 recoverable designs
  underpowered, 0.750 [0.500, 0.958]. How many of the 15 claims are among those 20 the
  author commits to before registration, in a sealed record that is published with the
  results. The rows await the author's §5 coding sheet and a second human coder. The port
  is not a prerequisite for the re-run: the sealed record fixes which 15 rows it reads.
  Nothing from the re-run, aggregate counts included, is made public until each group's
  outreach window has closed.
  First the port: the paper's 77 preconditions as
  rows, reconciled against its Appendix F by `rhob.audit.frame.reconcile`, coded twice for
  claim shape per [docs/SURVEY_PROTOCOL.md](docs/SURVEY_PROTOCOL.md). Authors are contacted
  per [docs/AUTHOR_OUTREACH.md](docs/AUTHOR_OUTREACH.md) before any table naming them is
  published. Then axis A: the enumeration's search patterns select against
  equivalence-test vocabulary, and the pre-registered remediation (revised patterns, the
  three unaudited members, the power re-run) addresses it. It is the one item that changes
  what the paper is evidence *about*.
- **HVTB's onset-timestamp patch — verified, not submitted.** The patch for the 89
  hack-verifiable Terminal Bench tasks is verified in Docker on 89 of 89
  ([`external/hvtb/verify_all_results.json`](external/hvtb/verify_all_results.json),
  `ea898e4`) and has not been submitted upstream (`b7a0395`, 2026-09-21). It records
  wall-clock time, not a step index.
- **After 2026-12-15: test the no-learner objection on someone else's learner.** Wong,
  Engels and Nanda's RL reward-hacking environment is open source, with a run that has a
  loophole and a baseline that does not. The question is whether hacking that an optimizer
  found is invisible to the proxy, as RHOB's construction assumes. The outcomes are fixed
  before any data exists, and one of them is stated now: **if the proxy separates the
  hacking and legitimate runs, that is the headline**, and it means RHOB's matched-proxy
  regime may be an artifact of scripting. The in-repo precedent is the Pair 01 pilot
  ([results/emergent_pilot/](results/emergent_pilot/README.md), `c6bb6f6`). The shipped
  config of Pair 01, the environment behind `gridworld_camping`, authors both when its
  exploit tile becomes available (episode 250) and a 20-episode forced exploration burst,
  so the config places its onset. With both set to 0 the tabular Q-learner still finds the
  tile on 40 of 40 seeds, and the onset label dissolves: mechanistic onset SD 76.5
  episodes, bimodal, and firing on all 40 legitimate twins. That shows authoring is what
  pins onset. It does not test the proxy, which the pilot keeps matched by construction;
  the outside probe asks that separate question.
- **D3 in `cannotfail`** — only once a second benchmark's harness shows what a channel
  digest would need to be told.

## Not planned

Learned families as certified benchmark members, a multi-agent hacking class as a
contribution, or new learned-agent trace collection for RHOB's own families before the
TMLR submission. Each was designed and argued against; the arguments are in the paper's
limitations section. The ledger extension above reran the scripted families through the
gate and was not that. The learner probe above is scheduled after 2026-12-15, and it is a test of the
construction, not a new benchmark member.

## What RHOB itself is for now

- **A fixture for the diagnostics.** Its admission ledger is the calibration set. Fed the
  ledger's measured resolution and informativeness, `cannotfail` must mark the same 30 of
  its 123 `proxy_matched` cells degenerate as the gate does, and it must call
  `onset_localizable`'s statistic fixed over the 118 cells of the 31 families whose onset
  label is the configured schedule, for the reason under What changed. Over all 123 it
  does not: the 5 cells of `continuous_camping` and `gridworld_camping` give the statistic
  six distinct values, and the verdict is `ok`. The ledger stores aggregates, not score
  vectors, so since `827aec9` `tests/test_cannotfail_parity.py` also re-certifies the six
  DEGENERATE families at the ledger's design: `cannotfail` must recompute each per-layout
  resolution and dispersion from the captured score vectors bit for bit against the gate
  re-run on the same machine, match the ledger's per-cell means within 1e-12
  (resolution) and 1e-6 relative (informativeness), or 1e-15 absolute for a dust-level
  proxy, and call each cell degenerate on its own. Two of the six, one of each kind of
  degeneracy, run on every push and the other four nightly (`2130ab2`). Beside it,
  `tests/test_v3/test_ledger_reproduces.py` re-certifies four families at every tier the
  ledger scores, on every push, and requires their statuses and outcomes exactly and
  every number within measured tolerances, so for those families a change to a detector,
  a family or the gate cannot leave the ledger certifying code that no longer exists
  without a test failing (`827aec9`, `2130ab2`).
- **A worked example of a check built so it can fail.** The matched-pair construction and
  the admission gate. Its nightly smoke screen first failed twelve families (2026-08-17,
  recorded in `ba26a4a`). 33 of its 36 DEGENERATE cells turned out to be Reward Skewness
  tied below its horizon (`584ba13`; the docstring of
  [`tests/test_v3/test_gate_honest_verdicts.py`](tests/test_v3/test_gate_honest_verdicts.py)),
  the harness describing itself rather than the families. The re-measurement is under
  Milestones.
- **Not a leaderboard to rank detectors on**, for the reasons under What changed.

## Discoverability: the Inspect Evals Register

HVTB — the Terminal Bench 2.1 hack-rate eval, whose code has been at
[`inspect_eval/`](inspect_eval/) on main since `b083e55` (2026-09-19) — is listed in the
[Inspect Evals Register](https://ukgovernmentbeis.github.io/inspect_evals/evals/hvtb/),
maintained by Generality Labs with contributions from the UK AI Security Institute, Arcadia
Impact and the Vector Institute. The register lists this repository's `inspect-eval`
branch, whose root is that directory, pinned at `a2c7341`.

That is a distribution channel, not a milestone. What is listed measures agents' hack
rates; it is neither the diagnostics nor RHOB's detector suite, the register publishes no
numbers for it, and its checks are structural — the Contributor Guide's scoping, validity
and scoring standards are "not mandatory for registry acceptance." It does not count as an
outside user. What it plausibly buys is that someone searching for reward-hacking evals in
Inspect finds this repository without having been told about it, and nothing here
substitutes for asking people directly.

## What would change or accelerate this

- **One outside maintainer running the diagnostics on their own benchmark.** Every run so
  far is the author's own: on RHOB, and in the survey on other groups' published scores. The
  first run by a benchmark's own maintainer, who can contest the result, is qualitatively
  different. This replaces, on purpose, the bar
  earlier versions of this file set, an external detector submission to RHOB.
- **A co-author or named collaborator from an established lab**, which changes the
  project's provenance from one researcher's repository to something others have checked.
- **A per-step label for HVTB.** HVTB is an external corpus in which hacking is decidable
  by construction and caused by the agent rather than scheduled by the authors, which RHOB
  is not. Its label is per trial. A step-indexed onset from real agents would let a
  detector's timing be scored against hacking nobody authored; that is future work, and a
  different paper.
