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
Eight other L0 detectors still return a constant below their own windows, which pads the
same control; that was found on 2026-11-19 and is open (Milestones).

## Retired

- The RHOB Transfer Score as a headline metric, and any leaderboard ranked on it.
- The challenge cadence and the workshop shared task built around it.
  [docs/WORKSHOP_PITCH.md](docs/WORKSHOP_PITCH.md) is kept as a withdrawn record.
- The internal onset column, for the reason above.
- Any claim that RHOB detects reward hacking. The replicated result is that the three
  sub-oracle rungs sit within 0.09 of chance for unsupervised detectors: max AUROC L0 0.551
  [0.541, 0.562], L1 0.534 [0.515, 0.554] and L2 0.587 [0.576, 0.600], against 0.975
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
  It is packaged for PyPI, so `pip install cannotfail` and `uvx cannotfail` work once 0.1.0
  is uploaded; [docs/RELEASING_CANNOTFAIL.md](docs/RELEASING_CANNOTFAIL.md) holds the
  steps. Until then it installs only from this repository, by the `git+https` line in its
  [README](tools/cannotfail/README.md#install). Its own CI,
  [`.github/workflows/cannotfail.yml`](.github/workflows/cannotfail.yml), installs the built
  wheel the way an outside maintainer would and runs the README's examples from it. Green
  says the tool's tests pass and its examples print what its README shows, nothing about the
  checks it is run on. A run on another benchmark reaches the author through
  [an issue form](https://github.com/Aarav500/rhob/issues/new?template=cannotfail_run.yml),
  or by email to the address in [SECURITY.md](SECURITY.md#private-reports-that-are-not-security-issues) when it should not be public.
- **One convention for every published number — open; done for the missing channel on
  the L1 rows and for the Reward Skewness row on both boards (the 20-draw Skewness re-score
  2026-09-03, the single-draw board 2026-11-19), not for the short horizons of eight other
  L0 rows, three L2 rows and the L1 rows on `continuous_camping`.** A cell a detector could not have scored is N/A, never a 0.5. The L1 row was fixed in the
  2026-08 correction. The L0 Reward Skewness row had the same defect from a different cause
  (its windows need 100 episodes; 19 families run 40 or 60) and was re-scored on all 20
  draws in `9da6b21`: those 19 families are now N/A, leaving 0.504 [0.494, 0.514] over 64
  cells on 14 families. Five of the 14 are still at exactly 0.500, from a near-constant
  proxy rather than the horizon. That is disclosed and not fixed. The single-draw board
  (`leaderboard/v5_leaderboard.json`) was regenerated on CI on 2026-11-19 (run 44447220658,
  `52656cb`) under the shipped configuration, with the behavioural sign randomized: its 8
  L1 rows are `null` on 88 of 123 cells and its Reward Skewness row on 59. Its ladder of
  means is L0 0.4876, L1 0.5701, L2 0.5716, L3 0.9830, against 0.4866, 0.5200, 0.7213 and
  0.9830 on the 2026-07-17 board, and of maxima 0.5351, 0.9270, 0.9430 and 0.9830
  ([CHANGELOG.md](CHANGELOG.md)). Still open: the short-horizon rows in the next item, and
  `leaderboard/cross_family_transfer.json`, which has not been regenerated and predates both
  the provenance block and the N/A schema.
- **Eight L0 detectors publish a constant below their windows — open; the fix changes the
  L0 ladder.** Reward Skewness is the only L0 detector that declares `min_episodes`. Below
  their windows, Reward Peak, Reward Autocorrelation, Reward Trend and Variance Window
  (100 episodes), Max Plateau and Reward MLP (50) and Gradient Reversal (60) return 0.5, and
  Reward CUSUM (50) scores every run 0, which ties at 0.5. Both boards record that constant
  as a measured 0.5: the first four on the 19 families that run 40 or 60 episodes, the other
  four on the 7 of those that run 40. On the single-draw board that is 104 of the
  per-family values behind the L0 mean of 0.4876, and in all 20 draws every one of those
  cells is 0.5. It is the defect `584ba13` fixed for Reward Skewness. The admission gate is
  not affected: it reads none of the eight. The same defect sits outside L0, in 41 more
  per-family values: three L2 detectors (Feature Consistency below 100 episodes, on the 19
  families; Feature Magnitude and Reward-Feature Correlation below 50, on the 7), and all
  eight L1 detectors on `continuous_camping`, whose 40-episode runs are below every L1
  window; those move the L1 and L2 figures, not the L0 ladder. The fix: declare each
  detector's minimum horizon; re-score those rows on the single-draw board and on the 20
  draws by the procedure of `scripts/merge_rescored_detector.py` (the same draws, and every
  cell scored before comes back unchanged); rebuild `leaderboard/v5_replicated.json`; then
  update every figure quoted from either board that moves, the L0 ladder first. Dropping
  the 104 L0 values from the committed single-draw board gives an L0 mean of 0.4834 and a
  best of 0.5474 (Reward CUSUM, over 20 families), against 0.4876 and 0.5351, and moves
  Reward Peak from third to second.
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
  board or the admission ledger is from the versions before their 2026-11-19 regeneration
  and must be updated to the regenerated ones before submission.
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
  six distinct values, and the verdict is `ok`. Recomputing the 30 from score
  vectors would need the runs regenerated, since the ledger stores aggregates.
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
