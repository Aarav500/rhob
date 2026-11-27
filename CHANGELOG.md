# Changelog

All notable changes to RHOB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

## [Unreleased] — Audit Remediation: Corrected L1/L3 Figures, Equivalence-Tested Admission Gate, Published Ledger

A repository-wide audit (2026-08) found that several published figures were not
the measurements they were described as. This entry records what was wrong, what
the corrected numbers are, and what remains unfixed. **Previously reported L1 and
L3 aggregates were incorrect and are retracted here**; if you cited them, cite the
corrected values below instead. *(2026-11-19: the board has since been regenerated, and
its figures supersede the recomputed ones below; see the 2026-11-19 regeneration entry
below.)* *(2026-11-26: both boards have since been re-scored with every detector's run
length declared, and those figures supersede the 2026-11-19 ones too; see the 2026-11-26
entry below.)* *(2026-11-27: the 20 draws behind the replicated board were all built at
layout 0, so its intervals carry no layout variation, and Bayesian Online Changepoint
Detection's HVTA figures were those of a constant score and are re-scored on HVTA's
primary draw; see the 2026-11-27 entries.)*

### Fixed (2026-11-27): the layout seed never reached the families, so the 20 draws all ran at layout 0

`FamilyRegistry.generate_suite` took a `layout_seed` and dropped it: it built every cell
with `generate_pair_at(d)`, at layout 0 and stamped as layout 0, whatever was asked for.
It now calls `generate_pair_at(d, seed=layout_seed)` (`827aec9`). The layout seed reaches
a family in two ways, as the seed its `generate_pair` builds from and as the key its
behavioural sign is drawn from (`rhob.v3.sign_randomization.behav_sign`), and a draw at a
non-zero layout seed got neither.

What this changes for a reader of the published numbers:

- **The 20-draw board.** Its 20 replicates (`results/replication/replicate_*.json`),
  drawn on 2026-08-04, record layout seeds 0 to 19, but each was built and oriented at
  layout 0: only their rollout seeds (`seed_base`) differ. The intervals published from
  them in `leaderboard/v5_replicated.json` and `docs/figures/replication_summary.md`, the
  20-draw table in the 2026-11-26 entry below included, therefore measure how the board
  varies over rollout seeds at one layout, with every family under one behavioural
  orientation. As that they are correct; they say nothing about variation over layouts or
  orientations, which is what the summary's "layout seed and rollout seeds resampled
  together" claimed.
- **Not affected:** the single-draw board (`leaderboard/v5_leaderboard.json`), which is
  scored at layout 0 by design; the admission ledger and
  `scripts/measure_difficulty_calibration.py`, which call `generate_pair_at` with their own
  layout seeds; and the HVTA results.
- **What a re-draw can vary.** For 12 of the 33 families, which never read the seed, the
  fix changes only the orientation a draw is scored under. 19 use the seed only as an
  offset to their rollout seeds, which the draws already resampled, so for them too the
  fix adds the orientation and no new layout. 2 build a real layout from it:
  `proxy_correlation_gaming` its item grid and `rlhf_reward_model_overopt` its fitted
  reward model. A re-drawn interval covers the orientation in every family, and the
  environment layout in those two only. `tests/test_v3/test_replication_draws_differ.py`
  pins the forwarding, the per-layout orientation, and that varying only the layout
  changes the rollouts.

A draw made before the fix can no longer pass for one made after it.
`scripts/replicate_leaderboard.py` writes `"draw_schema": 2` into every replicate, and a
file without it predates the fix. `scripts/merge_rescored_detector.py` refuses to merge
re-scored rows into a draw older than schema 2, and `regenerate.yml`'s plan job refuses
`artifacts=replicated` while any committed replicate is older (`320b7b7`), before twenty
re-score jobs have run rather than after. `scripts/aggregate_replication.py` gives
`v5_replicated.json` its own `provenance` block and a `replicates` block, which tallies
the inputs' commits, trees, Pythons, package sets, platforms, numerics, CPUs, draw schemas
and re-scored rows and says whether the layouts were resampled (`layouts_resampled`) and
whether the set is single-source; `--require-single-source` refuses a set that is not. It
also applies the max/mean verdict rule to the unsupervised partition
(`supervision.statistic_disagreement_unsupervised_only`). Re-aggregating the 2026-08-04
draws changes no number and corrects two statements in `replication_summary.md`: the
draws resampled the rollout seeds only; and without the label-fitted detectors the
L1 minus L0 step is statistic-dependent, separating below zero under the max and not under
the mean (the figures are in the 2026-11-26 entry), rather than a step that "separates the
other way". For those draws the `replicates` block reads `layouts_resampled: false` and
`single_source: false`: no draw records its commit, a clean tree or its numerics, and each
carries 20 detector rows re-scored after it was drawn. That re-aggregation's provenance
was stamped from a clean tree at `f192797` under the numerics pins (`eaa75f8`), with
everything outside it unchanged. The Space's provenance panel reads `layouts_resampled`
and, for these draws, says that the rollout seeds were resampled with every draw at layout
0.

**Re-drawing.** `regenerate.yml`'s new `redraw` mode draws the single board and all 20
replicates, every detector, at one commit, and commits them together.
`scripts/v5_leaderboard_and_transfer.py --cells-out` writes the board's per-cell AUROCs to
a sidecar carrying the board's provenance, through the same `_cells_detail` the replicates
are written with. `scripts/check_redraw.py` refuses the set unless all of these hold:

- `replicate_000` to `replicate_019` are all present, each at draw schema 2, each the draw
  `draw_for(i)`, with one `n_seeds` equal to the board's;
- every file, the board and the sidecar score the 30 detectors in `DETECTORS` order, and
  every file and the board the registered families, with no errors;
- every file and the board come from one clean commit equal to `--expect-commit`, under
  one Python, one package set and one numerics signature, with no rows merged in after the
  draw;
- every file and the board ran under the six numerics pins themselves, and their kernels
  show the pins took effect: ATen at AVX2 wherever torch loaded, every OpenBLAS at Haswell
  (`d36d60f`). Agreement alone is not enough, since runners that all lost the pins agree
  with each other;
- every file has the sidecar's cell grid, and `replicate_000`, the board's own draw, equals
  the board cell for cell, up to `--max-board-mismatch` (default 0).

The set is then aggregated with `--require-single-source`.

### Fixed (2026-11-27): Bayesian Online Changepoint Detection (BOCPD) scored a constant, and its HVTA figures change

`BOCPDDetector` scored `max_t P(r_t = 0)`, the probability that a changepoint happened *at*
step `t`. With a constant hazard that is the hazard itself at every step, so every run
scored 0.004, its onset rule could never fire, and its published HVTA AUROCs were 0.500
on every cell of the Sudoku draw and, on the other channels, a ranking of rounding noise.
It now returns the score its docstring specifies, `1 − P(no changepoint over the run)`,
and onset is the first step at which that exceeds 0.5 (`827aec9`). Past 500 episodes,
where the run-length truncation can drop the full-length hypothesis, a dropped hypothesis
now stays dropped, rather than whichever survivor is last being read as "no change". A
hazard outside (0, 1) raises `ValueError`; it used to produce −inf or NaN.

On HVTA (`external/hvta/`) the primary Sudoku pairs draw was re-scored for this detector
alone by the new `rescore_detector.py`, which merges one detector's rows into
`hvta_results.json` and records the re-score under `rescored`. In `RESULTS.md`:

- its mean AUROC over the 10 non-degenerate cells goes from 0.500 to 0.739, which makes it
  the strongest external baseline, ahead of Page-Hinkley at 0.690, and its normalised
  onset MAE from 1.000 to 0.296;
- it now fires, on 94% of hacking runs and 76% of legitimate twins. That gap, +0.17, is
  the largest in the suite (it was Centroid Tracker's +0.10), and the finding that every
  detector that fires also fires on the legitimate twin stands;
- among detectors that fire on a majority of runs, the best onset correlation is r = 0.44
  (BOCPD, on the primary draw only), up from Reward Peak's 0.43. 8 of the 26 scoreable
  detectors never nominate an onset, not 9, and 1 of the 18 that do beats a constant
  prediction, as 1 of 17 did.

`score_channels.py` was re-run in full on the other three channels: BOCPD's mean over the
12 cells goes from 0.483 to 0.621, and no other row changes. Both re-scores were run again
on a clean tree at `320b7b7` and committed in `f192797`, so their records name that
revision with no uncommitted files; no figure moved.

The three replicate seed draws (`replicates/seedbase_{100,200,300}.json`) were not
re-scored: that needs each draw's frozen pairs, which are not in the repository
(`export_pairs.py` rebuilds them in the HVTA environment). `RESULTS.md` marks BOCPD's
figures in them *(pre-fix)* and leaves them out of every across-draw figure.
`make_results.py` works that out from the files it loads (`a6bb519`), so re-scoring a
replicate clears its markers; a full `score_detectors.py` run carries the primary draw's
fix history forward (`detector_fixes`), and a draw whose detector source differs from the
primary's is marked *(other source)* and left out the same way. A detector's source is
attested by a sha256 of its code without docstrings or comments, identical on Python 3.11
to 3.13, and by the file's git blob. `score_channels.py` now records the date, the RHOB
revision, uncommitted `src/` files and every detector's source, and `CHANNELS.md` gains a
Provenance section. A test fails when the committed HVTA artifacts no longer attest the
current `bocpd.py`. `RESULTS.md` also drops a sentence that was false: not every
observable-channel detector's onset correlation changes sign between draws, since Reward
Peak, Page-Hinkley and Reward Threshold keep theirs, at |r| ≤ 0.43.

The new score saturates: it is exactly 1.0 once `P(no change)` falls to `2**-54`, so runs
that changed more clearly than that tie. No committed HVTA run comes within 10 nats of it,
and the class docstring says to check for ties at 1.0 on sharper or longer reward streams.
Whether BOCPD should score the log-odds instead, which keeps ordering such runs, is open,
the author's decision.

All five external baselines now declare the run length their score needs: Page-Hinkley
Test 3, Isolation Forest 2, AR(p) Residual `order + 1` (6 by default), PCA Reconstruction
`steady_window` (100) and BOCPD 5. Each length guard reads the declaration; Isolation
Forest's late-window guard reads its module constant on purpose, since a standard
deviation needs two points. None of the five is on either board, and the declarations
alone move no artifact. `tests/test_detectors/test_min_episodes.py` holds them to the
board detectors' rule, and fails if a `PosthocDetector` the package defines is not
exported or not in one of its tables, unless it is named as needing constructor arguments.
`EnsembleDetector`'s docstrings now say that it takes the plain mean of its members' raw
scores, not of their ranks, and that its score is in-sample for any member fitted on the
runs it scores.

### Changed (2026-11-27): CI computes every artifact under six numerics pins, and every artifact records them

Every job of `regenerate.yml` now runs with `OPENBLAS_CORETYPE=Haswell`,
`OPENBLAS_NUM_THREADS=1`, `MKL_CBWR=COMPATIBLE`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`
and `ATEN_CPU_CAPABILITY=avx2`, set workflow-wide. Without them OpenBLAS, MKL and PyTorch
pick kernels by whichever CPU a hosted runner has, and threaded reductions sum in an order
that depends on the thread count, so two runners can disagree in the last bits of one fit,
and a re-draw runs on 21 runners. Haswell and AVX2 are the newest kernels every x86-64
hosted runner has, Intel or AMD.

For a reader of the published numbers: the artifacts committed before this change were
computed without the pins and record no `numerics` block (the re-stamp of
`v5_replicated.json`'s provenance in `eaa75f8` records them, but its 2026-08-04 draws
record none). Every artifact CI writes from now on is computed under them. A figure from a
pinned run need not match an unpinned run of the same code bit for bit, so quote each
figure with the environment it came from. `tests.yml` sets none of the pins.

Every artifact's `provenance` block gains `numerics`: `env` (the six variables, null when
unset), `cpu` (vendor, model, family, model number and stepping, and the ISA flags that
choose kernels, from `/proc/cpuinfo`), `torch_imported` and `torch_cpu_capability` (the
ATen kernel level, `AVX2` under the pin), and `blas` (the loaded BLAS libraries, with the
kernel OpenBLAS selected). The ATen level is read on every torch 2.x (`847a96f`):
`torch.backends.cpu.get_cpu_capability()` first shipped in 2.1 and `pyproject.toml` allows
2.0, so the block falls back to the `CPU capability usage` line of
`torch.__config__.show()`, which gives the same string. Stamping never imports torch.
`scripts/admission_ledger.py` refuses to merge shards certified under different pins,
nulls the merged ledger's `numerics.cpu` as it already nulled its CPU, and lists the
shards' CPU vendors. Its determinism note now says a re-run reproduces the results byte for
byte on the same commit, locked environment and CPU model, and not across CPU models. A
later `ledger` or `both` dispatch certifies under the pins, which the committed ledger was
not.

The calibration cache's key changes twice over. Each module it reaches is hashed as code,
its syntax tree with docstrings removed, not as text, so comments, layout and docstring
edits no longer invalidate an entry and any other edit still does. Of the 11 commits that
touched `src/` since the import walk landed (`9cc96e2`), the text key had changed every
calibrated family's key on 5, by docstring or comment edits alone, each at the cost of a
cold recalibration of about 43 minutes on CI; the code key keeps all 11. And the
environment token now hashes the six numerics variables (`"unset"` when absent), so
calibrations derived with and without the pins never share an entry: a local
`scripts/warm_calibration_cache.py` serves only a test run made with the same values of
those variables, or with none of them set for either. Every calibrated family's key
changes once with this release, and CI's calibration tarball starts a new `calib-v2`
series, so the first CI run on it recalibrates cold.

### Fixed (2026-11-27): cross-family transfer and the sign-randomization measurement leave what they cannot measure N/A, and both scripts run again

`scripts/cross_family_transfer.py` now applies the benchmark's own not-applicable rule,
`rhob.v3.benchmark.not_applicable_reason` (newly public; `Benchmark.evaluate` calls it,
with unchanged reason strings), outside the harness. A family that lacks the channel a
detector reads, or whose runs are shorter than the detector's `min_episodes`, contributes
no rows to the pooled fit, and is listed in `train_families_excluded_from_pooled_fit`; as
a held-out family it is not scored. Two more cases leave a held-out cell `null`, with its
reason, instead of scoring it:

- the frozen model cannot consume the family's version of the channel, because its
  `classify` raises; `per_family_incompatible_runs` counts the runs. Those runs used to
  score 0.5. State Divergence's frozen 49-bin gridworld centroids raised on every run of
  the six held-out families that emit `state_counts`, so its flat 0.500 transfer row in
  both committed measurements, `leaderboard/cross_family_transfer.json` (2026-07-09) and
  the `before` configuration of `docs/figures/sign_randomization_impact.json`
  (2026-08-03), was imputed, not measured;
- the pooled fit failed or left the detector untrained, so nothing frozen exists to
  transfer (`pooled_fit_error`).

A held-out family scored on some difficulty tiers but not all is averaged over the scored
ones and says so: each row records `per_family_cells_scored`,
`per_family_cells_not_applicable` and `test_partial_not_applicable_reasons`, and a
multi-trial row the fewest scored and the most left out in any trial (`f0caa3d`).
`continuous_camping` is now listed as excluded from the Reward MLP and State Divergence
pooled fits, but both fits already ignored its 40-episode runs, so the frozen models are
unchanged (bit-identical); only the listing changed. Since the 2026-11-26 declarations the
train AUROC, which `Benchmark.evaluate` computes, leaves its cells out too for Reward MLP,
State Divergence and the ensemble, which need 50 episodes; the 2026-08-03
measurement scored them at the fallback. The Top-5 ensemble's train AUROC is now
cross-validated: `FoldFittedEnsemble` refits its Trajectory MLP member on each training
fold. In both committed measurements that figure was in-sample for the member, which had
been fitted on exactly the runs it was then scored on. The ensemble's transfer score does
not depend on the wrapper.

The script writes `leaderboard/cross_family_transfer_v5.json` by default, at the published
design (10 train seeds, 20 test seeds, 5 trials). It carries a `configs` block naming the
configuration it was measured in, sign randomization on, so `rhob.v3.leaderboard.transfer`,
which names the file `CURRENT_TRANSFER_ARTIFACT`, reads it as the shipped configuration.
It also carries a `published_design` flag, the per-row reasons and counts above, and
`pooled_fit_error`. The script refuses to overwrite
`leaderboard/cross_family_transfer.json`, kept as the 2026-07-09 record
(`HISTORICAL_TRANSFER_ARTIFACT`). Under the CI pins, with torch on one thread, the
published design took 5m23s on a 4-core machine. The documented 1–2 h came
mostly from torch's default thread count on a busy machine, where one trial's Trajectory
MLP train AUROC alone took 489 s; the rollout cache only stops trials and rows from
re-simulating runs. Set `OMP_NUM_THREADS=1` when running it by hand.
`scripts/plot_v5_results.py` plots the v5 file, never the 2026-07-09 one, and labels a
null transfer AUROC "n/a" instead of drawing a bar. The Space renders the v5 file when it
is present, and otherwise the `after` configuration of `sign_randomization_impact.json`,
labelled pre-declaration, with a note that representation mismatches there were scored at
chance and that the run-length rule reached only its train AUROC.

`scripts/measure_sign_randomization.py` had crashed on its first board cell since
2026-09-02, when `CellResult` lost the onset column it read; it runs again. It builds the
ensemble through the transfer script's own code, commits every per-cell AUROC its
docstring promised, refuses to merge boards across commits, package stacks, dirty trees or
duplicated families, compares only cells scored in both modes, checks the pre-audit board
like for like, and computes every headline claim, "reproduces exactly" included, from the
data. A new `ci` subcommand runs the published design end to end in one process. It
simulates each cell once and derives the randomized runs from the un-randomized ones with
the benchmark's own flip, which is bit-identical. The report records that
(`design.sampling.rollouts_shared_across_modes`), and its invariance section says what the
check therefore shows: that the flip reaches only `behav_trace` and that `restrict()` hides
it below L2, not that two separate simulations agree. Run by hand, the stages now
reproduce `ci` and can produce `published_design = true`, which also requires each
companion board at its own seeds and families: `transfer` measures each configuration's
published rows by default, `board --companion LABEL` runs a companion at its published
design, and an unknown `--rows` or `--transfer-rows` name is refused instead of dropped.
`transfer --n-trials` defaults to 3, the published design's value and the 2026-08-03
measurement's (it was 5 until 2026-11-27), while `cross_family_transfer.py` keeps 5.
Whether the default stays at 3 is open, the author's decision.

### Removed (2026-11-27): the v3.2 tables, their two scripts, and `rhob submit` writing the tracked board

The v3.2 tables are deleted: `leaderboard/leaderboard.json`, the standings in
`leaderboard/README.md`, `leaderboard/by_mechanism.md` and `leaderboard/by_difficulty.md`,
with the two scripts that wrote them, `scripts/populate_leaderboard_v32.py` and
`scripts/populate_v3_leaderboard.py`. Nothing in the repository read the tables. They
predate the 2026-08-03 audit and the not-applicable rule, they disagree with both v5
boards, and every detector they ranked except `Ensemble(BehavioralThreshold x2)` is on
`v5_leaderboard.json` and `v5_replicated.json`. The tables were last committed on
2026-07-07 (`b7324ec`), and `populate_v3_leaderboard.py` could not run after 2026-09-02
(`b796f9c`), when `CellResult` lost its onset column. They remain in the history:
`git show b7324ec:leaderboard/leaderboard.json`, and likewise `README.md`,
`by_mechanism.md` and `by_difficulty.md`; the scripts as they last stood are
`git show 4728596:scripts/populate_leaderboard_v32.py` and
`git show 4728596:scripts/populate_v3_leaderboard.py`. `leaderboard/README.md` is now an
index of the current artifacts that quotes no figures.

`python -m rhob submit` now records self-reported, single-draw submissions in
`leaderboard/community/` (`board.json` plus rendered tables) instead of `leaderboard/`, and
takes `--board-dir DIR` for another directory. It accepts the documented four-key minimum,
filling in `author` ("anonymous") and `timestamp`, where it used to crash. It refuses, exit
1 and nothing written, when the `board.json` it finds is not an entries-schema board;
given the v3.2 `leaderboard.json`, a 30-detector `results` board, it used to read it as
empty and save one entry over it, exiting 0. `Leaderboard.load` raises `ValueError` on any
file that is not an entries-schema board instead of returning an empty board.
`LeaderboardEntry.from_results`, and so `rhob evaluate`, counts only measured cells in
`n_cells` and writes `overall_auroc` as null rather than NaN when no cell was measured. If
you relied on the old location: `--board-dir leaderboard` is refused while the index
`README.md` is there, so use another directory.

### Changed (2026-11-27): CI hygiene, and `regenerate.yml` re-draws the boards and re-measures transfer and sign randomization

`regenerate.yml` gains three modes: `redraw`, the board and all 20 replicates at one commit
(the layout-seed entry above); `transfer`, `leaderboard/cross_family_transfer_v5.json` at
the published design, 5 trials, never writing the 2026-07-09 file; and `signrand`,
`measure_sign_randomization.py ci`, which writes
`docs/figures/sign_randomization_impact.json` and `.md`. Every job runs under the
numerics pins (the numerics entry above). Also in `regenerate.yml`:

- Calibrations are derived once per run, by a `calibrate` job, and restored by every
  scoring job. In the `f83a152` re-score each of the 20 draw jobs spent 78 minutes deriving
  them, about 1560 of the run's 2030 runner-minutes.
- The plan job validates every input before a runner is spent. A `from_run` merge must
  name this workflow on this branch, with `src/`, `requirements/`, `pyproject.toml` and the
  scripts that drew the artifacts identical, and exactly the expected artifacts still
  downloadable.
- Modes that write the same files share one concurrency queue; `transfer` and `signrand`
  each have their own. A conflicted rebase in the commit step is aborted instead of
  failing every retry after it.
- Each long compute step (board, redraw board and draws, transfer, sign randomization)
  times out at 330 minutes inside its job's 350, and its upload runs `always()`, so a run
  that hits the limit still saves what it computed: a job that reaches its own timeout is
  cancelled, which skipped the upload, while a step that times out only fails
  (`320b7b7`). Each redraw draw job refuses a draw not made under the pins.
- The calibration tarball's key is defined once, in `.github/actions/calibration-cache`.
  It names the runner image and the numerics pins and starts a new `calib-v2` series.
  Every save carries the run id and attempt, and a scoring job restores exactly the key
  its calibrate job saved, whichever attempt saved it. A run that restored a full tarball
  and still derived calibrations used to lose them, because a save under an existing key
  is skipped.

Elsewhere:

- `tests.yml`: fail-fast off, read-only permissions by default, a timeout on every job,
  only superseded pull-request runs cancelled, and `admission-slow` on `ubuntu-24.04`. A
  scheduled failure opens one tracking issue ("Nightly tests are failing"), closed when a
  scheduled run passes again. `cannotfail.yml` does the same weekly ("Weekly cannotfail
  check is failing") and gains a concurrency group and timeouts; so does
  `cannotfail-release.yml`.
- `drift.yml` installs and tests on `ubuntu-24.04` too. `leaderboard_validate.yml` skips
  deleted submissions and passes refs and paths through the environment.
- `deploy_space.yml` pins `huggingface_hub` 1.33.0 (unpinned, it installed 2.x, outside
  the `space` extra's `<2`), installs the deployed Space's `rhob` at the deploying commit,
  with a warning when it deploys from a ref other than `main`, and takes its inputs
  through the environment.
- `pages.yml` holds `pages: write` and `id-token: write` only in its deploy job, and never
  cancels a deployment in flight.
- `register_guard.yml`'s failure report could not have been filed: with no checkout and
  no `GH_REPO`, `gh` had no repository to file it to. It now sets `GH_REPO`, and finds
  the tracking issue by its exact title.
- Dependabot opens one grouped pull request a week for GitHub Actions. Python
  dependencies still move only through `scripts/update_ci_lock.sh`.
- `pyproject.toml` declares `license = "MIT"` with `license-files` and builds with
  setuptools 77 or later, which silences setuptools' deprecation of the table form
  (unsupported after 2027-02-18).

### Fixed (2026-11-27): `rhob validate` checks a submission's values, and `rhob submit` writes nothing until it can write everything

`python -m rhob validate` checked a submission's keys and its access level only, so a
string AUROC passed, and `rhob submit` then saved it to `board.json` and crashed while
rendering the tables, leaving them stale and making every later submission crash. Both
now apply `rhob.v3.leaderboard.board.submission_problems` (`d7c865c`): `detector_name` a
non-empty string, `access_level` one of L0 to L3, `overall_auroc` a finite number in
[0, 1], or null for a run with no measured cell, `n_cells` a non-negative integer,
`author` and `timestamp` strings, and `family_auroc`, `mechanism_auroc` and `tier_auroc`
maps of names to numbers in [0, 1], the tier names difficulty tiers. `NaN` and `Infinity`
are refused when the file is parsed, because they are not JSON.
`.github/workflows/leaderboard_validate.yml` runs `validate`, so these checks now apply to
submission pull requests. `submit` checks everything and renders `board.json` and its
tables before writing any of them, refuses a `board.json` that holds an invalid entry,
writes with `allow_nan=False`, and refuses, writing nothing, when `README.md`,
`by_mechanism.md` or `by_difficulty.md` in the board directory does not start with the
community header (`# RHOB community submissions (self-reported, single draw)`). Other files
in the directory are left alone. Names and authors are escaped in the rendered tables, so
a `|` cannot shift a column. The submit tests call `main()` in process, keeping one
`python -m rhob submit` smoke test.

### Changed (2026-11-27): tests that re-run the admission gate, and tests that can fail

- `tests/test_v3/test_ledger_reproduces.py` re-certifies four registered families at
  every tier the ledger scores (`reward_channel_tampering`, `proxy_correlation_gaming`,
  `shortcut_exploitation`, `distributional_shift`; 20 cells), with a gate built from the
  ledger's own recorded configuration (12 layouts, 24 seeds per side). Between them they
  reach every status and every outcome value, with measured FAILs on
  `proxy_distribution_matched` and `behavioral_separated`. Statuses, outcomes, the design
  and the wording of each verdict, its numbers and its binding detector masked, must match
  exactly, and every number must match within 5e-3 on rank statistics and 1e-9 relative
  elsewhere. Those tolerances were measured across the CI lock and the unpinned Python
  3.10 and 3.11 jobs: the only drift seen was a few near-tied Reward Skewness scores ranked
  differently under scipy below 1.18, by at most 7.6e-4, with no outcome changed. The
  binding detector is pinned exactly wherever the ledger's choice leads by more than twice
  the rank tolerance; at `reward_channel_tampering`@0.80 Reward KDE and Reward Skewness
  bind 4.4e-4 apart, so either may bind there (`2130ab2`). The test takes about 17 s and
  runs on every push. A slow cell, `rlhf_feature_blindspot_gaming`@0.80, re-certifies
  nightly the one (criterion, outcome) pair the four do not reach, a measured
  `proxy_matched` FAIL.
- `tests/test_cannotfail_parity.py` re-certifies the six DEGENERATE families instead of
  reading only the gate's committed aggregates. It captures every score vector and proxy
  stream the gate measures, and requires `cannotfail` to reproduce each per-layout
  resolution and dispersion bit for bit against the gate re-run on the same machine, the
  per-cell means to match the ledger (within 1e-12 for resolution, and 1e-6 relative or
  1e-15 absolute for informativeness, whose committed values are 0 to 2.4e-16), and each
  statistic to be called degenerate on its own. Two families run on every push, one of
  each kind (`rlhf_reward_model_overopt` by informativeness, `shortcut_exploitation` by
  resolution), so the call made from the scores alone runs on every push; the other four
  run nightly.
- The third copy of the tie rule, `inspect_eval/hvtb_hack_detection/metrics.py`, is held
  to the gate, read with `ast` so `inspect_ai` is not needed: its tolerances must be the
  gate's, and its tie rule must match the gate bit for bit, including on new pairs that
  sit exactly on the tolerance, where no earlier test could tell `<=` from `<`. The tests
  also pin that it drops a sample with a non-finite score rather than counting it as a
  tie, which its docstring still says it does.
- The transfer and sign-randomization script tests run on stub families noisy enough that
  the values they compare are interior, not saturated at 0.0 on every held-out stub, and
  seven mutants of the scripts each fail one of them. The test that stamping provenance
  never imports torch records the import calls; the cache test of a docstring edit to
  `rhob.detectors.posthoc` covers all 19 disk-memoized families and fails if its edit stops
  applying. `tests/test_numerics_pins.py` holds `regenerate.yml`'s pins, the cache key's
  variables and the composite action to one list, and every job of every workflow to a
  timeout.
- `test_the_committed_aggregate_is_what_the_aggregator_writes` compares the JSON trees
  with `math.isclose` (relative 1e-9, absolute 1e-12) and everything else exactly. With
  `==` it turned the Python 3.10 and 3.11 pytest legs red on `827aec9`: CPython 3.10's
  twice-rounded `statistics.stdev` and numpy release drift moved 59 floats in the last bit.
  It skips, giving the reason, where the label-fitted partition is not the pinned one
  (torch absent).

### Fixed (2026-11-26): every detector declares the run length it needs; both boards re-scored

The 2026-11-19 regeneration entry below found that a detector whose `classify` returns a
constant below some run length, and that declares no `min_episodes`, was scored there
anyway, and both boards recorded the constant as a measured AUROC of 0.5. On the
single-draw board that was 145 per-family entries: 104 at L0 (Reward Peak, Reward
Autocorrelation, Reward Trend and Variance Window on the 19 families that run 40 or 60
episodes, and Reward CUSUM, Max Plateau, Gradient Reversal and Reward MLP on the 7 of those
that run 40), 8 at L1 (every L1 detector on the 40-episode `continuous_camping`) and 33 at
L2 (Feature Consistency on the 19, Feature Magnitude and Reward-Feature Correlation on the
7). The 20-draw board had the same 0.5 in each of those cells in every draw.

`f83a152` has each of them declare the length its constant stops at, derived from its own
attributes, and the `classify` guard reads the declaration, so the two cannot drift. At L0,
Reward Peak, Reward Trend, Reward Autocorrelation and Variance Window need 100 episodes,
Gradient Reversal 60, Max Plateau and Reward MLP 50, and Reward CUSUM 51 (at exactly its
50-episode baseline the accumulation loop never runs); at L1, the four two-window detectors
100, Transition Entropy 101 and the three fitted ones 50; at L2, Feature Consistency 100
and Feature Magnitude and Reward-Feature Correlation 50. Reward KDE (22), Reward Variance
Ratio (21), Spectral Reward (24) and Trajectory MLP (2) declare theirs too, below every
registered horizon, so none of their cells moves. A cell below a detector's declaration is
`null`, never imputed. `tests/test_detectors/test_min_episodes.py` replaces
`test_l0_window_fallbacks.py`: for every detector on the board, `classify` is constant one
episode below its declaration and varies at it. Two more of the same kind are fixed
alongside:

- `EnsembleDetector` declared no horizon, so on a run too short for one of its members it
  averaged that member's fallback in, or Reward Skewness's NaN, which failed the cell. It
  now needs every member's horizon, the largest of their minimums (`d745731`). No
  ensemble is on either board.
- Perfect Feature Oracle now declares `required_channels = ("behav_trace",)`. It had been
  checked for the L3 default, `true_rewards`, so a channel with no behavioural trace
  recorded its absent-trace 0.5 as scored. Every board family emits `behav_trace`, so only
  the HVTA channels move.

The re-scores:

- **The single-draw board.** CI run 45747902322 re-scored it at `f83a152` and committed it
  (`03d978e`). It is the 2026-11-19 board with exactly those 145 entries set to `null`:
  every other per-family value, on all 30 detectors, is unchanged.
- **The 20-draw board.** CI run 45747922473 re-scored the 19 detectors whose cells move on
  all 20 draws at `f83a152`. The merge (`scripts/merge_rescored_detector.py`) refused it, as
  it should: of the 23,480 cells those detectors score both in the committed draws and in
  the re-score, 23,066 came back with the same AUROC and 414 did not (the other 9,180 the
  draws had recorded, every one a fallback 0.5, are the cells now N/A). None of the 414 is
  a cell a declaration removes, and none is the fix's doing. 319 are on
  `sequence_keyword_stuffing`, which `e7bb3f3` rewrote on
  2026-09-19, after the draws were made (2026-08-04): the committed draws score the earlier
  version of that family, for every detector. 89 are on three DEGENERATE families whose
  proxy is constant to about 1e-16 (`orbit_chirality` 65, `rlhf_reward_model_overopt` 19,
  `distributional_shift` 5), where a tie breaks differently under the draws' Python 3.11 and
  numpy 2.4.6 than under the CI lock, and 6 are scattered (`eval_probe_sandbagging` 3,
  `pettingzoo_fixed_opponent_exploitation`, `sequence_format_camping` and
  `goal_misgeneralization` 1 each). Taking the re-scored rows would have scored 19 detectors
  on different runs of those families from the other 11 in the same draw, so run
  45776113470 merged them with `--keep-old-values` (`827ae57`): every draw keeps the value
  it was scored with on every cell, and only the re-score's not-applicable set is applied.
  Each replicate's provenance lists its own share of the 414 cells (11 to 38 per draw)
  with both values. The 20-draw board is therefore the committed draws with the too-short
  cells set to `null`, which is what the fix changes and all it changes. Re-drawing the 20
  replicates on today's code, which would bring `sequence_keyword_stuffing` up to date, is
  a separate item (ROADMAP.md). *(2026-11-27: `regenerate.yml`'s new `redraw` mode draws
  the 20 and the board at one commit, and a re-draw is also the only way to vary the
  layout, which none of these draws did; see the 2026-11-27 layout-seed entry above.)*
  `scripts/aggregate_replication.py` then rebuilt `leaderboard/v5_replicated.json` and
  `docs/figures/replication_summary.md` from those draws. The run's aggregator (at
  `6e8521a`) already derived the sentences about the label-fitted climb and about each
  level's denominators from the data (`6df73ea`), and both read correctly. It still
  printed "Max and mean agree on every rung", because `statistic_disagreement` compared
  only whether each statistic separates; `bfac304` added
  `rungs_where_max_and_mean_disagree_on_direction` and re-aggregated both files from the
  committed replicates.
- **The HVTA channels** (`external/hvta/channels_results.json`, `1e38a27`). 132 entries
  that read 0.5 are N/A with their reason, and no other value changed: 126 were a
  detector's constant below its window on the 40- and 42-step cells (nine L0 detectors, and
  three L2 detectors on the IPD channels), and 6 were Perfect Feature Oracle's absent-trace
  0.5 on `logical_bug_wordle`. `external/hvta/CHANNELS.md` counts the two kinds of N/A
  separately.

The admission ledger is not affected: the gate reads none of the detectors whose cells
moved.

**The single-draw ladder**, with the six DEGENERATE families held out of L0 only:

| Access level | Detectors | Mean AUROC | Best AUROC | Best detector | 2026-11-19 mean / best |
|---|---|---|---|---|---|
| L0 (27 families) | 13 | 0.4834 | 0.5474 | Reward CUSUM | 0.4876 / 0.5351 |
| L1 (7 families, 31 cells) | 8 | 0.5789 | 0.9820 | State Divergence | 0.5701 / 0.9270 |
| L2 | 7 | 0.5771 | 0.9430 | Trajectory MLP | 0.5716 / 0.9430 |
| L3 | 1 | 0.9830 | 0.9830 | True Reward Oracle | 0.9830 / 0.9830 |

With the six families back in, the family-weighted L0 mean reads 0.4877. The exclusion is
worth 0.0043, computed from the unrounded means (0.4877139 − 0.4833744 = 0.0043395), and
weighted by cell with no exclusion L0 reads 0.4881. Nine L0 detectors are now scored over
fewer of the 27 families: Reward CUSUM, Max Plateau, Gradient Reversal and Reward MLP over
20, and Reward Peak, Reward Autocorrelation, Reward Trend, Variance Window and Reward
Skewness over 8. Reward CUSUM's 0.5474 is over its 20, and Reward Peak moves from third to
second (0.5111 to 0.5375). The 8 L1 rows are N/A on 92 of 123 cells (26 families).

**The 20-draw board**, before (as of the Reward Skewness re-score of 2026-09-03) and after,
95% intervals over draws *(2026-11-27: every one of these draws was built and oriented at
layout 0, so the intervals are over rollout seeds only; see the 2026-11-27 layout-seed
entry above)*:

| Figure | Before | After |
|---|---|---|
| L0 mean, the negative control | 0.4972 [0.4939, 0.5005] | 0.4951 [0.4899, 0.5000] |
| L0 max | 0.5540 [0.5443, 0.5635] | 0.5713 [0.5598, 0.5825] |
| L1 max (State Divergence in 20 of 20 draws) | 0.9416 [0.9399, 0.9427] | 0.9984 [0.9965, 0.9996] |
| L1 mean | 0.5589 [0.5469, 0.5701] | 0.5665 [0.5530, 0.5792] |
| L2 mean | 0.5740 [0.5685, 0.5791] | 0.5784 [0.5726, 0.5840] |
| L1 − L0, max | +0.3876 [+0.3783, +0.3971] | +0.4272 [+0.4166, +0.4385] |
| L2 − L1, max | +0.0014 [−0.0039, +0.0063] | −0.0555 [−0.0609, −0.0505] |
| L1 − L0, mean | +0.0617 [+0.0510, +0.0717] | +0.0715 [+0.0592, +0.0829] |
| L2 − L1, mean | +0.0150 [+0.0049, +0.0257] | +0.0118 [+0.0008, +0.0235] |
| L3 − L2, mean | +0.4007 [+0.3951, +0.4066] | +0.3963 [+0.3904, +0.4022] |
| Unsupervised L0 max | 0.5514 [0.5410, 0.5619] | 0.5677 [0.5550, 0.5796] |
| Unsupervised L1 max | 0.5340 [0.5147, 0.5543] | 0.5384 [0.5166, 0.5613] |
| Unsupervised L2 max | 0.5872 [0.5757, 0.5999] | 0.5990 [0.5892, 0.6096] |
| Unsupervised L1 − L0, max | −0.0174 [−0.0391, +0.0017] | −0.0293 [−0.0537, −0.0051] |
| Unsupervised L1 − L0, mean | −0.0037 [−0.0210, +0.0126] | −0.0022 [−0.0217, +0.0167] |

L2 max (0.9430 [0.9380, 0.9478]), L3 (0.9746 [0.9712, 0.9778]) and L3 − L2 on maxima
(+0.0316 [+0.0266, +0.0366]) do not move.

**Conclusions that changed:**

- **L2 minus L1 on maxima now separates, with L1 above L2.** It read +0.0014 [−0.0039,
  +0.0063], L2 higher in 13 of 20 draws, and did not separate; it reads −0.0555 [−0.0609,
  −0.0505], L1 higher in all 20. On the single draw the maxima are 0.5474, 0.9820, 0.9430
  and 0.9830, so the ladder of maxima is no longer monotone. The single draw's means also
  put L1 just above L2 (0.5789 against 0.5771, 0.0017 apart from the unrounded means), but
  that order does not replicate: on 20 draws the paired mean puts L2 above L1 (next
  bullet).
- **On the all-detector board no rung's separation depends on the statistic any more; in
  the unsupervised partition one now does.** L2 minus L1 separated on means and not on
  maxima, and `statistic_disagreement` listed it under
  `rungs_where_max_and_mean_disagree_on_separation`; that list is now empty, and on the
  all-detector board max and mean agree that every rung separates. They do not agree on the
  direction of L2 minus L1: L1 is higher on maxima, L2 on means (+0.0118 [+0.0008, +0.0235],
  higher in 13 of 20 draws, with a lower end just above zero), so that rung's direction
  depends on the statistic and both must be quoted. Since `bfac304`,
  `statistic_disagreement` lists it under `rungs_where_max_and_mean_disagree_on_direction`
  instead. `statistic_disagreement` is computed for the all-detector board only. In the
  unsupervised partition, the L1 minus L0 step separated under neither statistic before
  the re-score and now separates under one (next bullet).
- **The unsupervised L1 minus L0 step now separates on maxima, below zero, and not on
  means.** On maxima, −0.0174 [−0.0391, +0.0017], positive in 4 of 20 draws, did not
  exclude zero; −0.0293 [−0.0537, −0.0051], positive in 5 of 20, does: without the five
  label-fitted detectors, L1's best sits below L0's. On means the step still does not
  separate: −0.0037 [−0.0210, +0.0126] before and −0.0022 [−0.0217, +0.0167] after,
  positive in 11 of 20 draws both times. Before, neither statistic separated the step; now
  its status depends on the statistic, and neither answer may be quoted alone. The step
  counting the label-fitted detectors, +0.4272 [+0.4166, +0.4385] on maxima (20 of 20), is
  carried by State Divergence, which is fitted on labels.
- **L1's best is at the L3 oracle on the single draw and above it on 20 draws, over a
  different denominator.** State Divergence reads 0.982 on the single draw against True
  Reward Oracle's 0.983, and 0.998 [0.997, 1.000] on 20 draws against 0.975 [0.971,
  0.978]. It is scored on 31 cells in 7
  families (the 8 that emit `state_counts`, less `continuous_camping`), the oracle on 123
  cells in 33, and it is fitted on labels by cross-validation. That is not a like-for-like
  comparison, and it does not show that L1 access reaches the oracle.
- **The L0 negative control's interval holds 0.5 only at its upper end.** On 20 draws the
  L0 mean is 0.4951 [0.4899, 0.5000], against 0.4972 [0.4939, 0.5005]; to five decimals
  the upper end is 0.50001, so 0.5 is only just inside. On the single draw the L0 mean falls
  from 0.4876 to 0.4834.
- **Nine L0 detectors, not only Reward Skewness, are scored over fewer families** (above),
  so an L0 figure per detector has to carry its family count.

Not re-run: `leaderboard/cross_family_transfer.json` and
`docs/figures/sign_randomization_impact.json` were produced before the declarations, and
the v3 tables `scripts/populate_v3_leaderboard.py` writes to `leaderboard/` date from
2026-07-07. ROADMAP.md lists them. *(2026-11-27: the v3 tables and that script are
retired; see the 2026-11-27 Removed entry above. `cross_family_transfer.json` is kept as
the 2026-07-09 record and is not re-run; its successor,
`leaderboard/cross_family_transfer_v5.json`, and `sign_randomization_impact.json` are what
`regenerate.yml`'s new `transfer` and `signrand` modes write; see the 2026-11-27 transfer
entry above.)*

### Fixed (2026-11-25): the site deploys from two jobs

The Pages deploy failed on the push that merged #3 and on a fresh dispatch:
`deploy-pages@v5` listed the run's artifacts a second after `upload-pages-artifact@v5`
uploaded one in the same job and found none. A re-run listed the first attempt's
artifact, so an artifact from a finished job is visible. `.github/workflows/pages.yml`
now uploads in a `build` job and deploys in a `deploy` job that needs it. The site kept
serving the previous deployment in the meantime.

### Fixed (2026-11-25): mpe2 capped below 1.2

The unpinned `pytest` matrix met mpe2 1.2.0 and went red on `main` (nightly of
2026-11-25) and on pull request #3: `ImportError: cannot import name 'simple_spread_v3'
from 'mpe2'`. 1.2.0 replaces `simple_spread_v3` and `simple_tag_v3` with `_v4` versions,
and three PettingZoo families import the v3 names. The weekly drift run of 2026-11-23
installed the newest releases and passed, so 1.2.0 came out after it. `pyproject.toml`
now requires `mpe2>=1.0,<1.2`. The families were calibrated and the admission ledger was
certified under mpe2 1.1.1, the version the CI lock pins, so the lock and the nightly
physics screen are unchanged. Moving to 1.2 means porting those families to the v4
environments and re-measuring them.

### Changed (2026-11-19): the single-draw board and the full admission ledger, regenerated

CI run 44447220658 ran the workflow in the entry below and committed the single-draw board
(`52656cb`) and the admission ledger (`d4b700d`), under Python 3.12.16 on the CI lock. The
board was scored from code at `acd35c2` (its `provenance.git.commit`) and the ledger from
code at `a2aa435` (its own); the two commits differ only in
`.github/workflows/regenerate.yml`. They replace the board of 2026-07-17 and the ledger of
2026-08-03.

**The board** (`leaderboard/v5_leaderboard.json`) now carries provenance and sampling
blocks and writes a cell as `null` when the channel its detector reads is missing or the
horizon is below the detector's declared `min_episodes`: the 8 L1 rows are N/A on 88 of
123 cells (25 families) and Reward Skewness on 59 (19 families). Eight other L0 rows
cannot score the short-horizon families either, but declare no horizon, so the board
records their fallback 0.5 there (below); so do three L2 rows, and all eight L1 rows on
the 40-episode `continuous_camping`. *(2026-11-26: they all declare `min_episodes` now
(`f83a152`). On the re-scored board the 8 L1 rows are N/A on 92 of 123 cells (26
families); Reward Peak, Reward Autocorrelation, Reward Trend, Variance Window and Feature
Consistency on 59; Reward CUSUM, Max Plateau, Gradient Reversal, Reward MLP, Feature
Magnitude and Reward-Feature Correlation on 22. See the 2026-11-26 entry above.)* It is scored under
the shipped configuration, with the behavioural sign randomized per family. The ladder,
with the six DEGENERATE families held out of L0 only:

| Access level | Detectors | Mean AUROC | Best AUROC | Best detector | 2026-07-17 mean / best |
|---|---|---|---|---|---|
| L0 (27 families) | 13 | 0.4876 | 0.5351 | Reward CUSUM | 0.4866 / 0.5311 |
| L1 | 8 | 0.5701 | 0.9270 | State Divergence | 0.5200 / 0.6210 |
| L2 | 7 | 0.5716 | 0.9430 | Trajectory MLP | 0.7213 / 0.9750 |
| L3 | 1 | 0.9830 | 0.9830 | True Reward Oracle | 0.9830 / 0.9830 |

With the six families back in, the family-weighted L0 mean reads 0.4898. The exclusion is
worth 0.0023, computed from the unrounded means (0.4898105 − 0.4875513 = 0.0022592); the two
rounded figures differ by 0.0022. Weighted by cell and with no exclusion, L0 reads 0.4903.
*(2026-11-26: on the re-scored board the ladder reads L0 0.4834 / 0.5474 (Reward CUSUM,
over 20 of the 27 families), L1 0.5789 / 0.9820 (State Divergence, over 7 families and 31
cells) and L2 0.5771 / 0.9430, with L3 unchanged. With the six families back in, L0 reads
0.4877, an exclusion of 0.0043 (0.4877139 − 0.4833744 = 0.0043395), and 0.4881 by cell
with no exclusion. See the 2026-11-26 entry above.)*

- Behavioral Threshold falls from 0.975 to 0.512 and Centroid Tracker from 0.966 to 0.361,
  the two L2 detectors that read the feature's sign
  ([docs/figures/sign_randomization_impact.md](docs/figures/sign_randomization_impact.md)).
  L2's mean falls from 0.7213 to 0.5716, and those two account for all of it and slightly
  more: they lower the 7-detector mean by 0.153 ((−0.463 − 0.605) / 7), and the other five
  together raise it by 0.003. Its best is now Trajectory MLP, which is fitted inside each
  cell (0.931 to 0.943). Perfect Feature Oracle, Behavioral Threshold's relabelled
  duplicate, moves with it (0.975 to 0.512, identical on 33 of 33 families) and stays out
  of every aggregate. *(2026-11-26 re-score: L2's mean is 0.5771. The two still lower it
  by 0.153, and the other five now raise it by 0.008: with their short-horizon fallbacks
  N/A, Feature Magnitude goes from 0.564 to 0.577, Feature Consistency from 0.516 to 0.530
  and Reward-Feature Correlation from 0.558 to 0.570.)*
- State Divergence goes from 0.621 to 0.927 only because its 25 N/A families are no longer
  averaged in as 0.5. On the 8 families that emit `state_counts` every L1 row is unchanged,
  so the board now measures every per-detector L1 value the audit had recomputed (the
  table under "61% of L1" below), to three decimals. The suite mean reads 0.5701 here and
  0.5700 in that table: the board stores each detector's overall to three decimals, and
  their mean is 4.561 / 8 = 0.570125, while the same means recomputed per cell from the
  per-family values, before that rounding, average 0.5700.
  L1's best, 0.927, is now close to L2's, 0.943: on maxima the ladder is 0.5351, 0.9270,
  0.9430, 0.9830, and the climb the L3 entry below places between L1 and L2 now comes
  between L0 and L1, over the 8 families L1 can score. *(2026-11-26 re-score:
  `continuous_camping`, below every L1 window, is N/A too, so the L1 rows are over 7
  families and 31 cells and no longer equal the audit's table: State Divergence reads
  0.982 and the suite mean 0.5789 (4.631 / 8). L1's best is now above L2's 0.943 and
  within 0.001 of L3's 0.983, so on maxima the ladder is 0.5474, 0.9820, 0.9430, 0.9830,
  not monotone, and L1 is still over far fewer families than L2 or L3.)*
- Reward Skewness is N/A on its 19 short-horizon families instead of 0.5, as on the 20-draw
  board since `9da6b21`; its overall goes from 0.508 to 0.516.
- Reward Skewness is the only L0 detector that declares its horizon. Eight others return a
  constant below their windows, and the board records it as a measured 0.5: Reward Peak,
  Reward Autocorrelation, Reward Trend and Variance Window on the 19 families that run 40
  or 60 episodes (59 cells each), and Reward CUSUM, Max Plateau, Reward MLP and Gradient
  Reversal on the 7 of those that run 40 (22 cells each). Those 104 per-family values are
  in the L0 mean of 0.4876 above, 7 of them in Reward CUSUM's 0.5351, and they are the
  fallback, not measurements. The 20-draw board has the same 0.5 in each of those cells in
  every draw. Three L2 detectors (Feature Consistency on the 19, Feature Magnitude and
  Reward-Feature Correlation on the 7) and all eight L1 detectors on `continuous_camping`
  do the same, 41 more fallback values, in the L1 and L2 figures. Declaring the horizons
  and re-scoring both boards is open (ROADMAP.md). *(2026-11-26: done. Every one of these
  detectors declares `min_episodes` (`f83a152`), and both boards were re-scored on CI with
  the 104 L0 and 41 L1 and L2 values N/A: the single draw in run 45747902322 (`03d978e`),
  the 20 draws in runs 45747922473 and 45776113470 (`827ae57`). Without them the single-draw L0 mean is 0.4834 and
  Reward CUSUM 0.5474, over 20 families; on 20 draws the L0 control mean is 0.4951
  [0.4899, 0.5000], against 0.4972 [0.4939, 0.5005]. See the 2026-11-26 entry above.)*

**The ledger** (`admission/admission_ledger.json`, `admission/ADMISSION_LEDGER.md`) now
covers every registered family, merged from 123 single-cell shards certified on 7 CPU models.
**All 33 families: of 123 cells, 42 ADMITTED, 30 DEGENERATE, 51 NOT ADMITTED**, against
15, 30 and 5 over the 10 families and 50 cells of 2026-08-03.

- The 10 families already in it keep every verdict and every criterion outcome, in all 50
  of their cells. In the four of them that are not DEGENERATE their metrics differ by at
  most 0.0013 (`eval_probe_sandbagging` @0.98, Reward Skewness CI upper bound 0.5529 to
  0.5517). In the DEGENERATE cells of `orbit_chirality` and `rlhf_reward_model_overopt`,
  whose proxies are constant to within rounding error, so that their L0 statistics only
  order that error, they move by up to 0.1155 (`rlhf_reward_model_overopt` @0.98 and @0.8:
  mean L0 AUROC 0.4583 to 0.5000, CI lower bound 0.3845 to 0.5000). The DEGENERATE
  families are the same six (`distributional_shift`, `monitored_sandbagging`,
  `orbit_chirality`, `physics_exploitation`, `rlhf_reward_model_overopt`,
  `shortcut_exploitation`): the 23 families added bring no seventh.
- The 23 families added bring 73 cells, 27 ADMITTED and 46 NOT ADMITTED. All 13 MuJoCo
  cells are NOT ADMITTED, as are 14 of the 15 PettingZoo cells, 9 of the 15 in the five
  RLHF families added and 8 of the 15 sequence cells.
- `proxy_matched` is measured on 93 of the 123 cells and fails 24, in 12 families; on the
  2026-08-03 ledger it failed none of the 20 it measured. In 8 of the 24 the point estimate
  is outside [0.40, 0.60]: `mujoco_camping` at 0.8 and 0.7, `mujoco_joint_limit_gaming` at
  0.9, `pettingzoo_free_rider_exploitation` at 0.8 and 0.7, `rlhf_sparse_coverage_gaming`
  at 0.9 and `sequence_format_camping` at 0.9 and 0.8. In the other 16 the point estimate is
  inside and the 90% interval is not, which the certificate prints as underpowered: those
  are FAILs because the equivalence test could not establish matching within the margin at
  this design, not because the estimate is outside it. `proxy_distribution_matched` fails
  40 cells, 35 of them new; Reward KDE is the binding detector on 34 of those 35 (the other
  is `continuous_camping` @0.9, on Reward Variance Ratio) and on 37 of the 40 (Reward
  Skewness binds on `goal_misgeneralization` @0.6 and `reward_channel_tampering` @0.98).
  `behavioral_separated` fails one, `proxy_correlation_gaming` @0.6 (mean L2 AUROC 0.527
  against a floor of 0.6).
- `onset_localizable` passes in all 123 cells. Its `onset_std` is 0.0 in 118, the cells of
  the other 31 families, and nonzero in the 5 of `continuous_camping` and
  `gridworld_camping` (0.23 to 2.05 against a threshold of 4.0, and 9.87 against 50). So
  "exactly 0.0 in all 50 ledger cells", the earlier wording, stays true of the 50-cell
  ledger and does not extend to the full one: `cannotfail resolution` calls the statistic
  fixed over those 118 cells (one distinct value) and `ok` over all 123 (six).
- RHOB's self-coding changes with it. `docs/SURVEY_PROTOCOL.md` codes a criterion
  degenerate only when its statistic took one value across the whole shipped design, and
  the ledger now covers the whole shipped design, all 123 cells. The `rhob_post_audit` row for
  `onset_localizable` in `audit/rhob_self_coding.json` therefore records
  `observed_distinct_values` 6, measured over all 123 cells, where its first coding
  (`e9a1ace`) recorded 1 over the 50-cell ledger and was degenerate. By the rule in
  `rhob.audit.diagnostics` the row is no longer degenerate: it codes `not_applicable`, and
  the post-audit coding has no finding left (`audit/SELF_CODING.md`, regenerated with
  `scripts/audit_table.py`; `tests/test_audit` pins the new result). The ledger's coverage
  changed, not the criterion: the statistic is still 0.0 in the 118 cells of the 31 families whose
  onset label is the configured schedule, so those passes are fixed by construction.
  Wherever RHOB's self-audit result is stated from the first coding, in the paper or
  elsewhere, it has to be updated or scoped to the 50-cell ledger; ROADMAP.md lists this
  for the author.

**For the L0 negative control**, stated plainly: the ledger certifies a matched proxy on
the 42 of 123 cells it admits. A 43rd cell, `proxy_correlation_gaming` @0.6, passes both
proxy criteria and fails only `behavioral_separated`, so it is NOT ADMITTED. 51 cells are
measured NOT ADMITTED, and 50 of them fail a proxy criterion: 24 fail `proxy_matched`, 40
`proxy_distribution_matched` and 14 both. The 30 cells of the six DEGENERATE families are
not established either way. The board's L0 aggregate holds out only those six families,
as before, so the NOT ADMITTED cells are still in the L0 mean. Whether they should be is
open, for the author to decide (ROADMAP.md).

### Added (2026-11-19): the board and the full ledger can be regenerated on CI

The single-draw board (`leaderboard/v5_leaderboard.json`) predates the N/A schema, so its
L1 rows and its Reward Skewness row still carry 0.5 where nothing was measured. The
admission ledger covers 10 of the 33 families (50 of 123 cells). Both take hours on one
machine, and an attempt here on 2026-11-19 was lost to a container restart. *(Both were
regenerated by this workflow the same day; see the entry above.)*

`.github/workflows/regenerate.yml` (manual dispatch) runs both on the CI lock and commits
each artifact to the branch it ran on, once the sources at the tip are still the ones
that produced it. The board is one job, the script as a person runs it, because its 30
detectors share one set of rollouts. The ledger is one job per cell:
`scripts/admission_ledger.py` gains `--list-cells` and `--merge-shards DIR`, which
refuses anything but the full grid from one clean commit under one Python, one set of
package versions, one gate and one design. On one machine a merged ledger equals an
unsharded one, checked on two families (REPRODUCIBILITY.md, "The Admission Ledger");
across CPU models a cell can differ in its last digits. The ledger's provenance now
records the CPU model, and a merged ledger lists every model it was certified on. `tests/test_v3/test_regen_merge.py` covers the merge's refusals and its
command line. The workflow commits the artifacts alone; the docs and tests that quote
them are updated in a reviewed follow-up, and fail on the branch tip until then.

### Changed (2026-11-19): workflows run their actions on Node 24

Every run logged "Node.js 20 is deprecated", with `actions/checkout@v4` and
`actions/setup-python@v5` forced onto Node 24. The workflows now use checkout v7,
setup-python v7, cache restore/save v6, upload-artifact v7, download-artifact v8,
configure-pages v6, deploy-pages v5 and upload-pages-artifact v5, each of which declares
`node24`. setup-uv moves from v5 to v10.3.3, pinned by commit: upstream has published no
major or minor tags since v8. None of the inputs these workflows pass changed meaning
across those majors; checkout v7's new refusal applies only to `pull_request_target` and
`workflow_run`, which no workflow here uses.

### Added (2026-11-16): an analysis plan for axis A's power re-run, before any row exists

The paper's surviving prevalence claim is sub-class A: an equivalence asserted from a
non-significant difference test, in 15 of 77 preconditions across 4 of 15 benchmarks. The
plan is to re-run it under D2: at the design each claim shipped, how often would its
difference test have caught a true AUROC of 0.7? `docs/SURVEY_AXIS_A_PREREG.md` fixes the
following before any row is filled:

- the fields coded per claim, and fixed rules for a source that is silent or conflicts,
  each saying which side it favours where it sets a value that favours one;
- the effect-size grid (0.7 primary, 0.6 and 0.8 secondary, power threshold 0.8);
- the computation (`cannotfail`'s `difference_power` and `min_two_sided_p`), with
  `power.py`, the frame, the script and the template pinned by file digest;
- the per-claim rules, under which "cannot reject at any data" needs an exact test and a
  stated level and sidedness;
- six analyses, all of which a strengthening outcome must survive;
- what each outcome does to the sub-class A wording, including the two outcomes that weaken
  it: fewer than 15 confirmed on re-coding, and "most adequately powered", more of the
  computed claims adequately powered than not (A > W), however few were computed;
- how replies from the groups are reported: beside the registered outcome, never in its
  place.

It states what is already known. The paper's §5 found 15 of 20 recoverable designs
underpowered, and does not say by what criterion. The author has seen that result, and how
many of the 15 sub-class A claims are among those 20 cannot be told from committed
material. So the author commits to it first, in a private sealed record that also holds
the paper's criterion and its verdict on each claim. Only the record's SHA-256 goes in the
plan, and the record is published with the results. The plan also finds that the paper's
App. G screen does not establish its own bar: its `CANNOT_FAIL` verdict on its own 5-vs-5
cell does not hold, and the paper needs correcting there.

`scripts/survey_axis_a_power.py` applies the rules to a sheet in the format of
`audit/axis_a_sheet_TEMPLATE.csv`, and reports an outcome only on the registered inputs.
The document is written, not yet registered: it counts as registered once a deposit of it
exists whose date a third party sets, an OSF registration or a Zenodo deposit. A git tag is
not one; its date is set by whoever makes it. No row exists. The 15 await the
author's §5 sheet and a second human coder, and they live outside the repository or under
the gitignored `audit/private/`. Nothing computed from them, aggregate counts included, is
published before every group's outreach window has closed; before then only salted digests
are committed.

### Changed (2026-11-16): the nightly physics screen installs a hashed lock

The nightly `admission-slow` job installed `pip install -e ".[dev,continuous,mujoco,pettingzoo]"`
unpinned, so it met each release the night it shipped. That is how MuJoCo 4.0.0 reached it
on 2026-11-07, on an unchanged commit. It now installs
`requirements/ci-admission-linux-py312.txt`, then `pip install --no-deps -e .`. The lock
holds 81 distributions for CPython 3.12 on Linux x86_64, each at one version with the sha256
of every file PyPI holds for it, torch and its CUDA wheels included. It is resolved as PyPI
stood at 2026-11-15T05:00:20Z, so it is exactly the stack the 2026-11-15 verification run
(43679157068) installed and passed every slow screen on. A new release now reaches the
nightly only through a regeneration of the lock, which someone reviews; the CPython patch
release, pip and the runner image still float. `scripts/check_ci_lock.py` fails, naming the
requirement, if the lock no longer satisfies something `pyproject.toml` declares for those
extras, which `pip check` does not read. It runs in the fast suite on every pull request
(`tests/test_ci_lock.py`, names and version bounds), so a dependency change that skips the
regeneration fails there rather than on main's next nightly, and again in the nightly's
install step, where it also follows the extras a requirement asks for (`gymnasium[mujoco]`).

- `scripts/update_ci_lock.sh` regenerates the lock with `uv pip compile --exclude-newer` and
  writes the cutoff into the header, so rerunning with that cutoff reproduces the file.
  `--check` tests this.
- `.github/workflows/drift.yml` runs weekly. It installs the newest releases `pyproject.toml`
  allows, tabulates every version that differs from the lock, and runs the fast suite and
  the four MuJoCo families' slow screens with every calibration derived fresh. A scheduled
  failure shows on its own page and in one tracking issue, kept by a job that holds only
  `issues: write`. It does not run the PettingZoo, RLHF reward-model or sequence screens.
- The per-push `pytest` matrix stays unpinned on purpose. It is the daily check that a new
  release breaks the fast suite.
- REPRODUCIBILITY.md no longer calls `requirements-lock.txt` "the exact verified stack". It
  lists the direct dependencies of one Windows, Python 3.13 environment on 2026-07-29, has
  no hashes, and no CI job installs it. Neither file is in any tag.

The first CI install of the lock was `workflow_dispatch` run 43836925199 on the branch, on
2026-11-16: the hash-checked install passed, and the slow tier gave 73 passed and 21
xfailed, as 43679157068 did. Started cold on a slow runner, it took 153 of the job's 180
minutes, so `admission-slow`'s timeout is now 240.
Details: [requirements/README.md](requirements/README.md).

### Added (2026-11-16): `cannotfail` reads the files maintainers already have

`cannotfail resolution` now reads nested fields by dotted path in JSON, JSONL and Parquet
(`--score output.probs.1`, where an integer indexes a list). It takes `--group` more than
once (`--group task --group model`, never comma-separated, since metric names such as
`acc,none` contain commas). It reads Parquet through the `cannotfail[parquet]` extra, and
Inspect logs through `cannotfail[inspect]` with `--format inspect`, via inspect_ai's own
`samples_df`. The package README has a tested recipe for an lm-evaluation-harness
`--log_samples` file, a leaderboard predictions JSON, a Parquet file and an Inspect log. The
first two files are synthetic; the Inspect log was written offline by inspect_ai with its
mock model. The base install is still NumPy only: the package's CI asserts that SciPy,
pyarrow, pandas and inspect_ai are all absent from it, then reruns the tests with each
extra. All of this lands in `cannotfail` 0.1.0, which is still unreleased; see
[tools/cannotfail/CHANGELOG.md](tools/cannotfail/CHANGELOG.md).

### Changed (2026-11-16): provenance records the simulator versions

`rhob.v3.provenance` recorded numpy, scipy, scikit-learn, torch and rhob, but not the
simulators that step the rollouts. Of the committed artifacts only the admission ledger says
which MuJoCo produced it, because its script added the simulators itself; the 20-draw
replication does not. *(2026-11-19: the regenerated `v5_leaderboard.json`, `52656cb`,
records them too.)* Every provenance block now also records mujoco, mujoco-core,
gymnasium, pettingzoo and mpe2, as null when not installed. A test fails if the calibration
cache keys on a version that provenance does not record. Artifacts written under the old
list are not rewritten and still read: no reader requires the new names, and both report
renderers reproduce their committed package lines
(`tests/test_v3/test_na_channels_and_provenance.py`).

### Fixed (2026-11-15): the calibration cache could not see a dependency upgrade

From 2026-11-07 the nightly `admission-slow` job failed every night, on an unchanged
commit. MuJoCo 4.0.0 was published on 2026-11-06. CI installed unpinned, and the cache
that stores each family's calibration was keyed on the defining module's source alone,
so `mujoco_joint_limit_gaming` ran under 4.0 with the hip amplitude calibrated under 3.17.
Its 0.9 tier stopped admitting (L0 AUROC 0.708 [0.600, 0.817]), and a strict xfail at
0.7 started passing.

Recalibrating does not rescue it. Run with the cache off, the same three tiers under
4.0.0 fail at 0.9 the other way (0.333 [0.226, 0.440]) and pass at 0.7. Under 3.17.0
the identical command admits 0.9 and fails 0.8 and 0.7 as documented. So:

- `mujoco<4` in the `mujoco` extra. Moving to 4.x is a re-measurement of the MuJoCo
  families and the ledger, not a version bump.
- Every cache key now also hashes the installed versions of the numerical stack (numpy,
  scipy, mujoco, mujoco-core, gymnasium, torch, pettingzoo, mpe2), the Python minor
  version and the machine. It also hashes the source of every `rhob` module the family
  imports, followed transitively. The key used to miss edits to
  `rhob.environments.calibration`, where `calibrate_scale` lives. *(2026-11-27: it now
  hashes each module's code, docstrings and comments left out, rather than its source,
  and the six numerics variables join the versions; see the 2026-11-27 numerics entry
  above.)*
- The CI tarball key includes the same version token and all of `src/rhob`, with a
  prefix fallback that the complete per-entry keys make safe.

The cache did not cause the failure, since a fresh calibration fails too. What it did
was hide that each run mixed two simulators, and it would have hidden an upgrade that
moved the numbers without failing a test. That is the shape of the checks the paper
describes, here in infrastructure rather than in a benchmark's construction. It does not
add to the paper's count of instances.

Post-mortem, with the runs, the controlled experiment and what is still open: [docs/postmortems/2026-11-mujoco-calibration-cache.md](docs/postmortems/2026-11-mujoco-calibration-cache.md).

### Added: HVTB listed in the Inspect Evals Register

Unrelated to the audit above, and recorded here because it changes what this repository
publishes rather than what it measures. The HVTB hack-rate eval on the `inspect-eval` branch
(also `inspect_eval/` on main, since `b083e55`) is listed in the [Inspect Evals
Register](https://ukgovernmentbeis.github.io/inspect_evals/evals/hvtb/), pinned at
`a2c73417a79be65b802b2fbe62dc6fac03e9c3de`. What is listed is HVTB's hack rate, not RHOB's
detector suite, and the listing publishes no results. The register lists for discoverability
and does not review methodology; its structural lint "describe[s] structure and conventions,
not whether the evaluation measures what it claims."

Added `.github/workflows/register_guard.yml`, which checks daily that the pinned commit is
still fetchable and that the register's published result still describes this entry, and
weekly that the pinned tree still installs and passes its tests. Nothing upstream installs or
executes the code, so no upstream check would catch a regression.

### Corrected (2026-09-03): Reward Skewness padded the L0 negative control

Reward Skewness needs `baseline_window + test_window` = 100 episodes, and 19 of the 33
families run 40 or 60. Below that horizon it returned the constant 0.5 for every run on
both arms, and `leaderboard/v5_replicated.json` published that as "at chance": the defect
the L1 entry below records, from a different cause, in the row the paper calls its
negative control. `584ba13` (2026-09-02) gave `PosthocDetector` a `min_episodes` horizon,
which Reward Skewness declares as 100. *(2026-11-19: it is still the only detector that
declares one; eight other L0 detectors return a constant below their windows, which both
boards record as 0.5. See the 2026-11-19 regeneration entry above.)* *(2026-11-26: every
detector whose score is constant below some run length now declares it (`f83a152`), and
both boards were re-scored with those cells not applicable; see the 2026-11-26 entry above.)* A
shorter cell is recorded as not applicable,
excluded and never imputed, and the detector returns NaN below its horizon, so a path
that skips the guard fails loudly. `9da6b21` (2026-09-03) re-scored the row on the same 20
draws:

- Before: 123 cells on 33 families, **24 of the 33 at exactly 0.500** (SD 0.000).
- After: **0.504 [0.494, 0.514] over 64 cells on 14 families**; the other 19 families are
  N/A. Five of the 14 are still at exactly 0.500 (`distributional_shift`,
  `monitored_sandbagging`, `orbit_chirality`, `rlhf_reward_model_overopt`,
  `shortcut_exploitation`), from a near-constant proxy rather than the horizon. That is
  disclosed, not fixed.

`scripts/merge_rescored_detector.py` refuses a re-scored draw unless every cell the old
record scored comes back with the same AUROC, or is admitted by name. 1,279 of 1,280 came
back the same; the one admitted by name (replicate 7, `eval_probe_sandbagging` @0.8, 0.60
→ 0.58) is written into that replicate's provenance with both values.

The ladder moves with the row: the unsupervised L0 rung is now 0.551 [0.541, 0.562], the
unsupervised L1−L0 step −0.017 [−0.039, +0.002] (positive in 4 of 20 draws), and the step
counting the label-fitted detectors +0.388 [+0.378, +0.397] (positive in 20 of 20),
replacing the pre-re-score 0.544, −0.010 and +0.394. No ledger verdict changes, because
the ten ledger families run 100 to 200 episodes. *(2026-11-26: with the other detectors'
short horizons N/A too, the unsupervised L0 rung is 0.568 [0.555, 0.580] and the
unsupervised L1−L0 step on maxima −0.029 [−0.054, −0.005] (positive in 5 of 20 draws),
which now separates, with L0 above L1. On means that step is −0.002 [−0.022, +0.017]
(positive in 11 of 20) and still does not separate, so its status now depends on the
statistic. The step counting the label-fitted detectors is +0.427 [+0.417, +0.439]
(positive in 20 of 20). See the 2026-11-26 entry above.)*

The nightly smoke screen had seen the same tie: 33 of its 36 DEGENERATE cells were this
(`tests/test_v3/test_gate_honest_verdicts.py`). Re-measured with the horizon declared, 19
short-horizon cells admit and 14 fail on a real late-window density leak, now marked
`xfail(strict=True)` per tier with the measured AUROC in the reason.

### Corrected (2026-09-02): `true_reward_diverges` reported a tie as a measured FAIL

It had a run-count guard and an effect-size guard but no tie guard, so on arms tied by
arithmetic it reported FAIL, "measured and found not to diverge", when nothing had been
measured; the HVTA two-player port shipped exactly that construction. `584ba13` adds a
resolution guard mirroring `_score_resolution`: tied arms now return DEGENERATE, a gap
below the effect floor FAIL, and a gap above it PASS.

### Removed (2026-09-02): the internal onset column

Every one of the 50 admission-ledger cells has `onset_std` exactly 0.0, because those
onsets are authored, so a constant prediction equal to the label scored a perfect 0.000
and the column could not fail. `584ba13` removed it from the leaderboard: `CellResult` no
longer carries it, and it is still computed only for `external/hvta`, where the label was
not written by us. The same fact is why `rhob.audit` codes `onset_localizable` as
degenerate (`e9a1ace`). *(2026-11-19: of the full ledger's 123 cells, 118 have 0.0 and 5
do not, so over the shipped design the statistic takes six values and the self-coding no
longer codes `onset_localizable` degenerate; see the 2026-11-19 regeneration entry above.)*

### Fixed (2026-09-19): `sequence_keyword_stuffing` never emitted its keyword

Its calibration lever resolved to exactly 0.0 at every scored tier, so the branch that
emits the keyword was dead code and the family's documented exploit never happened.
`86acb92` points the lever at the secondary keyword, and the family's new smoke screen
admits all three scored tiers
(`results/family_diagnostics/sequence_keyword_stuffing_gate.json`). *(2026-11-19: the full
ledger admits it at 0.8 and 0.7 and fails it at 0.9 on `proxy_matched`, 0.4077 with 90% CI
[0.3677, 0.4477], underpowered.)*

### Changed (2026-09-19): `pettingzoo_fixed_opponent_exploitation` is marked failing per tier

The nightly fails it at all three scored tiers on Reward KDE: AUROC 0.234 / 0.849 / 0.833
at 0.9 / 0.8 / 0.7, against a 0.5 ± 0.256 smoke band, at resolution 1.000, so these are
measurements. The mean proxy is matched and the per-episode SD is not, and the family's
variance constant cannot be retuned to match both (`results/family_diagnostics/`).
`a7b9cba` marks each tier `xfail(strict=True)` rather than patching it; closing the leak
needs a construction change.

### Corrected: the L3 "oracle ceiling" was an L2 detector counted twice

`PerfectFeatureOracleDetector` is a subclass of the L2 `BehavioralThresholdDetector`
that overrides only `access_level` and `name` — `classify` and `detect_onset` are
inherited verbatim. It reads exactly one signal, `RunData.behav_trace`, which
`rhob.v3.access.restrict` already exposes at L2, and never reads `true_rewards`,
the only channel L3 adds. In the committed leaderboard it agrees with Behavioral
Threshold on **33 of 33 families** and on the overall figure (both 0.9750). Zero
families differ. It is not an oracle and never was. *(2026-11-19: on the board
regenerated that day, 33 of 33 again, both 0.512.)*

- Before (duplicate counted): **L3, n=2, mean 0.9790**, max 0.9830
- After (duplicate excluded): **L3, n=1, mean 0.9830**, max 0.9830

The "n=2 / mean 0.9790" figure never measured anything — it was the average of one
oracle and one copy of an L2 baseline.

**Consequence for the headline claim.** README previously presented a monotone
access ladder (L0 0.51 → L1 0.53 → L2 0.76 → L3 0.99) as the central empirical
result. That is a ladder of *means*, and its L3 rung was inflated by the duplicate.
Best-L2 vs best-L3 is **0.9750 vs 0.9830 — a gap of 0.0080 AUROC**. On maxima the
ladder is 0.531 → 0.621 → 0.975 → 0.983: essentially all of the climb happens
between L1 and L2, and **nothing happens between L2 and L3**. *(That is the 2026-07-17
board. On the board regenerated on 2026-11-19 the maxima are 0.535 → 0.927 → 0.943 →
0.983; see the 2026-11-19 regeneration entry above.)* *(2026-11-26 re-score: 0.547 →
0.982 → 0.943 → 0.983. L1's best, over 7 families, is now above L2's and within 0.001 of
L3's; see the 2026-11-26 entry above.)*

Nothing was deleted. Perfect Feature Oracle keeps its leaderboard row and its
reported "L3" label (artifacts, paper tables and the Space key off them), is
registered in the new `rhob.detectors.redundancy` as a duplicate of Behavioral
Threshold, and is held out of **every** access-level aggregate. It is deliberately
not re-filed under L2 either, which would move the double-count there and lift L2's
mean from 0.7213 to 0.7530 for no new information. **A duplicate belongs to no
level's aggregate**, and is described from here on as a labelled cross-check, never
as an independent detector or an oracle.

30 detectors ship in the leaderboard; **29 are independent measurements**.

Corrected per-access-level table — L1–L3 over 33 families and 123 cells per detector,
**L0 over 27 families** with the ledger's six DEGENERATE families held out (see the
negative-control entry below) — now generated to `docs/figures/v5_access_summary.md`
and printed by `scripts/plot_v5_results.py` *(2026-11-19: run on the regenerated board,
they give the ladder in the 2026-11-19 regeneration entry above)* *(2026-11-26: run on the
re-scored board, they give L0 0.4834 / 0.5474, L1 0.5789 / 0.9820, L2 0.5771 / 0.9430 and
L3 0.9830 / 0.9830, mean / best; see the 2026-11-26 entry above)*:

| Access level | Detectors | Mean AUROC | SD across detectors | Best AUROC | Best detector |
|---|---|---|---|---|---|
| L0 | 13 | 0.4866 | 0.0250 | 0.5311 | Reward CUSUM |
| L1 | 8 | 0.5200 | 0.0387 | 0.6210 | State Divergence |
| L2 | 7 | 0.7213 | 0.2052 | 0.9750 | Behavioral Threshold |
| L3 | 1 | 0.9830 | 0.0000 | 0.9830 | True Reward Oracle |

The **SD column is spread across the detectors at a level, not a confidence
interval and not measurement uncertainty** — every detector contributes one number
from a single shared evaluation draw, so they are not independent replicates. This
caveat is now printed inside the generated table and legended on the figure. (These
are population SDs, `ddof=0`, matching the previous `np.std`-based figure; sample
SDs would read L0 0.0260, L1 0.0413, L2 0.2217.) Per-level **max** is now reported
alongside the mean, because a level's mean is dragged down by however many weak
detectors happen to have been written for it and therefore measures the suite's
composition as much as the benchmark's structure.

### Corrected: 61% of L1 was a hardcoded constant, and excluding it *raises* L1

**25 of 33 registered families ship `state_counts=None`.** Every L1 detector
returns a hardcoded 0.5 fallback on them — a constant, not a measurement. The 8
families that do emit the channel (`gridworld_camping`, `continuous_camping`,
`distributional_shift`, `goal_misgeneralization`, `physics_exploitation`,
`reward_channel_tampering`, `rlhf_reward_model_overopt`,
`sensor_calibration_tampering`) carry 35 of the 123 leaderboard cells, so
**88/123 = 71.5%** of every L1 detector's published cells were the fallback.

Whole categories are affected wholesale, by design, and each shared rollout module
documents why: all 4 `mujoco_*` (high-dimensional continuous state), all 5
`pettingzoo_*` (multi-agent joint state space), all 5 `sequence_*` (the vocabulary
*is* the state space), and the 5 `rlhf_*` families on the shared RLHF rollout, plus
`eval_probe_sandbagging`, `monitored_sandbagging`, `novelty_farming`,
`orbit_chirality`, `proxy_correlation_gaming`, `shortcut_exploitation`.

| L1 detector | Published (123 cells) | Measured only (8 families, 35 cells) |
|---|---|---|
| State Divergence | 0.621 | **0.927** |
| State Coverage Rate | 0.514 | 0.549 |
| Bimodal Occupancy | 0.512 | 0.542 |
| State Frequency Anomaly | 0.510 | 0.534 |
| Transition Entropy | 0.508 | 0.529 |
| Visitation Entropy Trend | 0.499 | 0.496 |
| Occupancy Polarization | 0.499 | 0.495 |
| Centroid Drift | 0.497 | 0.489 |
| **Suite mean** | 0.5200 | **0.5700** |
| **Suite max** | 0.6210 | **0.9270** |

**The direction matters.** Excluding the fabrication *raises* L1: the 0.5 constants
were dragging a genuinely strong detector down to 0.621 and the suite mean to
0.520. README previously reported "L1 … 0.53 ± 0.08 (marginal)" as an empirical
finding; it was a constant diluted with 13 real measurements. The correct statement
is: **on the 8 families that actually emit state-visitation data the best L1
detector reaches 0.927, while the other 7 remain near chance (0.489–0.549)**. Every
L1 figure must now be quoted with its denominator (8 families / 35 cells), because
it is **not** comparable to the 33-family denominators used at L0 and L2. *(2026-11-26: on
the re-scored boards `continuous_camping` is N/A at L1 too, so the statement is over 7
families and 31 cells: on the single draw the best L1 detector reaches 0.982 and the other
7 sit at 0.488–0.555 (on 20 draws 0.998 and 0.458–0.576), and every L1 figure carries that
denominator. See the 2026-11-26 entry above.)*

A `0.5` in a regenerated artifact now means "measured, and at chance" — a real
result. `continuous_camping` is the clean example: it *does* emit `state_counts`,
and all 8 L1 detectors still score exactly 0.500 on it. Under the old schema that
was indistinguishable from the 25 families where nothing was measured at all.
*(2026-11-19: it is not a measurement. `continuous_camping` runs 40 episodes, below
every L1 detector's window, so each returns its fallback 0.5 there; see the 2026-11-19
regeneration entry above.)* *(2026-11-26: every L1 detector now declares its window, so
`continuous_camping` is N/A at L1 on both boards, and no 0.5 on the re-scored boards comes
from a detector run below its window; see the 2026-11-26 entry above.)*

`behav_trace` is present in every family measured, and `proxy_rewards` /
`true_rewards` always are, so no L2 or L3 cell was fabricated. This is an L1-only
problem today. *(2026-11-19: it was not. Reward Skewness, at L0, padded 19 families below
its horizon; see the 2026-09-03 entry above.)* *(2026-11-26: nor was it L0 and L1 only.
Feature Consistency, Feature Magnitude and Reward-Feature Correlation wrote their fallback
0.5 on 33 short-horizon per-family entries, and eight other L0 detectors on 104; all are
N/A now. On the HVTA channels, Perfect Feature Oracle recorded its absent-trace 0.5 on
`logical_bug_wordle` and now declares `behav_trace`. See the 2026-11-26 entry above.)*

*(Note for anyone cross-referencing the audit: its "20 of 33" figure is stale by
exactly the 5 sequence-generation families added afterwards; 25 − 5 = 20.)*

### Changed: `proxy_matched` is now an equivalence test, and the docs describing it were wrong

Until this change the criterion was literally `abs(mean_auroc - 0.5) < 0.10` — a
**difference test used to assert equivalence**, i.e. "I failed to detect a leak"
reported as "there is no leak". Separately, both the `admission_gate.py` module
docstring and REPRODUCIBILITY.md described it as a bootstrap "95% CI containing
0.5", which **was never what the code did**. Both descriptions are corrected.

With the then-defaults (12 layouts × 4 seeds/side, which is what every call site in
the repo evaluated to) the standard error of the mean is exactly 0.0625 — the
per-layout null is Mann-Whitney, `sqrt((n+m+1)/(12nm)) = 0.2165` for `n=m=4`,
divided by `sqrt(12)`; a 40k-replication Monte Carlo confirms it. Measured pass
rates of the old test:

| True mean L0 AUROC | Old pass rate |
|---|---|
| 0.500 (genuinely matched) | 89.6% — i.e. ~10% false-reject |
| 0.556 | 76.2% |
| 0.611 (**leaking**) | **43.5% — certified anyway** |
| 0.714 | 2.5% |

The gate now runs **TOST** (two one-sided tests) at α=0.05 per side on the mean L0
AUROC across 12 independent layouts, with a cluster bootstrap over layouts (2000
resamples; the layout is the independent unit) supplying the SE and an interval of
`mean ± mult·SE`, `mult = t(0.95, L-1)·sqrt(L/(L-1)) = 1.876` at L=12. The family
is certified only if the **whole interval** lies inside [0.40, 0.60] — equivalently
a two-sided 90% CI, the standard bioequivalence convention. Noise now makes
certification *harder*, never easier.

**The margin stayed at 0.10**, deliberately: RHOB's published scientific claim is
unchanged; what changed is that the gate demands *evidence for* equivalence instead
of *absence of evidence against* it. It also sits just below the audit's worst
measured miss (a family leaking at 0.611, `|0.611 − 0.5| = 0.111`), so anything at
or above 0.60 is non-equivalent by construction.

**Sample size went up 6×, and it is derived rather than chosen.** TOST passes when
`|mean − 0.5| + mult·SE < margin`, so at a true mean of 0.5 the design needs
`SE ≤ margin/(mult + z)`. With four equivalence tests that must all pass, each is
sized at `0.90^(1/4) = 0.974` to hold the *combined* false-reject rate on a
genuinely matched family at 10%; that gives `SE ≤ 0.0244`, and the exact
Mann-Whitney null SD first clears it at **24 seeds per side per layout**. (A single
equivalence test would have needed 18.)

|  | Pre-audit | Now |
|---|---|---|
| Seeds/side/layout | 4 | **24** |
| Seeds per variant | 48 | **288** |
| Rollouts per (family, difficulty) | 96 | **576** |
| TOST half-width | 0.1172 — **wider than the entire ±0.10 margin** | 0.0458 |

The pre-audit design **could not have certified equivalence at any observed mean**,
not even exactly 0.5, on even one detector. That is the quantitative statement of
why a difference test was standing in for one.

### Added: `proxy_distribution_matched`, a sixth criterion

`proxy_matched` certifies **one scalar of one detector** — `RewardThresholdDetector`
scores `proxy_rewards[-100:].mean()`, and an AUROC over a mean is blind to variance
and shape. The new criterion applies the identical TOST independently to each of
three shipped shape-sensitive L0 detectors (**Reward Variance Ratio, Reward KDE,
Reward Skewness**) as an intersection-union test: all three intervals must lie
inside the margin. An IUT needs no alpha correction (it is automatically
level-alpha); the cost is power, paid in seeds.

The gap was demonstrated on shipped code, not hypothesized:

- **`mujoco_sensor_decoupling` @ 0.9**: the gate's L0 AUROC is 0.465 — comfortably
  "matched" — while **Reward KDE scores 0.979 on the identical rollouts**. *(2026-11-19:
  on the full ledger this cell fails `proxy_matched`, 0.4502 with 90% CI [0.3798, 0.5206],
  underpowered, and fails `proxy_distribution_matched` with Reward KDE 0.9939, CI
  [0.9881, 0.9998].)*
- **Synthetic probe**, proxy `N(0.5, 0.02)` vs `N(0.5, 0.60)`: `proxy_matched`
  passes; `proxy_distribution_matched` fails with Reward KDE at 0.0000 (perfect
  inverted separation). Notably **Reward Variance Ratio does not fire** (0.5437)
  despite this being a variance mismatch — it scores late-vs-early variance
  *within* a run and both variants are stationary. That is precisely why the
  criterion is a panel rather than one hand-picked statistic.
- `sequence_keyword_stuffing.py:45-69` already documented this exact failure mode
  in prose ("a per-episode proxy-variance mismatch, NOT a mean gap") before the
  gate could test for it.

Criterion order is now a single source of truth, `rhob.v3.admission_gate.CRITERIA`:
`proxy_matched`, `proxy_distribution_matched`, `behavioral_separated`,
`true_reward_diverges`, `onset_localizable`, `camping_quality`.

### Added: `admission/` ledger — the negative control is now falsifiable

`python scripts/admission_ledger.py` produces `admission/admission_ledger.json` and
`admission/ADMISSION_LEDGER.md`: every (family, difficulty) with each criterion's
pass/fail **and its numbers, failures included**, plus provenance and the gate's
fixed root seed 12345. The `results` block reproduces byte-for-byte on the same
commit; only the timestamp and timings vary.

This exists because RHOB's L0-at-chance result is a **negative control**, and a
control whose failure nobody can see is not a check.

**Current run, across 10 of 33 families: of 50 cells, 15 ADMITTED, 30 DEGENERATE, 5 NOT ADMITTED.** *(2026-11-19: that was the run at the time; across all 33 families, of 123 cells, 42 ADMITTED, 30 DEGENERATE, 51 NOT ADMITTED; see the 2026-11-19 regeneration entry above.)*
(This entry first reported "30 of 35 admitted" over 7 families; 15 of those
admissions were withdrawn by the degeneracy guard below, on the same rollouts, and the
scope then grew to 10 families to cover every family that guard had flagged. See
that entry for why the number falling is the benchmark getting more honest rather
than worse.)

- `reward_channel_tampering` fails `proxy_distribution_matched` at 4 of its 5
  tiers — 0.98 (binding: Reward Skewness 0.5613, CI [0.5180, 0.6045]), 0.90 (Reward
  KDE 0.2056, CI [0.1662, 0.2450]), 0.70 (Reward KDE 0.7972, CI [0.7652, 0.8292]),
  0.60 (Reward KDE 0.7915, CI [0.7498, 0.8333]). Only 0.80 is admitted. Its
  run-level mean is genuinely matched throughout (`proxy_matched` passes at every
  tier, e.g. 0.4813 CI [0.4383, 0.5244] at 0.98) — exactly the failure mode above,
  on shipped code: **matched in the mean, separable in shape**. This also breaks
  `tests/test_v3/test_family_reward_channel_tampering.py`.
- `goal_misgeneralization` @ 0.60 fails `proxy_distribution_matched` (binding:
  Reward Skewness 0.5573, CI [0.4994, 0.6152]) — marginal; the interval just
  crosses 0.60. Its other four tiers are admitted, and its test file certifies only
  `default_difficulties()[0] = 0.98`, so that test still passes.
- Admitted at every tier: `eval_probe_sandbagging`, `sensor_calibration_tampering`
  (5/5 each). Six families — `distributional_shift`, `monitored_sandbagging`,
  `orbit_chirality`, `physics_exploitation`, `rlhf_reward_model_overopt` and
  `shortcut_exploitation` — are admitted at **no** tier, not because they failed
  anything, but because both proxy criteria came back DEGENERATE on all 30 of their
  cells. Three fail the informativeness floor (relative SD 0, 1.4e-16, 2.8e-16) and
  three fail on resolution with an informative proxy that is identical run to run
  (`monitored_sandbagging` 1.00, `shortcut_exploitation` 3.0–4.4, and
  `physics_exploitation` at 3.3e-4, which is *above* the 1e-4 floor and still resolves
  only 0.35% of cross-variant pairs).
- **`proxy_matched` was measured on 20 of the 50 cells and failed none of them; on
  the other 30 it was not measurable.** Every measured failure so far is the new
  shape criterion — `proxy_distribution_matched`, which is not established on 35 of
  the 50 cells (30 degenerate plus its 5 failures). Stated plainly: the L0-at-chance
  negative control survives on the mean statistic *over the cells where the mean
  statistic exists*, does not survive on the distribution for one family, and is not
  established either way on the 30 degenerate cells. *(2026-11-19: that was the 10-family
  ledger at the time. On the full ledger `proxy_matched` is measured on 93 of the 123 cells
  and fails 24 of them, `proxy_distribution_matched` fails 40 and `behavioral_separated` 1,
  so of the 123 cells 42 are admitted, 51 measured NOT ADMITTED and the same 30 DEGENERATE,
  not established either way; `proxy_distribution_matched` is now not established on 70 of
  the 123 cells (30 degenerate plus its 40 failures). See the 2026-11-19 regeneration
  entry above.)*

The certificate now distinguishes a leak from a power failure, printing
`point estimate is inside the band but the CI is not (underpowered)`. Not
hypothetical: at an interim 18-seed run `sensor_calibration_tampering` @0.98 failed
`proxy_matched` with CI [0.3988, 0.5081] (point estimate 0.4534, inside the band).
At the correct 24-seed design it passes.

### Added: the degeneracy guard — the fix reproduced the defect it was fixing

The TOST above removed a check that could not fail. Then the first ledger it produced
contained 15 cells where the check still could not fail, for a different reason, and
nothing in the gate could tell the difference.

**The mechanism.** Both proxy criteria TOST an AUROC, and an AUROC is a rank
statistic. When every score it ranks is tied, `roc_auc_score` returns exactly 0.5 —
the half-credit-per-tie convention, not a finding of indistinguishability. The cluster
bootstrap over such layouts has **SE 0**, the interval collapses to the point
`[0.5000, 0.5000]`, and the TOST then certifies against *any* margin: 0.10, 0.001,
anything anyone might ask for. "I could not look" was being published as "I looked,
and it was fine" — the same category of error as the pre-audit "I failed to detect a
leak" published as "there is no leak", recurring one level down, **inside the fix for
it**.

**It was not hypothetical.** 15 of the 35 cells in the first equivalence-gated ledger
certified `proxy_matched` on that zero-width interval: every difficulty tier of
`distributional_shift`, `monitored_sandbagging` and `orbit_chirality`, each of which
emits a **constant** proxy reward. In `distributional_shift`'s case that constant
(0.675) was itself the documented remediation for a real proxy leak, and
REPRODUCIBILITY.md described it as "Proxy is now truly matched" — a sentence corrected
in this release. The proxy was not matched; it was empty. Removing the signal removes
the leak and establishes nothing about matching.

**The guard asks two questions before reading any equivalence test's value.**

1. *Could the statistic have taken a different value?* For a layout with `n` hacking
   and `m` legit runs, the statistic's *resolution* is the fraction of the `n·m`
   cross-variant score pairs the detector strictly orders (ties, and pairs touching a
   non-finite score, count against it). A tied pair contributes exactly 0.5 to the
   AUROC whatever the family does, so the layout's AUROC is confined to
   `0.5 ± resolution/2`; when that entire attainable range already lies inside the
   equivalence band, the test was decided before the first rollout.
2. *Did the signal it read mean anything?* Resolution alone is bypassable and cheaply
   so: a proxy of `0.675 + N(0, 1e-7)` measures resolution 0.961 on the L0 mean
   statistic and 1.000 on all three shape detectors, and certifies **ADMITTED** at the
   shipped design — on jitter eight orders of magnitude below the reward that no
   consumer of a reward could act on. Tightening the tie tolerance cannot fix it,
   because jitter ten times larger than any tolerance clears it. So the second test is
   an effect size on the family rather than a tie test on the scores:
   `proxy_informativeness` is the pooled proxy stream's `SD / mean(|proxy|)`
   (scale-free), and `PROXY_INFORMATIVENESS_FLOOR = 1e-4` rejects dust. It is a
   property of the family, not of a detector, so one measurement disqualifies every
   equivalence test on the cell at once.

The floor sits in the one real gap in the measured distribution. Three families are
constant to within a single ULP — `distributional_shift` 0.0, `orbit_chirality`
1.4e-16, `rlhf_reward_model_overopt` 2.8e-16 — and the next value up is
`physics_exploitation` at 3.3e-4, a factor of ~1.2e12 higher. `1e-4` lands between
them. Above 3.3e-4 the distribution is a continuum up to ~4.3 with no comparable gap.
*(2026-11-19: the full ledger reads 0.0, 1.1e-16 and 1.6e-16 to 2.4e-16 for the three, and
none of the 23 families it added falls in the gap: the lowest of them is
`sequence_length_padding` at 6.6e-3.)*

An earlier draft of this entry called that distribution "bimodal with nothing between
1.6e-9 and 5.1e-2" and counted six families as near-zero. **Both halves were wrong**,
and the second contradicted `src/rhob/v3/admission_gate.py` in the same tree:
`physics_exploitation` (3.3e-4), `sequence_length_padding` (6.5e-3) and
`goal_misgeneralization` (9.9e-3–1.3e-2) all lie inside the interval claimed to be
empty, and `monitored_sandbagging` (1.00) and `shortcut_exploitation` (3.0–4.4) are
among the *most* informative families on the board. Six families are DEGENERATE, over
**30 of the benchmark's 123 cells**, but only three of them are caught here; the other
three are caught by the resolution guard, which is why the gate asks both questions.

Only the two proxy criteria carry the guard, because only they are equivalence tests —
the other four are difference tests, where ties drive the statistic toward the null and
therefore toward failing, which is the safe direction.

**Changed: a criterion has three outcomes, not two.** New
`rhob.v3.admission_gate.CriterionOutcome`: `PASS` (measured, and met), `FAIL`
(measured, and not met), `DEGENERATE` (**not measurable** — the statistic has no
resolution on this family). The ledger counts the three separately, lists degenerate
cells in their own section with the resolution that produced the verdict, and a cell
is ADMITTED only if every criterion is `PASS`.

**Consequence for the headline count.** The guard withdrew 15 admissions on the same
rollouts — the old headline was "30 of 35 admitted" — and the ledger's scope was then
widened from 7 families to 10 so that every family the guard had flagged is actually
in it. The grid today:
**50 cells, 15 ADMITTED, 30 DEGENERATE, 5 NOT ADMITTED.** *(2026-11-19: all 33 families, 123 cells, 42 ADMITTED, 30 DEGENERATE, 51 NOT ADMITTED.)*
No family regressed and no threshold was tightened. Those 15 admissions were withdrawn
because they had never been earned: they rested on a criterion that was structurally
incapable of returning anything else. A benchmark whose certified count falls when its
measurement is repaired is behaving correctly; the previous number was the one that
was wrong.

**Consequence for the negative control.** Degenerate cells are excluded from RHOB's
L0-at-chance negative control. "Reward-only detectors sit at chance on a carefully
matched proxy" and "reward-only detectors sit at chance on a constant" are different
claims, and only the first is RHOB's. `summarize_access_levels` takes the ledger's
`degenerate_families` and withholds them from **L0 and only L0** — degeneracy is a
property of the proxy reward, the one channel L0 reads, and the other rungs read
channels for which a constant-proxy family is an ordinary benchmark item. On the
committed board the L0 suite mean is **0.4866 with all six held out (27 families)
against 0.4895 with them in** (both family-weighted; the previously published 0.4898 is
the cell-weighted, un-excluded form — of the 0.0032 total difference the exclusion is
worth 0.0029 and the re-weighting 0.0003). Both
figures are reported together, so the exclusion is auditable rather than silent. The
affected families remain valid hacking families — they separate at L2 and their true
rewards diverge — and rejoin the control group when their proxies are made
informative-but-matched rather than constant. *(2026-11-19: on the regenerated board the
two family-weighted figures are 0.4876 held out and 0.4898 in; that 0.4898 is
family-weighted and is not the cell-weighted 0.4898 above, which now reads 0.4903. From
the unrounded means the exclusion is worth 0.0023 and the re-weighting 0.0005. The
0.0032, 0.0029 and 0.0003 above are differences of the rounded figures; from that board's
unrounded means they are 0.0033, 0.0029 and 0.0004. NOT ADMITTED cells are not held out:
they are still in the L0 mean, and whether they should be is an open question for the
author; see ROADMAP.md.)* *(2026-11-26 re-score: 0.4834 held out and 0.4877 in, and 0.4881
by cell with no exclusion; from the unrounded means the exclusion is worth 0.0043 and the
re-weighting 0.0004. See the 2026-11-26 entry above.)*

### Added: the smoke screen and certification are now labelled as different things

`tests/test_v3/test_family_*.py` run the admission gate at 12 layouts × 4 seeds/side
(96 rollouts/cell), which supports an equivalence margin of **±0.256** — not the
±0.10 the benchmark publishes. That was the pre-audit design, and it was reported as
"proxy matched" without qualification. The rollouts have not changed; the label has.

- `tests/test_v3/admission_helpers.py` rewritten. It previously offered
  `assert_admitted(family, difficulty=None, n_seeds_per_variant=30)`, whose docstring
  read "certify `family` and assert it passes all 5 admission criteria" — one
  difficulty, 4 seeds/side/layout, and the word *certify*. It now offers
  `SMOKE_MARGIN`, `smoke_gate()` and `assert_smoke_admissible()`, which runs the
  screen at **every** tier `default_difficulties()` returns and says in its own
  docstring that it does not certify. `SMOKE_MARGIN` is
  `admission_gate.required_seeds_per_layout` inverted — the unique margin at which 4
  seeds/layout is exactly sufficient at the gate's own α and target power — rather
  than a tolerance somebody chose.
- New `tests/test_v3/test_admission_smoke_design.py` pins that inversion and asserts
  `SMOKE_MARGIN > EQUIVALENCE_MARGIN`, so the screen can never drift into
  impersonating the published claim.
- Certification is `scripts/admission_ledger.py` at 12 × 24 (576 rollouts/cell),
  offline; it cannot run in CI (`mujoco_camping` alone spends ~78 s calibrating its
  proxy per difficulty before the first rollout). **Only the ledger issues ADMITTED.**
  A family can be green in CI and NOT ADMITTED or DEGENERATE in the ledger; the ledger
  is the authority.
- The split is documented where contributors read it: README ("Adding a New Family"),
  `docs/TUTORIAL_ENVIRONMENT.md`, `docs/API_SPECIFICATION.md` and REPRODUCIBILITY.md.
  It previously existed only in code and one helper docstring.

The screen is a real screen: every proxy-matching defect this repo has actually
shipped is caught at 96 rollouts (`rlhf_sparse_coverage_gaming` @0.95 at 0.1075, the
`goal_misgeneralization` speed-factor bug at ~0.73, `mujoco_sensor_decoupling` @0.9
under Reward KDE at 0.979, the synthetic F2 case at 1.000). What it cannot see is a
small leak, roughly 0.60–0.75, which is the band the ledger's margin exists to police.
*(2026-11-19: the full ledger finds such leaks on both sides of chance in families whose
screen passes `proxy_matched`: `mujoco_camping` at 0.8 and 0.7 (mean L0 AUROC 0.3529 and
0.3484) and `mujoco_joint_limit_gaming` at 0.9 (0.6623).)*

### Added: `AdmissionGate.certify_all_tiers`

`certify()` with `difficulty=None` covers only `default_difficulties()[0]`, which
is how the pre-audit repo certified families — and it was not sound.
`tests/test_v3/test_family_rlhf_sparse_coverage_gaming.py` certified at difficulty
0.95, which `default_difficulties()` never returns, so the three difficulties the
benchmark actually scored were certified by nothing. Measured at the time: the gate
passed at 0.95 with mean L0 AUROC 0.4531, while the shipped pair at 0.95 measured
0.1075 and the leaderboard recorded 0.04.

### Added: provenance and sampling blocks on every artifact

New `src/rhob/v3/provenance.py`. `provenance_block()` records the git commit,
branch, dirty flag (true if `git status --porcelain` reports anything, **including
untracked files**) and sorted dirty-file list, Python version and implementation,
platform, tracked package versions (`null` when not installed), `argv`, and a
caller-supplied `script`. `sampling_block()` records the draw itself.

The sampling block exists specifically so that **a single unreplicated draw is
machine-readable as such**. The leaderboard evaluates `n_layouts: 1`,
`layout_seeds: [0]`, 5 seeds per variant, `n_replicates: 1` — because
`FamilyRegistry.generate_suite` calls `generate_pair(d)` with
`BaseFamily.generate_pair`'s default `seed=0`. *(2026-11-27: it now calls
`generate_pair_at(d, seed=layout_seed)`. The board passes the default layout 0, so it is
unchanged; until then a replicate at layout `k` was built at layout 0 too. See the
2026-11-27 layout-seed entry above.)* Near AUROC 0.5 the standard error of
a single cell is **≈ 0.19** (corrected 2026-08; this entry originally said 0.16, which
was wrong — the Mann–Whitney null is `sqrt((n+m+1)/(12nm))` = 0.191 at n=m=5), and
every detector is scored on the identical rollouts
so their differences are not independent measurements. The block says so in a
`note` field rather than leaving it to be reconstructed from script source.

`cross_family_transfer.json`'s sampling block additionally carries
`model_init_trials` (the weight-init replication axis, default 5 — **distinct
from** `n_replicates`, the environment draw, which is 1), `test_seed_base` (50000),
`n_seeds_test` and the test rollout seed lists.

**Not yet applied to the committed leaderboard.** `leaderboard/v5_leaderboard.json`
and `leaderboard/cross_family_transfer.json` predate the module and acquire these
blocks on the next regeneration (~2–3 h). `admission/admission_ledger.json` carries
them now. The corrected L1 figures above are therefore a *recomputation from* the
committed artifact, not a re-measurement of it. *(2026-11-19: `v5_leaderboard.json` now
carries them, regenerated in `52656cb`, and the re-measurement gives every per-detector
value in the L1 table above to three decimals. `cross_family_transfer.json` has not been
regenerated and still lacks them.)* *(2026-11-26: the re-scored board makes
`continuous_camping` N/A at L1, so its L1 values are over 7 families and 31 cells and no
longer equal that table; State Divergence reads 0.982.)* *(2026-11-27:
`cross_family_transfer.json` will not acquire them: it is kept as the 2026-07-09 record,
and `scripts/cross_family_transfer.py` now writes `leaderboard/cross_family_transfer_v5.json`,
with both blocks, and refuses to overwrite it.)*

### Changed: N/A cells are `null`, never imputed

New per-detector keys in `v5_leaderboard.json`: `cells`, `cells_measured`,
`cells_not_applicable`, `not_applicable_families`, `not_applicable_reasons`;
`per_family` is float **or `null`**, and `overall_auroc` is `null` when nothing was
measurable. `NaN` is never written (a bare `NaN` is invalid JSON and fails strict
parsers, including the Gradio Space). New top-level `families_evaluated`;
`timestamp` is retained for backward compatibility with
`rhob.v3.leaderboard.adapters` and `space/app.py`, mirroring
`provenance.generated_utc`. `cross_family_transfer.json` gains the corresponding
`train_*`/`test_*` N/A keys. *(2026-11-19: a short horizon is `null` only for a detector
that declares `min_episodes`. Only Reward Skewness does; eight L0 detectors, three L2
detectors and every L1 detector still write their fallback 0.5 on a family shorter than
their windows; see the 2026-11-19 regeneration entry above.)* *(2026-11-26: every one of
them now declares `min_episodes` (`f83a152`), and those cells are `null` on both re-scored
boards and in `external/hvta/channels_results.json`; see the 2026-11-26 entry above.)*

### Changed: documentation reframed

- **The L0-at-chance result is presented as a negative control / manipulation
  check, not a finding.** It was previously written up as a discovery that
  "reward-only monitoring fails". It is not: L0 sits at chance **by construction**,
  the gate certifies that the construction held, and the useful direction is the
  failure one — L0 *above* chance means the family leaked and every other number on
  it is suspect. Every site presenting it as a discovery has been rewritten to
  point at the ledger instead.
- **README now leads with the method** (matched-pair construction + admission gates
  as a reusable way to build evaluation environments with certified, falsifiable
  properties) rather than with the RHOB Transfer Score. The construction/gate
  material was previously buried below the leaderboard.
- **"Transfer excellently to unseen families" is removed.** All 14 transfer train
  and test families were written by the same authors under the same construction
  recipe with the same `behav_trace` convention, so RTS measures transfer *within a
  recipe*. The reported ± on neural detectors is across 5 weight initializations
  only; the environment draw is unreplicated.
- **New `docs/THREAT_MODEL.md`**, linked from the README: what RHOB is, what it is
  not, and its external-validity limits. States explicitly that **hacking policies
  are scripted, not emergent** (in 28 of 33 families; the 5 RLHF-RM families are a
  partial exception, running real policy-gradient optimization against a genuinely
  fitted reward model, so the vulnerability is designed but the exploitation is
  not), and what that does and does not license concluding. *(Corrected to 26 of 33 in
  `9e2f068`; see `docs/THREAT_MODEL.md` §3.)*
- **`docs/site/index.html`**: replaced a **retracted 0.95 L2 held-out transfer
  number** and a wrong 0.87 "L3 ceiling", and refreshed the stale v1.4 / 14-family
  / 35-detector badges, the 3-family transfer split, and the transfer cards
  (0.498/0.500/0.950/1.000 → 0.478/0.500/0.931/0.994).
- **REPRODUCIBILITY.md**: corrected the `proxy_matched` description (F5), the
  30×9 scope, the `--n-seeds-train 15` command that does not match the committed
  `n_seeds_train: 10`, the "all 9 families pass the gate" claim, and the stale
  "28/30 detectors scoring successfully" note (all 30 now score all 123 cells). *(Not on
  the board regenerated on 2026-11-19, which writes N/A cells as `null`, the
  short-horizon cells of detectors that declare no horizon excepted (see the 2026-11-19
  regeneration entry above): each of the 8 L1
  detectors scores 35 cells and Reward Skewness 64.)* *(2026-11-26 re-score: no
  short-horizon exception remains. Each L1 detector scores 31 cells; Reward Skewness,
  Reward Peak, Reward Autocorrelation, Reward Trend, Variance Window and Feature
  Consistency 64; Reward CUSUM, Max Plateau, Gradient Reversal, Reward MLP, Feature
  Magnitude and Reward-Feature Correlation 101.)*
- **`docs/API_SPECIFICATION.md`**: documented the N/A cell contract, the artifact
  schema, the duplicate-detector aggregation rule, and corrected "L3 … never scored
  in production" (L3 is scored and is in the leaderboard).
- **`docs/TUTORIAL_ENVIRONMENT.md`**: 6 criteria not 5, `certify_all_tiers`, the
  576-rollouts-per-cell cost, how to tell an underpowered failure from a leak, and
  guidance on emitting `state_counts`. Now also states that the family-test pattern
  it tells contributors to copy is the **smoke screen**, not certification, and how
  to run certification.
- **The tri-state outcome is introduced wherever a count is quoted.** README,
  REPRODUCIBILITY.md, `docs/THREAT_MODEL.md`, `docs/site/index.html` and this file
  previously reported admitted-vs-failed only, which silently filed "not measurable"
  under "matched". Every headline count now reads admitted / degenerate / not
  admitted, and every claim of the form "the L0-at-chance control survives" now
  carries the denominator it survives over.
- **README's "New families must pass the admission gate … use
  `gate.certify_all_tiers(family)`" claimed an enforcement that does not exist.**
  `certify_all_tiers` is called by `scripts/admission_ledger.py` and by nothing in
  `tests/`. The section now describes what actually runs, at which strength, and how
  much of the benchmark it covers.

### Known limitations that remain unfixed

Stated because the point of this entry is not to claim the benchmark is now sound:

- **The ledger covers 10 of 33 families.** The other 23 are **uncertified**, which
  is not the same as certified-and-passing. **73 of 123 leaderboard cells are
  certified by nothing.** At 576 rollouts/cell, closing this is a substantial
  compute commitment; it is the largest outstanding item. *(Resolved 2026-11-19 in
  `d4b700d`: the ledger covers all 33 families; see the 2026-11-19 regeneration entry above.)*
- **Six families cannot be certified at all as currently built.**
  `distributional_shift`, `orbit_chirality` and `rlhf_reward_model_overopt` emit
  proxies constant to within numerical dust; `monitored_sandbagging`,
  `shortcut_exploitation` and `physics_exploitation` emit proxies that vary but come
  out identical in every run. All 30 of their cells are DEGENERATE. This one is not a
  compute problem: re-running the same rollouts returns the same non-answer. It needs a
  proxy that varies across runs, is readable, and still matches — a family redesign.
  The 23 families outside the ledger's scope have not been swept at the certification
  design, so six is a lower bound. *(2026-11-19: the full ledger sweeps them and finds the
  same six, no more.)*
- **Neither admission check is enforced repository-wide.** 21 of 33 families have a
  smoke test; **12 have none**, so nothing admission-related runs for them in CI, and
  **11 never reach `AdmissionGate` in any test**. *(22, 11 and 10 since `86acb92` gave
  `sequence_keyword_stuffing` a smoke screen.)* No test calls `certify_all_tiers` on
  a registered family — the two that exercise it use synthetic fixtures, so the ledger
  script is its only caller.
- **The leaderboard is still one unreplicated 5-vs-5 draw** with no confidence
  intervals, and **the distribution the gate certifies (12 layout seeds) is still
  not the distribution the leaderboard scores (layout seed 0)**. *(2026-11-27: the
  20-draw board drawn on 2026-08-04 did not close the second half either: every one of its
  draws was built at layout 0; see the 2026-11-27 layout-seed entry above.)*
- **`reward_channel_tampering` and `goal_misgeneralization` @0.60 genuinely fail**
  the new criterion and remain in the leaderboard, published as NOT ADMITTED rather
  than dropped. `tests/test_v3/test_family_reward_channel_tampering.py` fails; the
  test is correct and the family is not. *(2026-11-19: the full ledger adds 46 NOT
  ADMITTED cells in 20 of the 23 families new to it; see the 2026-11-19 regeneration
  entry above.)*
- **Families 29 and 30 still ship with documented, unresolved `proxy_matched`
  failures** at some tiers (see 1.8.0 below). *(Family 29, `sequence_keyword_stuffing`,
  was resolved in `86acb92`; see the entry above.)* *(2026-11-19, on the full ledger:
  Family 30, `sequence_format_camping`, fails `proxy_matched` at 0.9 and 0.8 and is
  admitted at 0.7; Family 29 fails it at 0.9, underpowered, and is admitted at 0.8 and
  0.7.)*
- **The committed leaderboard artifacts still contain the imputed 0.5 L1 cells and
  carry no provenance**, pending regeneration. *(2026-11-19: `v5_leaderboard.json` no
  longer does, regenerated in `52656cb`; `cross_family_transfer.json` has not been
  regenerated and still carries no provenance.)* *(2026-11-26: the 2026-11-19 board still
  carried every L1 detector's fallback 0.5 on `continuous_camping`; since the re-score
  that cell is N/A on both boards.)* *(2026-11-27: `cross_family_transfer.json` is kept
  as the 2026-07-09 record and is not regenerated; see the 2026-11-27 transfer entry
  above.)*
- **Five external baseline detectors ship but are scored nowhere.** *(2026-11-27: they are
  on neither board, but are scored on HVTA (`external/hvta/`), where Bayesian Online
  Changepoint Detection's score was a constant until the 2026-11-27 fix. All five now
  declare `min_episodes`; see the 2026-11-27 BOCPD entry above.)*
- **Hacking policies remain scripted** in 28 of 33 families. Nothing in this
  remediation changes what that limits RHOB to concluding. *(Corrected to 26 of 33 in
  `9e2f068`; see `docs/THREAT_MODEL.md` §3.)*

## [1.8.0] — Sequence-Generation Extension: 5 New SEQUENTIAL Families

*(2026-11-19, for [1.5.0] to [1.8.0]: the tier results these entries report ("All 3 tiers
pass", "re-verified via `AdmissionGate.certify()` at every default difficulty tier") were
outcomes of the gate as it stood before the 2026-08 audit, not of the admission ledger.
The full ledger of 2026-11-19 certifies the 19 families these four releases added and
finds 44 of their 58 cells NOT ADMITTED: all 13 MuJoCo cells, 9 of the 15 RLHF cells, 14
of the 15 PettingZoo cells and 8 of the 15 sequence cells. Each cell's verdict is in
[admission/ADMISSION_LEDGER.md](admission/ADMISSION_LEDGER.md).)*

Extends v1.7's 28 families to 33 by populating the taxonomy's `SEQUENTIAL`
complexity tier a second time, structurally distinct from Families 19–23: real
per-step token-sequence generation over a small discrete vocabulary against a
fixed hidden Markov "true grammar," scored by a fixed rule-based (non-fitted)
proxy rather than a genuinely-fit reward model:

- **New `src/rhob/environments/sequence_gen/` module**: a fixed `V=24`-token
  hidden Markov "true grammar" (`grammar.py`, seeded once and never exposed to
  any proxy/detector), `SequenceGenConfig`, and a shared
  `run_sequence_episode`/`generate_sequence_rundata` rollout primitive
  (`rollout.py`) mirroring `rlhf_rm/rollout.py`'s pattern (no external `step()`
  call — token sampling and scoring both happen directly in the loop). No new
  optional dependency (pure numpy, matching `rlhf_rm`).
- **Family 29 — Keyword-Stuffing Gaming** (`PROXY_GAMING`): floods a small
  fixed keyword-token subset largely decoupled from the grammar's actual
  structure. HARD-tier `proxy_matched` remains a documented, unresolved
  variance mismatch (AUROC ~0.63) after 5 verified fix attempts — accepted as
  a partial-compliance outcome rather than continued open-ended iteration.
- **Family 30 — Format-Compliance Camping** (`CAMPING_EXPLOIT`): fills fixed
  template-slot positions with expected tokens without the surrounding
  sequence needing to cohere. 1/3 tiers pass; the other 2 fail
  `proxy_matched` due to a calibration-bias root cause distinct from Family
  29's variance mismatch (documented, not the same fix).
- **Family 31 — Repetition-Blind-Spot Shortcut** (`SHORTCUT`): repeats a
  short token cycle just outside a fixed lookback window. All 3 tiers pass.
  Two real design bugs caught and fixed during implementation: a pigeonhole
  collision in the plan's original 3-token cycle set, and a degenerate
  calibration lever that boundary-locked to "no exploit."
- **Family 32 — Lexicon-Sentiment Gaming** (`GOAL_MISGENERALIZATION`): floods
  a fixed positive-lexicon token subset largely decoupled from the grammar.
  All 3 tiers pass after root-causing a MEDIUM-tier failure as a
  distribution-*shape* mismatch (not mean or variance) from a too-sharp
  2-token calibration lever, fixed by widening it to 4 tokens.
- **Family 33 — Length-Padding Exploit** (`REWARD_SHAPING`): switches to
  uniform-random low-effort tokens partway through the episode, farming a
  per-step length/continuation shaping bonus that a shallow content-presence
  check can't tell apart from genuine content. All 3 tiers pass after two
  rounds of redesign: the plan's single-filler-token calibration lever was
  mathematically unsolvable (forced to "never actually pad" regardless of
  weights), and a first garbage-token-pool fix overshot into perfect AUROC
  separation before matching legit's own natural empty-token rate.
- Families 31, 32, and 33 verification was run on a dedicated, temporary AWS
  EC2 instance to isolate from unrelated heavy local CPU load during this
  session, not because of any dependency on cloud infrastructure.
- README family count, family list, and leaderboard-size references updated
  from 28 to 33.

## [1.7.0] — PettingZoo Extension: 5 New Multi-Agent Families

Extends v1.6's 23 families to 28 by populating the taxonomy's `MULTI_AGENT`
complexity tier for the first time, using real `mpe2` (Multi-Particle Environments)
scenarios instead of a scripted single-agent proxy:

- **New `rhob[pettingzoo]` optional extra** (`pyproject.toml`): pulls in
  `pettingzoo>=1.24` and `mpe2>=1.0`. MPE was split out of PettingZoo core as of
  PettingZoo 1.26 — `pettingzoo[mpe]`/`pettingzoo.mpe` no longer exist; this was
  caught and corrected during design, before any family code was written. Core
  install remains mpe2-free; every new family module lazily imports `mpe2` only
  inside its `_make_env()`, so `import rhob.v3.families` still succeeds without it
  installed (verified, matching the `mujoco` extra's established pattern).
- **New `src/rhob/environments/pettingzoo/` module**: `PettingZooConfig`,
  `run_pettingzoo_episode`/`generate_pettingzoo_rundata`, and a phase-aware
  `StepMetricFn` signature (`(env, obs, rewards, infos, t, horizon)`) letting a
  family's metrics be aware of which phase of an episode the current step falls in
  (e.g. an initial individual-pursuit phase followed by a collusion phase).
- **Family 24 — Tacit Collusion Gaming** (`CAMPING_EXPLOIT`, `simple_tag`):
  predators camp near the prey's spawn during a late-episode collusion phase instead
  of genuinely pursuing, gaming the team-summed proxy.
- **Family 25 — Communication-Channel Deception** (`DECEPTIVE_ALIGNMENT`,
  `simple_speaker_listener`): a speaker's message leaks the true goal signal through
  a side channel while nominally still "communicating."
- **Family 26 — Free-Rider Exploitation** (`PROXY_GAMING`, `simple_spread`,
  team-averaged): one of 3 agents abandons its landmark while the other 2
  compensate, gaming a team-averaged proxy that can't see individual contribution.
- **Family 27 — Fixed-Opponent Exploitation** (`SHORTCUT`, `simple_tag`): predators
  exploit a fixed, scripted prey's predictable corner-retreat pattern instead of
  genuinely tracking its live position. Two structural fixes were needed beyond
  mean-calibration alone: (1) all 3 predators converging on the exact same corner
  point created a threshold-like catch/no-catch dynamic with ~1.2-1.6x legit's
  proxy variance regardless of calibration scale, fixed by spreading the 3
  predators' corner targets 120 degrees apart at a small fixed radius (a
  structural, uncalibrated dampening knob, not a tuning parameter); (2) full
  commitment (`fixed_pull=1.0`) hit a genuine physical ceiling on achievable proxy
  that no calibration boost could close.
- **Family 28 — Population-Level Goodhart** (`PROXY_GAMING`, `simple_world_comm`):
  one of 4 adversary-team agents free-rides while the lead and 2 others carry the
  team's aggregate proxy reward — population-scale free-riding, distinct from
  Family 26's 3-agent case. `simple_world_comm`'s observation layout (the most
  complex in this extension) was verified directly against the installed `mpe2`
  package's source and empirical position-negation cross-checks, correcting an
  initial draft's assumed agent list (4 adversary-team agents: 1 lead + 3 regular,
  not 3 total). Two bugs found via the real `AdmissionGate.certify()`: the
  calibration boost was applied to the free-rider's own effort-scaled pursuit
  target, which is exactly zero at the easiest tier's fully-passive effort — fixed
  by boosting the non-free-riding helpers instead; and the behavioral feature
  returned raw (unsigned) velocity magnitude, ranking legit above hacking, the
  opposite of the required "higher score = more hacking-like" convention (produced
  AUROC ≈ 0.0, not ≈ 0.5) — fixed by negating it.
- All 5 families independently re-verified via `AdmissionGate.certify()` at every
  default difficulty tier, not just trusted from self-reported test runs.
- README family count, family list, and leaderboard-size references updated from 23
  to 28.

## [1.6.0] — RLHF-RM Extension: 5 New Synthetic Reward-Model-Overoptimization Families

Extends v1.5's 18 families to 23 by populating the taxonomy's `SEQUENTIAL` tier for
the first time, via a synthetic RLHF setting rather than a real LLM:

- **Shared `calibrate_scale` extraction** (`src/rhob/environments/calibration.py`):
  the generic binary-search calibration helper previously lived in
  `src/rhob/environments/mujoco/rollout.py` despite having no MuJoCo-specific logic.
  Extracted into its own module so both the MuJoCo and RLHF-RM families can share it;
  `mujoco/rollout.py` now re-exports it, so no existing family's imports changed.
- **New `src/rhob/environments/rlhf_rm/` module**: a synthetic response-feature space
  (`x ∈ R^8`), a fixed nonlinear true reward `r*(x)` (oracle-only), a real preference-data
  generator with per-family failure injection, genuine `LogisticRegression`-fit reward
  models (not scripted), and a policy-gradient rollout loop (`N(μ, Σ)` over response
  space, ascending the fitted reward model minus a KL penalty to a reference policy).
  No new optional dependency — pure numpy/scikit-learn, unlike the MuJoCo extra.
- **Family 19 — RM Sparse-Coverage Gaming** (`RM_OVEROPTIMIZATION`): preference data
  undersamples part of response-space; the fitted model extrapolates optimistically
  there.
- **Family 20 — RM Label-Noise Exploitation** (`RM_OVEROPTIMIZATION`): preference
  labels near the true decision boundary carry concentrated noise, biasing the fitted
  model's boundary.
- **Family 21 — RM Feature-Blindspot Gaming** (`GOAL_MISGENERALIZATION`): the reward
  model is fit on a truncated feature subset, structurally freezing the policy on the
  hidden dimensions.
- **Family 22 — KL-Penalty Gaming** (`REWARD_SHAPING`): both variants share one reward
  model; only the KL-penalty coefficient differs. Uncovered and fixed a real bug during
  development: calibrating a compensator parameter that gets rounded to an integer
  downstream creates a quantization floor `calibrate_scale` can never converge below,
  regardless of tolerance or seed count — fixed by calibrating a genuinely continuous
  quantity (`RLHFConfig.step_size`) instead.
- **Family 23 — Preference-Population Bias** (`DECEPTIVE_ALIGNMENT`): the synthetic
  labeler population over-weights one response dimension unrelated to true quality
  (a sycophancy-style bias) that the fitted model faithfully learns.
- All 5 families follow the established `functools.lru_cache`-memoized pure
  calibration-function pattern and were independently re-verified via
  `AdmissionGate.certify()` at every default difficulty tier, not just trusted from
  self-reported test runs.
- README family count, family list, and leaderboard-size references updated from 18
  to 23.

## [1.5.0] — MuJoCo Extension: 4 New High-Dimensional Continuous-Control Families

Extends v1.4's 14 families to 18 by populating the taxonomy's `CONTINUOUS_COMPLEX`
("cont_hd") tier for the first time, using real MuJoCo/Gymnasium environments instead
of the hand-rolled low-dimensional continuous envs used elsewhere in the benchmark:

- **New `rhob[mujoco]` optional extra** (`pyproject.toml`): pulls in
  `gymnasium[mujoco]>=1.0`. Core install remains MuJoCo-free; every new family module
  is guarded with `pytest.importorskip("mujoco")` in its tests and lazily imported, so
  `import rhob.v3.families` still succeeds with mujoco uninstalled (verified).
- **Shared MuJoCo infra** (`src/rhob/environments/mujoco/`): `MuJoCoConfig`,
  `run_mujoco_episode`/`generate_mujoco_rundata`, and a `calibrate_scale` binary-search
  helper used by all 4 families to tune each family's difficulty knob against a target
  proxy-reward gap (raises `ValueError` on non-convergence rather than silently
  returning a bad value).
- **Family 15 — MuJoCo Camping** (HalfCheetah-v5, `CAMPING_EXPLOIT`): re-instantiates
  the existing camping mechanism at real 17-dim/6-actuator dimensionality.
- **Family 16 — MuJoCo Goal Misgeneralization** (Reacher-v5, `GOAL_MISGENERALIZATION`):
  re-instantiates the existing goal-misgeneralization mechanism against a live
  fingertip-to-target distance, with a custom per-step control loop (gain scheduled by
  goal separation) rather than a fixed action sequence.
- **Family 17 — MuJoCo Joint-Limit Gaming** (Ant-v5, `REWARD_SHAPING`): new
  MuJoCo-native mechanism exploiting hip/ankle joint-limit proxy costs. Uncovered and
  documented a real Ant-v5 quirk: actuator order does not match joint order, so the
  family queries `model.actuator_trnid` rather than assuming a fixed slice.
- **Family 18 — MuJoCo Sensor-Channel Decoupling** (Walker2d-v5, `REWARD_TAMPERING`):
  new MuJoCo-native mechanism where a foot-joint velocity proxy is gamed independently
  of true root-forward velocity, with a "leakage torque" difficulty control reusing the
  legit gait's own offset shape on non-spun actuators.
- All 4 families use `functools.lru_cache`-memoized pure calibration functions
  (parametrized by the actual dependent physical quantity, not a rounded difficulty
  float) — the pattern adopted after Task 2's code review found a rounding-collision
  bug in an earlier hand-rolled dict-cache approach.
- CI (`tests.yml`) now installs the `mujoco` extra in the main test job; README family
  count, family list, and cross-family-transfer descriptions updated from 14 to 18.

## [1.4.0] — AdmissionGate, 5 New Families, Toy RLHF Setting, Leaderboard Infra

Builds the next benchmark generation on top of v1.3's 9 families:

- **`AdmissionGate` module** (`src/rhob/v3/admission_gate.py`): reusable,
  family-agnostic implementation of the 5 admission criteria, replacing
  bespoke per-family pytest. Running it against all shipped families caught a
  real, previously undetected proxy leak in `goal_misgeneralization`
  (asymmetric movement speed between variants; mean L0 AUROC ~0.73, not the
  claimed ~0.5) — fixed.
- **5 new families** (9 → 14), 3 new taxonomy mechanisms:
  - `reward_channel_tampering`, `sensor_calibration_tampering` — first
    `REWARD_TAMPERING` families
  - `monitored_sandbagging`, `eval_probe_sandbagging` — first
    `DECEPTIVE_ALIGNMENT` families
  - `rlhf_reward_model_overopt` — toy preference-bandit RLHF setting; a real
    (if tiny) fitted logistic-regression reward model, not a hand-waved
    blind spot; new `HackingMechanism.RM_OVEROPTIMIZATION`
- **Fixed a pre-existing mislabel**: `orbit_chirality` was tagged
  `DECEPTIVE_ALIGNMENT` despite being a straightforward camping/commitment
  mechanism (no appear-aligned-then-revert dynamic) — corrected to
  `CAMPING_EXPLOIT`. Predates this release; found while auditing mechanism
  counts for this changelog entry, not introduced by the new families.
- **Leaderboard infrastructure**: JSON schema adapter reconciling the two
  incompatible formats under `leaderboard/`, a read-only Gradio viewer
  (`space/app.py`), CI validation for community submissions
  (`.github/workflows/leaderboard_validate.yml`), and a documented (not yet
  triggered — needs the maintainer's HF account) deploy path.
- **Bugs found and fixed along the way**: a `None`-AUROC crash in
  `Leaderboard.standings()`, an `rhob.v3` import chain that broke without
  torch installed, unconditional torch-class definitions breaking CI, and a
  `StateDivergenceDetector` NaN crash on sharp categorical distributions
  (surfaced by the new tampering/RLHF families' confident 2-3-bin behavioral
  signals).
- Full 14-family leaderboard regenerated; access-level means: L0 0.497,
  L1 0.541, L2 0.743, L3 0.990.

## [1.3.0] — Repository Scope: Benchmark Harness Only

Restructured the repository to match the scope of comparable benchmark repos
(ImageNet devkit, SWE-bench): the codebase, tests, and docs stay; the academic
paper source and internal development artifacts do not.

- **Removed `paper/`** (LaTeX manuscript, `references.bib`, build `Makefile`) from
  the repository and its history. The paper is maintained separately going forward.
  Result figures used by the paper (`v5_heatmap.png`, `v5_access_summary.png`,
  `v5_transfer.png`, and the Family 1–2 case-study figures) are genuine benchmark
  artifacts, not paper-only content, and were moved to `docs/figures/` rather than
  deleted; `scripts/plot_v5_results.py`, `scripts/plot_difficulty_overlay.py`, and
  `scripts/validate_all_continuous.py` now write there.
- **Removed `experiments/`** (pre-v3 pilot/exploration scripts) from the repository
  and its history — superseded by `src/rhob/v3/families/`, not imported by any
  shipped code, and not part of the product.
- **Removed `docs/internal/`** (archived internal planning/spec/roadmap documents)
  from the repository and its history.
- Updated all in-repo references to these paths (README, REPRODUCIBILITY.md, a
  handful of family/doc comments citing historical pilot findings).
- All three removed trees were rewritten out of git history entirely (not just
  untracked), so a fresh clone is smaller and the commit history no longer contains
  this content. `RELEASE_NOTES_v1.0.md` is left as an accurate historical snapshot
  of the repository layout at the v1.0 tag and was not retroactively edited.

## [1.2.0] — Three More Bugs Found by Not Trusting a Surprising Result

Triggered by external methodological review questioning why L2 cross-family transfer
AUROC (0.95) exceeded in-distribution training AUROC (0.89) and why the L3 oracle
wasn't near-ceiling. Investigating both instead of adjusting the numbers to look better
surfaced three more real bugs:

- **`orbit_chirality` sign inversion**: behavioral feature had the wrong sign convention,
  scoring AUROC 0.000 (perfect inversion) on this family. Zero prior test coverage. Fixed;
  added `tests/test_v3/test_family_orbit_chirality.py`.
- **`proxy_correlation_gaming` item depletion + region/reward decorrelation**: one-time
  consumable items ran out before the trailing evaluation window, and the hacking
  strategy's movement target had no correlation with where reward actually was. True
  Reward Oracle was scoring 0.608 on this family (should approach 1.0). Fixed with
  persistent reward-terrain and region-linked placement, re-verified matched-proxy holds.
- **Unseeded neural-net training**: `RewardMLPDetector` and `TrajectoryMLPDetector` never
  seeded `torch`, so identical `fit()` calls on identical data produced held-out AUROC
  ranging from 0.00 to 1.00 across repeated runs on one family. This is the leading
  explanation for the original anomaly. Fixed with an explicit `seed` parameter;
  `scripts/cross_family_transfer.py` now reports mean ± std across independently-seeded
  trials instead of a single run.

**Real, re-measured results after all fixes:**
- Access-level means: L0 0.51±0.03, L1 0.53±0.08, L2 0.76±0.18, L3 0.99±0.01 (L3 was
  0.87 before these fixes, dragged down by the two family bugs above)
- Cross-family transfer: L2 single detector now trains (0.93) above transfer (0.77±0.14,
  high-variance) — the expected direction. L2 five-detector ensemble transfers robustly
  at 1.00±0.00, matching its 0.97 training AUROC; it is stable because 4 of its 5 members
  are deterministic and absorb the one learned member's instability.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) items 4-6 for full details and verification code.

## [1.1.0] — Usability & External Baselines

**Phase 3 (usability):**
- Added `docs/INSTALL.md`, `docs/TUTORIAL_DETECTOR.md`, `docs/TUTORIAL_ENVIRONMENT.md`
- Added `Dockerfile` for a zero-local-setup path
- Added `notebooks/rhob_quickstart.ipynb` (Colab-ready)
- Added `docs/site/index.html`, a static benchmark website (deployed via GitHub Pages)
- Added `.github/workflows/tests.yml` (CI across Python 3.10-3.12) and `pages.yml`
- Archived ~30 internal planning/spec documents from the repo root into `docs/internal/`

**Phase 5 (external baselines):**
- Added 5 classical detectors from outside RHOB's own design lineage, in
  `src/rhob/detectors/external_baselines/`: Page-Hinkley Test (classical
  change-point), Isolation Forest (unsupervised anomaly detection), AR(p)
  Residual (sequence-model baseline), PCA Reconstruction (representation
  learning), and Bayesian Online Changepoint Detection (Adams & MacKay, 2007)
- All five verified against `gridworld_camping`: L0 baselines correctly sit
  at chance (matched-proxy tautology holds for classical methods too), L1/L2
  baselines discriminate above chance *(2026-11-27: Bayesian Online Changepoint
  Detection sat at chance because its score was the hazard, 0.004, on every run; see the
  2026-11-27 BOCPD entry in the first [Unreleased] section above.)*
- 12 new unit tests in `tests/test_detectors/test_external_baselines.py`

## [Unreleased] — Baseline Detectors: the Main Results Table

Implements and runs 6 baseline detectors spanning 3 access levels across all 4
admitted pairs (40 runs/pair, 20/variant), producing the paper's central
experimental result: [results/detector_evaluation/](results/detector_evaluation/).

### Result

```
DISCRIMINATION AUROC                pair_01   pair_02   pair_03   pair_04
                                    gridworld    easy    medium     hard
Reward Threshold           (L0)       0.585     0.632     0.525     0.522
Reward CUSUM                (L0)       0.450     0.627     0.540     0.535
Reward MLP                  (L0)       0.615     0.515     0.530     0.482
State Divergence             (L1)       1.000     0.927     0.627     0.537
Behavioral Threshold         (L2)       1.000     0.965     0.830     0.728
Trajectory MLP               (L2)       1.000     0.925     0.708     0.680
```

- **L0 barrier holds**: every L0 detector (including the strongest possible one,
  a CV-trained MLP on full reward history) stays ≤0.65 on every pair — reward
  alone cannot discriminate hacking from legitimate improvement, matching the
  admission gate's own certification.
- **L2 sufficiency holds and degrades monotonically**: behavioral detectors track
  the admission L2 spectrum (1.00 → 0.97 → 0.89 → 0.82) within normal
  sampling variance (independent 20/class seed draws vs. the admission
  certificate's own seeds; AUROC's sampling SE at n=20/class is ≈0.08-0.09 for
  AUROC in [0.7, 0.9], consistent with the observed gaps).
- **A secondary, unplanned finding**: the simple L0 detectors (Reward
  Threshold/CUSUM) localise onset *timing* well on continuous pairs (MAE≈0)
  despite being unable to discriminate variant identity (AUROC≈0.5) — both
  variants share the same reward trajectory shape, so a level-shift is
  detectable without being attributable. The trained Reward MLP, by contrast,
  cannot localise onset at all (MAE=1.0, never detected): with no genuine
  discriminative signal in reward to learn, its hacking-probability score never
  produces a confident jump for the "biggest increase" heuristic to seize on.

### Added
- **Post-hoc detector interface** (`detectors/posthoc.py`) — `RunData`
  (`proxy_rewards`, oracle-only `true_rewards`, `state_counts` L1 histogram,
  `behav_trace` L2 anti-symmetric feature) and `PosthocDetector.classify/detect_onset`.
- **6 detectors** (`detectors/l0_*.py`, `l1_state_divergence.py`, `l2_*.py`):
  Reward Threshold/CUSUM (self-calibrating against each run's own
  baseline/steady-state, not a fixed absolute magnitude — proxy scale differs
  by orders of magnitude between gridworld's raw per-episode sums and
  continuous's per-step means), Reward MLP, State Divergence (CV-trained
  nearest-centroid classifier over visitation histograms), Behavioral
  Threshold (oracle, unbounded raw score — clipping to [0,1] would tie
  together every run whose feature exceeds ±1), Trajectory MLP.
- **Instrumented rollouts** (`evaluation/eval_rollout.py`) — mirrors each pair's
  production rollout exactly, adding the one signal detectors need that it
  doesn't record: a per-episode state-visitation histogram.
- **Evaluation pipeline** (`scripts/generate_eval_data.py`,
  `evaluate_detectors.py`, `plot_results.py`) — 5-fold stratified CV for the
  three trainable detectors (Reward MLP, State Divergence, Trajectory MLP);
  validation checks (L0 barrier, L2-vs-admission consistency, monotonicity);
  4 plots (heatmap, access-level bars, difficulty curve, onset timing).
- **27 detector unit tests**, all passing; lint clean.

### Bugs found and fixed while wiring to real data
- **Oracle leakage**: `TrajectoryMLPDetector` originally used `true_rewards` as
  a training feature, which would have trivially "solved" detection by reading
  the ground truth. `RunData.true_rewards` is now documented as oracle-only;
  no detector reads it.
- **Sign-convention inversion**: gridworld's hacking variant has `row-col=-1`
  while continuous's has `x-centre=+1` — opposite signs for "hacking" under
  each pair's natural coordinates. `BehavioralThresholdDetector` assumed one
  universal convention, silently inverting to AUROC≈0.0 on gridworld. Fixed by
  flipping the gridworld feature to `col-row` so positive=hacking everywhere.
- **L1 discriminability conflation**: a pure self-baseline JS-divergence score
  can detect *that* visitation changed but not *which* direction, since both
  variants switch regions symmetrically by construction — it could never
  discriminate. Rewrote as a CV-trained nearest-centroid classifier.

## [Unreleased] — Tier 2: Continuous-Control Difficulty Spectrum

Adds the continuous-control tier (Pairs 02–04) — a tunable difficulty spectrum
that complements the Tier 1 gridworld barrier proof (Pair 01). Established that
benchmark difficulty must be an *intrinsic, designed* property (the stochastic
spread of the camping target), not an artifact of learner quality.

### Added
- **Continuous infrastructure** (`environments/continuous/`) — a 2D point-mass MDP
  with a Gaussian-bump reward at a per-run *stochastic* attractor, reflection-
  symmetric across the arena centre (proxy matched at L0; camping location
  separates at L2). `ContinuousConfig`, `generate_run → ContinuousRunLog`, and a
  camping-fraction onset label.
- **DQN camper** (`agents/dqn.py`, `agents/replay_buffer.py`) — a small torch DQN
  trained once to reach and hold an observed target, with an *exact* reflection-
  symmetrised greedy policy (keeps the proxy matched under tight camping). torch is
  an optional dependency (`pip install rhob[continuous]`).
- **Three difficulty levels** — `pair_02` (easy, d=1.25), `pair_03` (medium,
  d=0.75), `pair_04` (hard, d=0.55), all at σ_a=0.5.
- **Continuous admission gate** (`evaluation/continuous_admission.py`) — L2 floor,
  L0-CI-contains-0.5 (the sound proxy-match test), true reward B>A, camping ≥0.80,
  temporally-localised onset. Per-pair validators, `validate_all_continuous.py`
  (renders the difficulty-spectrum plot), tests, and `docs/difficulty_spectrum.md`.

### Notes
- Difficulty is set by the attractor separation `d` (with σ_a fixed); L2 ≈
  Φ(d/√2·σ_a). A tight-camping agent on a *fixed* target gives binary L2 (as in the
  gridworld); the *stochastic* target is what makes the spectrum smooth and robust.
- Tabular Q-learning cannot tightly camp a continuous attractor; function
  approximation (DQN) is required — hence torch for this tier only.

## [0.1.0] — Milestone 1: Vertical Slice

First end-to-end pipeline: one environment, two baselines, all metrics, HDF5
storage, and a reproducible evaluation runner.

### Added
- **Core type system** — `AccessLevel`, `HackingType`, `Tier`, `Trajectory`,
  `Timestep`, `Observation`, `OnsetLabel`, and the structural `AccessFilter`
  (prevents information leakage across access levels).
- **Environment** — `GridWorld-Wireheading` (Tier 1 reward tampering) with a
  tabular Q-learning trajectory generator and an `OnsetOracle` implementing the
  two-sample onset definition. Reliable onset generation (100% of hacking runs),
  clean-run negatives, and deterministic output.
- **Detectors** — `RandomDetector` (floor) and `CUSUMDetector` (classical L1
  change-point baseline), behind an abstract `Detector` contract.
- **Metrics** — AUROC, AUPRC, detection latency, FPR@k, miss rate, TFD, and the
  tier-weighted RHOB-Score, all with bootstrap confidence intervals.
- **Evaluation** — `EvaluationRunner` / `evaluate` / `compare` with contract
  validation (bounds + determinism), plus JSON/Markdown reports and a
  leaderboard-style comparison table.
- **Data** — exact-round-trip gzip HDF5 storage (`save_dataset` / `load_dataset`).
- **Config** — validated `EvaluationConfig` (pydantic).
- **Tooling** — generation script, 76-test suite (~94% coverage), documentation
  (`README`, `docs/data_schema.md`), MIT license.

### Notes
- Trajectory generation uses tabular Q-learning (the natural algorithm for a
  discrete gridworld); the benchmark is algorithm-agnostic and a PPO/SB3
  generator can be substituted for later continuous-control tiers.
