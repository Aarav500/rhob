# Changelog

All notable changes to RHOB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

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
  baselines discriminate above chance
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
