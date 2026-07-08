# Changelog

All notable changes to RHOB are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
semantic versioning.

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
