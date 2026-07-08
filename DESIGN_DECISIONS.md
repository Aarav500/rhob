# RHOB Design Decisions

**Document:** DESIGN_DECISIONS · **Status:** Living record
Preserves the reasoning behind the benchmark. Every significant engineering and
research decision from Milestone 1 is recorded with: **Problem · Alternatives
considered · Final decision · Rationale · Trade-offs · Long-term implications**.

Decisions tagged **[Debt]** knowingly deferred work tracked in
[`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) / [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md).

---

## A. Packaging & dependencies

### D1. `src/` layout
- **Problem:** how to lay out the package so tests exercise the installed artifact.
- **Alternatives:** flat `rhob/` at repo root.
- **Decision:** `src/rhob/…` with `pythonpath=["src"]` for tests.
- **Rationale:** flat layouts import the working tree and hide packaging bugs;
  src-layout forces the install path a user takes.
- **Trade-offs:** minor setup friction (path or editable install).
- **Long-term implications:** packaging bugs surface early; safe base for PyPI.

### D2. Minimal, tiered dependencies
- **Problem:** the spec listed `stable-baselines3` (→ torch) and `gymnasium` as
  core deps, but neither is used.
- **Alternatives:** keep them core; vendor a light RL lib.
- **Decision:** core = `numpy, scipy, scikit-learn, h5py, pydantic, click`; SB3 /
  gymnasium moved to an optional `environments` extra.
- **Rationale:** a grep confirmed no imports; forcing multi-GB torch contradicts a
  tabular design and hurts adoption.
- **Trade-offs:** the PPO generation path is opt-in, not built-in.
- **Long-term implications:** `pip install rhob` stays light and torch-free.

---

## B. Trajectory generation (environment dynamics)

### D3. Tabular Q-learning generator (not PPO/SB3)
- **Problem:** produce reliable, reproducible training runs that exhibit onset.
- **Alternatives:** PPO via SB3 (as the spec suggests), SAC, DQN, synthetic curves.
- **Decision:** tabular Q-learning for the discrete gridworld.
- **Rationale:** SB3 was unavailable (heavy) and PPO on a tiny gridworld is slow,
  stochastic, non-deterministic, and unreliable at producing clean onsets;
  tabular Q-learning is the natural, fast, deterministic, stable choice.
- **Trade-offs:** deviates from the spec's stated algorithm; not directly a
  continuous-control demonstration.
- **Long-term implications:** upholds algorithm-agnosticism; a PPO/SB3 generator is
  documented for continuous tiers; the `algorithm` field records the generator.

### D4. Training-episode time axis (training-curve representation)
- **Problem:** what does "step `t`" of a trajectory mean?
- **Alternatives:** within-episode environment steps (s,a,r,s′).
- **Decision:** each step is one training **episode**; record per-episode returns.
- **Rationale:** onset is about *when during training* objectives diverge; the
  env-step axis answers a different question and buries the signal.
- **Trade-offs:** not directly comparable to per-transition anomaly detectors.
- **Long-term implications:** detectors monitor the training process, matching the
  real "is this run hacking?" use case.

### D5. Record the greedy policy's return
- **Problem:** which return to log per training step.
- **Alternatives:** noisy behaviour-policy (ε-greedy) episode return.
- **Decision:** the greedy policy's deterministic rollout return.
- **Rationale:** it estimates `E_{π_t}[R]` (the quantity the onset definition
  uses) and is (near) noise-free and deterministic.
- **Trade-offs:** curves are smoother/less "realistic" than raw training noise.
- **Long-term implications:** clean, deterministic curves; the oracle's windowed
  test still handles them; L2 features come from the same rollout.

### D6. Reliable onset via **scheduled exploit activation** (core research decision)
- **Problem:** make reward hacking emerge reliably, reproducibly, and with varied
  onset times across seeds — pure exploration would not.
- **Alternatives (each built and rejected):** (1) terminating goal + distant
  wirehead — episodes truncate, exploit never found; (2) fixed-horizon reach-and-
  hold + interior wirehead — interior cells can't be camped, goal wins; (3)
  campable edge wirehead — goal's hold action was the default `argmax`, a fragile
  index artifact; (4) STAY action + optimistic init — deterministic across seeds
  or, with noise, unstable/non-converging.
- **Decision:** ε-greedy Q-learning + a STAY action + wirehead adjacent to the
  goal + the exploit activating at a per-seed episode `E_activate`.
- **Rationale:** it makes onset generation reliable (100%), deterministic per
  seed, seed-varied (usable CIs), and grounded in real Q-learning, after the four
  alternatives empirically failed.
- **Trade-offs:** the emergence is *scheduled*, not spontaneous — a modeling
  assumption a reviewer may probe.
- **Long-term implications:** dependable data generation; detectors never see
  `E_activate`; the difficulty knob becomes activation timing (D9).

### D7. Explicit STAY action (5 actions)
- **Problem:** "holding" a cell depended on wall-bump artifacts, asymmetric
  between goal and wirehead.
- **Alternatives:** four movement actions only.
- **Decision:** add a STAY action.
- **Rationale:** makes camping any cell symmetric, so reward magnitudes — not
  geometry accidents — determine behaviour.
- **Trade-offs:** `action_dim = 5`; slightly larger action space.
- **Long-term implications:** robust, artifact-free dynamics reusable by future
  tabular environments.

### D8. Clean runs = exploit inactive
- **Problem:** produce controllable clean negatives.
- **Alternatives:** treat seeds that happen not to hack as clean.
- **Decision:** clean-configured runs never activate the exploit.
- **Rationale:** deterministic and controllable; "happens not to hack" is really a
  *missed* hack and can't hit a target ratio.
- **Trade-offs:** clean and hacking runs differ by construction, not by chance.
- **Long-term implications:** a controllable 70/30 split with unambiguous
  legitimate-learning negatives.

### D9. Difficulty = activation timing
- **Problem:** what should the `tamper_accessibility` knob control?
- **Alternatives:** spatial distance to the wirehead (the spec's framing).
- **Decision:** higher accessibility → earlier `E_activate` → earlier onset.
- **Rationale:** with the wirehead adjacent (D6), distance no longer gates
  discovery; timing is the meaningful lever.
- **Trade-offs:** a distance/discovery-difficulty axis is deferred.
- **Long-term implications:** difficulty controls onset timing and pre/post
  balance; other difficulty axes come with later environments.

---

## C. Onset definition & labeling

### D10. Two-sample windowed onset test
- **Problem:** define onset operationally on finite, noisy curves.
- **Alternatives:** CUSUM-oracle, regression/slope-based (spec variants).
- **Decision:** true significantly decreasing **and** proxy increasing across
  adjacent windows (Welch t-tests).
- **Rationale:** the blueprint's primary definition; directly encodes
  "objectives diverge."
- **Trade-offs:** favours sharp onsets; gradual sub-threshold onsets aren't
  labelled.
- **Long-term implications:** the other definitions become ablation variants for
  the sensitivity study.

### D11. `δ` threshold in **global** true-σ units
- **Problem:** near-constant greedy curves make within-window variance ≈ 0.
- **Alternatives:** pooled within-window σ.
- **Decision:** `δ = significance_delta × std(true_curve)`.
- **Rationale:** the global σ is stable and non-zero; the pooled σ collapses and
  destabilizes the test.
- **Trade-offs:** a global scale choice rather than a local one.
- **Long-term implications:** robust threshold; a zero-variance guard handles
  constant windows.

### D12. `lookback_k = 20` default (not 1000)
- **Problem:** the spec default `k=1000` exceeds short benchmark curves.
- **Alternatives:** keep `k=1000`; fully adaptive `k`.
- **Decision:** `k=20`, clamped by `_effective_k` for short curves.
- **Rationale:** `k=1000` targets million-step real runs; M1 curves are hundreds
  of episodes.
- **Trade-offs:** a scale-specific default to document.
- **Long-term implications:** onset params travel with the data; `k` scales with
  trajectory length.

### D13. Labels baked at generation **[Debt]**
- **Problem:** where to compute onset labels.
- **Alternatives:** label lazily at evaluation from stored `reward_true`.
- **Decision:** oracle runs in `generate()`; label stored (with `reward_true`).
- **Rationale:** keeps the vertical-slice runner simple and evaluation fast.
- **Trade-offs:** no `relabel()` API; `EvaluationConfig` onset params are inert.
- **Long-term implications:** the M4 onset-sensitivity ablation needs a re-label
  path (REFACTOR_PLAN #2) — the data supports it, the API doesn't yet.

### D14. `is_hacking_run = configured AND onset-detected` **[Debt]**
- **Problem:** how to mark a configured-hack that didn't manifest.
- **Alternatives:** a separate "missed-hack" status.
- **Decision:** treat a non-manifesting hack as a clean negative.
- **Rationale:** simplest boolean; moot at M1's 100% reliability.
- **Trade-offs:** conflates "clean" with "missed hack."
- **Long-term implications:** a data-integrity hazard once Tier 2/3 reliability
  < 100% (REFACTOR_PLAN #6).

---

## D. Core data model & access control

### D15. Structural access enforcement
- **Problem:** guarantee detectors can't see forbidden fields.
- **Alternatives:** honor-system convention; runtime assertions.
- **Decision:** `AccessFilter` builds a fresh **frozen** `Observation` with
  above-level fields `None` and features read-only.
- **Rationale:** convention leaks by accident and is untestable; the benchmark
  hinges on this separation.
- **Trade-offs:** a per-step object allocation + array copy.
- **Long-term implications:** leakage impossible by construction and tested; the
  per-step cost is a scaling target (REFACTOR_PLAN #8).

### D16. Array-based trajectories (`Timestep` unused)
- **Problem:** represent a trajectory efficiently.
- **Alternatives:** a list of per-step `Timestep` objects.
- **Decision:** dense columnar arrays; build `Observation`s on demand.
- **Rationale:** memory-efficient and HDF5-native; vectorizable metrics.
- **Trade-offs:** the `Timestep` dataclass ended up unused (cosmetic debt).
- **Long-term implications:** clean storage mapping; delete or wire in `Timestep`.

### D17. Visitation-distribution L2 features **[Debt]**
- **Problem:** what behavioural signal constitutes L2.
- **Alternatives:** raw state/action sequences; Q-value snapshots.
- **Decision:** normalized state-visitation over the greedy rollout.
- **Rationale:** compact and shifts structurally at onset (occupancy moves to the
  exploit) — the signal a structural detector would use.
- **Trade-offs:** env-specific dimensionality; no cross-env schema yet.
- **Long-term implications:** needs a declared `feature_spec` for the transfer
  claim (REFACTOR_PLAN #3).

### D18. L1/L2 implemented; L3/L4 declared **[Debt]**
- **Problem:** how many access levels to build now.
- **Alternatives:** implement L1–L4 fully; model levels as unordered tags.
- **Decision:** `IntEnum` L1–L4; `Observation` carries L1/L2 only.
- **Rationale:** M1 detectors are L1-only; L3/L4 plumbing with no consumer is
  speculative.
- **Trade-offs:** declaring L3/L4 currently degrades silently to L2.
- **Long-term implications:** resolve the contract while `Observation` is young —
  add fields or narrow the enum (REFACTOR_PLAN #3).

---

## E. Detector interface & baselines

### D19. Streaming `step()` interface
- **Problem:** how detectors consume data.
- **Alternatives:** a batch interface over the whole trajectory.
- **Decision:** per-step `step(obs) -> float`.
- **Rationale:** models real-time monitoring and enforces O(1)-per-step cost.
- **Trade-offs:** no vectorized path; even vectorizable detectors loop in Python.
- **Long-term implications:** add an optional batch path additively (REFACTOR_PLAN
  #8); interface stays deployment-realistic.

### D20. Seeded, reset-per-trajectory random baseline
- **Problem:** a floor baseline that satisfies the determinism contract.
- **Alternatives:** truly stochastic random.
- **Decision:** re-seed on `reset()`.
- **Rationale:** true randomness breaks reproducibility.
- **Trade-offs:** "random" is deterministic given a seed.
- **Long-term implications:** a reproducible ~0.5-AUROC floor.

### D21. Adaptive, level-aware CUSUM
- **Problem:** an honest L1 change-point baseline that isn't fooled trivially.
- **Alternatives:** fixed-reference classical CUSUM; a pure change-detector.
- **Decision:** accumulate standardized upward deviations against a running
  Welford mean/σ; squash to `[0,1]`.
- **Rationale:** a fixed reference fires throughout early learning; a pure change
  detector drops on the post-onset plateau — both hurt AUROC.
- **Trade-offs:** tuned constants (`slack_k`, `threshold_h`, `warmup`).
- **Long-term implications:** honest proxy-only L1 baseline; strong on sharp
  tampering (consistent with the "reward tampering is easiest" prediction).

---

## F. Metrics & aggregation

### D22. Pooled per-environment + per-trajectory metrics
- **Problem:** clean runs have no positives (undefined per-trajectory AUROC).
- **Alternatives:** per-trajectory only; single pooled only.
- **Decision:** pooled per-env AUROC/AUPRC (incl. clean negatives) **and**
  per-trajectory AUROCs for hacking runs.
- **Rationale:** pooling captures clean-run false alarms; per-trajectory enables
  paired significance tests.
- **Trade-offs:** two notions of AUROC to explain.
- **Long-term implications:** supports both leaderboard scoring and statistical
  comparison.

### D23. Tier-weighted RHOB-Score + bootstrap CIs **[Debt]**
- **Problem:** one citable ranking number.
- **Alternatives:** unweighted mean; Pareto/latency-first ranking.
- **Decision:** tier-weighted mean AUROC (1.0/1.5/2.0/2.5) with percentile
  bootstrap CIs.
- **Rationale:** a single scalar enables ranking; weighting rewards hard tiers.
- **Trade-offs:** the score-level CI is currently computed over too few points;
  secondary aggregates are unweighted.
- **Long-term implications:** move to a trajectory-level bootstrap (REFACTOR_PLAN
  #1) — the primary metric's headline CI.

### D24. Normalized latency semantics
- **Problem:** compare timeliness across environments of different length.
- **Alternatives:** raw step counts; clip early detections to zero.
- **Decision:** `(t_detect − t*)/T`; `inf` if never, `nan` if clean, negative if
  early.
- **Rationale:** normalization is cross-env comparable; clipping hides early
  firing.
- **Trade-offs:** negative/`inf`/`nan` require careful downstream handling.
- **Long-term implications:** clean cross-environment latency reporting; pair with
  absolute delay for interpretability.

---

## G. Storage

### D25. HDF5, gzip `float64`, exact round-trip
- **Problem:** a storage format that preserves determinism.
- **Alternatives:** `float32` (smaller), NPZ, Parquet, SQLite.
- **Decision:** per-run HDF5 groups, gzip `float64`, JSON attrs for label/metadata.
- **Rationale:** `float32` breaks exact save/load equality and thus the
  determinism guarantee; HDF5 fits grouped numeric data with metadata.
- **Trade-offs:** larger files than `float32`.
- **Long-term implications:** exact reproducibility; lazy/streamed loading needed
  at scale (REFACTOR_PLAN #9).

---

## H. Configuration & evaluation harness

### D26. Single `EvaluationConfig` (not the full hierarchy)
- **Problem:** how much config machinery M1 needs.
- **Alternatives:** the spec's five-level YAML/env precedence chain now.
- **Decision:** one frozen, validated pydantic `EvaluationConfig`.
- **Rationale:** the hierarchy is later-milestone scope and would bloat the slice.
- **Trade-offs:** onset fields are inert today (see D13); no file layering.
- **Long-term implications:** the full hierarchy (`CONFIG_SPEC`) builds on the same
  validated model.

### D27. Pre-scoring contract validation
- **Problem:** a broken detector can silently corrupt metrics.
- **Alternatives:** no validation; full up-front certification.
- **Decision:** check bounds (raise `ScoreBoundsError`) and determinism on a
  sample before scoring.
- **Rationale:** catches gross violations cheaply; full certification is heavier
  than M1 needs.
- **Trade-offs:** determinism is checked on only one trajectory.
- **Long-term implications:** centralize under a detector registry's
  `validate_contract` (REFACTOR_PLAN #5).

### D28. Minimal env registry + `TIER1` fallback **[Debt]**
- **Problem:** discovery and tier lookup without heavy machinery.
- **Alternatives:** no registry; entry-point plugins now.
- **Decision:** a dict-based registry (`register/list/get` + tier lookup);
  unregistered envs default to `TIER1`.
- **Rationale:** enough for discovery at one environment; entry-points premature.
- **Trade-offs:** import-hub coupling; silent tier fallback can mis-weight scores.
- **Long-term implications:** move to entry-points; make the fallback explicit
  (REFACTOR_PLAN #10).

---

## I. Testing & documentation

### D29. Consistency-based onset test (not hand-labels)
- **Problem:** the project plan's `MANUAL_LABELS` can't exist for generated data.
- **Alternatives:** fabricate a hand-label table.
- **Decision:** assert onset self-consistency (`validate_label`) + expected-window
  membership + a synthetic sharp-onset the oracle must recover.
- **Rationale:** a stronger, honest correctness check than circular hand-labels.
- **Trade-offs:** no property/gold-file tests yet.
- **Long-term implications:** add property + regression tests (REFACTOR_PLAN #7).

### D30. MIT license; onset formalized in `data_schema.md`
- **Problem:** licensing and where the formal definition lives.
- **Alternatives:** a separate `onset_definition.tex`; a restrictive license.
- **Decision:** MIT; onset definition + generation rationale in `docs/data_schema.md`.
- **Rationale:** adoption needs permissive licensing; a single data-contract source
  beats a fragmented TeX note in M1.
- **Trade-offs:** the formal write-up isn't yet in paper-ready TeX.
- **Long-term implications:** migrate the formalization into the paper during the
  theory milestone.

---

## Decision index

| ID | Decision | Debt? |
|----|----------|:-----:|
| D1 | src-layout | |
| D2 | Minimal deps; SB3/gymnasium optional | |
| D3 | Tabular Q-learning generator | |
| D4 | Training-episode time axis | |
| D5 | Record greedy-policy return | |
| D6 | Scheduled exploit activation (core) | |
| D7 | STAY action | |
| D8 | Clean = exploit inactive | |
| D9 | Difficulty = activation timing | |
| D10 | Two-sample onset test | |
| D11 | δ in global-σ units | |
| D12 | `lookback_k = 20` | |
| D13 | Labels baked at generation | ● |
| D14 | Failed hack → clean | ● |
| D15 | Structural access filter | |
| D16 | Array-based trajectories | ○ |
| D17 | Visitation L2 features | ● |
| D18 | L1/L2 only; L3/L4 declared | ● |
| D19 | Streaming detector interface | ○ |
| D20 | Seeded random baseline | |
| D21 | Adaptive level-aware CUSUM | |
| D22 | Pooled + per-trajectory metrics | |
| D23 | Tier-weighted RHOB-Score + CIs | ● |
| D24 | Normalized latency semantics | |
| D25 | HDF5 float64 exact round-trip | |
| D26 | Single `EvaluationConfig` | |
| D27 | Pre-scoring contract validation | |
| D28 | Minimal env registry + TIER1 fallback | ● |
| D29 | Consistency-based onset test | |
| D30 | MIT license; docs consolidation | |

● tracked in REFACTOR_PLAN · ○ cosmetic
