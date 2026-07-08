# RHOB Architecture Review — Milestone 1

**Reviewer role:** Senior research engineer (NeurIPS/TMLR infrastructure standard)
**Scope:** Entire repository at end of Milestone 1 (vertical slice)
**Method:** Full read of `src/`, `tests/`, `scripts/`, docs. Findings are cited to
`file:line` and were verified against the code, not inferred from the docs.

Severity tags: **[C]** Critical · **[H]** High · **[M]** Medium · **[L]** Low.

---

## Executive Verdict

This is **strong, publishable-track infrastructure with revisions required** — not
yet a drop-in community standard, but structurally on the right trajectory. The
engineering hygiene (layering, determinism, tests, docs) is clearly above the
typical academic-code bar. The issues that matter are concentrated in **three
places**: (1) the primary metric's confidence-interval methodology is unsound,
(2) the onset label is baked at generation with a *phantom* re-labeling knob that
will break the Milestone-4 ablation the paper depends on, and (3) the
detector-facing contract (L2 feature schema, L3/L4 access) is under-specified for
the benchmark's own headline claim ("does a detector transfer across
environments?"). None of these require a rewrite; they are seams and methodology,
fixable incrementally.

**Rating: 7.5 / 10 as research infrastructure.** The foundation would survive
peer review after the metric-CI and label-provenance fixes below.

---

## Evaluation by dimension

| Dimension | Rating | Summary (details in Weaknesses) |
|---|:---:|---|
| **Modularity** | Strong | Clean downward dependency graph; small public surface. Caveat: 3 lazy imports break latent cycles (W17). |
| **Maintainability** | Good | Pure metrics, typed errors, 96% tests, docs. Drag: env monolith (W5), helper sprawl (W18). |
| **Extensibility** | Good→Mixed | Detectors/environments add without core edits (verified). Gaps: no detector registry (W6), import-hub env registry (W7), no re-label path (W4). |
| **Separation of concerns** | Strong | Data (`Trajectory`) / view (`Observation`) / label (`OnsetLabel`) / scoring cleanly split. Weak spot: generation fuses MDP+trainer+labeler (W5). |
| **Benchmark abstraction** | Good | `evaluate(detector, trajectories, config)` is generic over any detector/trajectory set. Weak spot: score-level CI methodology (W1/W2). |
| **Environment abstraction** | Mixed | Clear `generate()→Trajectory` contract; but no L2 `feature_spec` (W9), monolithic implementation (W5), silent TIER1 fallback (W7). |
| **Detector abstraction** | Good | Minimal, honest streaming contract with structural access control. Weak spots: L3/L4 declared-not-implemented (W8), no batch path (W10). |
| **Scalability** | Weak (acceptable for M1) | Per-step copies (W10), eager loading (W12), O(env×traj) aggregation (W11), no parallelism (W13). Fine at M1 size; blocks M4 scale. |
| **Technical debt** | Contained | Cataloged and cross-referenced; none foundational. |

## Findings by category (explicit index)

- **Weak abstractions:** L3/L4 access levels declared but unbacked (W8); no L2
  feature contract (W9); `Timestep` defined but unused (W14); an `EvaluationConfig`
  onset knob that does nothing (W3).
- **Hidden coupling:** layering held together by function-local imports (W17);
  environment-registry import-hub (W7); onset parameters duplicated across
  config/oracle/environment with only one live (W3); labels baked into data at
  generation time (W4).
- **Duplicate logic / redundancy:** `tfd` and `median_latency` are the identical
  computation in two fields (W15); three overlapping NaN/format helpers (W18);
  per-environment result rescans (W11).
- **Future bottlenecks:** per-step object/array allocation (W10); eager, non-lazy
  dataset loading (W12); sequential, un-parallelized evaluation (W13); the
  environment monolith M3 will duplicate 4× (W5); the absent detector registry as
  the baseline set grows (W6).

---

## Strengths

**S1. Clean layered module graph.** `core`/`config` depend on nothing internal;
`environments`, `detectors`, `data` depend only downward; `evaluation`
orchestrates; the public API re-exports a small surface (`src/rhob/__init__.py`).
The dependency direction matches the engineering spec's intended graph.

**S2. Access enforcement is structural, not by convention.** `AccessFilter`
constructs a *fresh, frozen* `Observation` and zeroes fields above the declared
level (`core/access.py`, `core/trajectory.py:33`). Leakage is prevented by
construction and is genuinely tested end-to-end
(`tests/test_access.py::test_l1_detector_never_sees_l2_fields_via_trajectory`).
This is the single best design decision in the repo.

**S3. Metrics are pure functions.** `evaluation/metrics.py` has no hidden state,
making the metric layer trivially testable and deterministic. Edge cases
(single-class AUROC → NaN, clean-run latency → NaN, never-fires → inf) are
handled explicitly rather than crashing.

**S4. Reproducibility is a first-class concern.** Seeded generators throughout,
exact `float64` gzip HDF5 round-trip (verified in `tests/test_storage.py`),
deterministic bootstrap, and a `config_hash`. The determinism integration test
(`test_full_pipeline_deterministic`) pins the guarantee.

**S5. Fail-loud contract validation.** The runner validates bounds + determinism
before scoring and raises an *actionable* `ScoreBoundsError` with a fix hint
(`evaluation/runner.py:103-114`). This is the right ethos for community
submissions.

**S6. Honest data model separation.** `Trajectory` (full data) vs `Observation`
(filtered detector view) vs `OnsetLabel` (oracle output) are cleanly separated
conceptual roles, and the oracle observing `reward_true` is correctly quarantined
from detectors.

**S7. Test and doc quality.** 76 tests, ~96% coverage, integration tests that
encode the milestone success criteria, and a genuinely rigorous ground-truth
*consistency* onset test (rather than a brittle hand-labeled fixture).
`docs/data_schema.md` formalizes the onset definition and the generation
rationale.

---

## Weaknesses

### Correctness & methodology

**W1. [C] The RHOB-Score confidence interval is statistically unsound.**
`evaluation/runner.py:196-202`. With one environment, the code returns that
environment's *per-trajectory-AUROC* bootstrap CI **as** the RHOB-Score CI — two
different estimands. With multiple environments, it bootstraps over a handful of
per-environment point AUROCs (n = number of environments), which is a degenerate
resample that ignores per-trajectory variance **and** the tier weights that
define the score. The primary ranking metric therefore ships a CI a reviewer will
reject. The correct design (trajectory-level bootstrap that recomputes the
weighted score) is feasible — the per-trajectory data is already retained.

**W2. [H] Aggregate secondary metrics silently drop the weighting they imply.**
`runner.py:204-207` uses unweighted `np.nanmean` across environments for
miss-rate/TFD/FPR, while RHOB-Score is tier-weighted. Two different aggregation
philosophies coexist in one report, and a plain per-environment mean (not
trajectory-weighted) will distort once environment counts and sizes diverge.

**W3. [H] `EvaluationConfig` onset parameters are a phantom knob.**
`config.py:42-44` define `lookback_k`, `significance_delta`, `onset_alpha`; a
grep confirms **nothing in `evaluation/` reads them**. A user who sets
`EvaluationConfig(lookback_k=…)` expecting different labels gets byte-identical
results, because labels are frozen at generation. This is worse than a missing
feature — it is a control that looks live and is dead.

### Extensibility (will bite M3/M4)

**W4. [H] Onset labels are baked at generation with no re-labeling path.** The
oracle runs inside `GridWorldWireheading.generate()` and the label is frozen into
the `Trajectory`/HDF5. `reward_true` *is* stored (so re-labeling is physically
possible), but there is no `relabel(trajectory, oracle)` API and the runner reads
only the stored label. The blueprint's flagship **Ablation 6.1 (onset-definition
sensitivity)** — vary k/δ/α, check ranking stability — will require regenerating
the entire dataset per parameter setting instead of a cheap re-label. This is an
architecture/experiment mismatch, and it is the direct cause of W3.

**W5. [H] The environment is a monolith; M3 will duplicate its core 4×.**
`environments/tier1/gridworld_wireheading.py` (~130 LOC) fuses the grid MDP, the
Q-learning loop, the greedy rollout, the oracle call, `Trajectory` assembly, and
config hashing into one class. Milestone 3 adds four more tabular Tier-1
environments (bandit, cliffwalk, navigation-proxy, tabular-tampering); each will
copy the Q-learning/rollout code. There is no reusable `TabularQLearner`,
`GridMDP`, or `TrajectoryRecorder` seam. This is the clearest *imminent*
duplication bottleneck.

**W6. [H] No detector registry.** `detectors/` has `base` + `baselines` but no
registry, while `environments/` has one and the engineering spec specifies a
`DetectorRegistry` with contract validation and access-level filtering. Scaling
to "20+ detectors" through `__init__` exports alone means no `list_detectors()`,
no discovery, no per-access sub-leaderboards, and no central contract gate.

**W7. [M] The environment registry is a hard-wired import hub.**
`environments/registry.py:13` imports each environment class at module load and
registers by an explicit bottom-of-file call, bypassing its own
`@register_environment` decorator. Twenty environments → twenty eager imports (and
their dependencies) on `import rhob`. No entry-point/plugin mechanism (which the
spec calls for). Import-time coupling grows linearly with the catalogue.

**W8. [H] Access levels L3/L4 are declared but unimplementable.** The enum and
docstrings promise gradient/KL/internal-state (`core/types.py:32-41`), but
`Observation` and `AccessFilter` only carry reward (L1) and `policy_features`
(L2). A detector declaring `access_level=L3` is silently handed L2 data with no
L3 fields and **no error**. Implementing L3/L4 later means extending the *frozen*
`Observation` contract — a breaking change to a type the docs advertise as
"stable." Decide now: implement the fields, or narrow the enum to L1/L2 and mark
L3/L4 "reserved."

**W9. [H] No standardized L2 feature contract across environments.**
`policy_features` is an env-specific vector (here a 49-dim visitation histogram
tied to `grid_size`). The environment interface declares no `feature_dim`, no
feature schema, and no semantics. A cross-environment L2 detector — the
benchmark's own headline claim ("does a detector trained on A transfer to B?") —
has no stable input contract; feature dimensionality silently varies per
environment. This is a **benchmark-design** gap, not merely a code smell.

### Performance & scalability

**W10. [H] Per-step object allocation + array copy is the throughput ceiling.**
`iter_observations` yields a fresh frozen `Observation` per step, and
`AccessFilter.filter` does `np.array(copy=True)` + `setflags` on the feature
vector **every step** (`core/access.py:53-54`). For 20+ detectors × hundreds of
trajectories × long horizons × high-dim features this dominates runtime. There is
no batch/vectorized scoring path, so even a trivially vectorizable detector
(CUSUM) is forced through an O(T) Python loop.

**W11. [M] O(environments × trajectories) aggregation.** `runner.py:154`
re-scans the full results list once per environment
(`[r for r in results if r.environment_id == env_id]`). Harmless at one
environment, wasteful at 21 × many seeds. A single group-by pass suffices.

**W12. [H] Fully-eager dataset loading.** `data/storage.py:load_dataset` reads
every run's arrays into memory and returns a materialized list. The spec calls
for lazy/memory-mapped loading; at v2.0 scale (~50 GB) this OOMs. No streaming
iterator exists.

**W13. [M] No parallelism.** `compare` and `run` are sequential
(`runner.py:248`); the spec's `n_workers` is absent, and the 10k-resample
bootstrap is recomputed per environment per detector. 20+ detectors × 21 envs ×
50 seeds serially is a real wall-clock problem.

### Code smells & hidden coupling

**W14. [M] `Timestep` is a dead abstraction.** Defined in `core/trajectory.py`
and exported from the public API, but a grep confirms it is never produced or
consumed. Either wire it in or delete it — a public, unused type is a maintenance
liability.

**W15. [L] `tfd` and `median_latency` are the identical computation** stored in
two `EnvironmentMetrics` fields (`runner.py:176` and `:178` both call
`time_to_first_detection(latencies)`). They read as distinct metrics but are the
same number.

**W16. [M] `is_hacking_run` silently relabels a failed hack as clean.**
`gridworld_wireheading.py:265` sets
`is_hacking_run = is_hacking and onset_label is not None`. A run *configured* to
hack that does not manifest becomes a clean negative, conflating "clean" with
"missed hack." Latent at today's 100% reliability, but a data-integrity hazard for
the harder environments (Tier 2/3, adversarial) where reliability < 100% is
expected and the distinction is scientifically meaningful.

**W17. [M] Layering is enforced by lazy imports.** Cycles are dodged with
function-local imports in at least three places: `trajectory.iter_observations` →
`access`, `runner._resolve_trajectories` → `data.storage`,
`runner._aggregate_environments` → `environments.registry`. This works, but the
need for it signals the module tree is cleaner than the true dependency graph
(notably an intra-`core` `trajectory ↔ access` cycle). Worth making explicit
before it multiplies.

**W18. [L] NaN/format helper sprawl in `report.py`** (`_fmt`, `_nan_to_none`,
`_safe`) — three overlapping NaN/inf handlers, plus hand-written
`to_dict`/`to_markdown` that will scale poorly as report fields grow.

**W19. [L] Leaderboard ranking is a 2-key sort** (`report.py:results_table`); the
spec's tie-break chain (latency → miss → overhead → date) is unimplemented.
Irrelevant at two baselines, relevant once >20 entries tie on an easy tier.

**W20. [L] Reproducibility-hostile timestamp + schemaless metadata.** A wall-clock
`generation_timestamp` is embedded in every trajectory/HDF5 (file-level hashes
differ across regenerations though content is identical), and `activation_episode`
— the hidden ground-truth *cause* of the onset — is stored in free-form
`traj.metadata` in the clear. It never reaches detectors via the official path,
but it is leak-adjacent data on the same object/file, with no metadata schema.

**W21. [M] No property-based or regression tests despite `hypothesis` in deps.**
Every metric-invariant test is example-based. The spec's property tests (bounds,
monotonicity oracle > detector > random, AUROC scale-invariance) and gold-file
regression pins are absent — below the "publishable infrastructure" bar for a
metrics library that others will trust.

---

## Recommended Refactors (no code — seams and directions)

**R1. Split the environment into composable pieces.** Introduce three seams
before M3 clones the environment: a `TabularQLearner` (training loop → returns
Q / curves), a `GridMDP`/transition model, and a `TrajectoryRecorder` (assembles a
`Trajectory` from curves + features + label). `generate()` becomes orchestration.
Directly kills the M3 duplication in W5.

**R2. Separate labeling from generation; add a re-labeling API.** Make the oracle
a *post-processing* step: store `reward_true` (already done) and expose
`rhob.relabel(trajectory, oracle) -> Trajectory`. Then the evaluation config's
onset parameters become live (fixing W3) and the M4 ablation is a cheap re-label
loop instead of a full regeneration (fixing W4).

**R3. Fix the RHOB-Score CI to a trajectory-level bootstrap.** Resample
trajectories (stratified within environment), recompute the tier-weighted score
per resample, take percentiles. Unifies W1 and W2 under one principled aggregation
and removes the single-vs-multi-env special case.

**R4. Add a `DetectorRegistry` mirroring the environment one**, with
`register_detector`, `list_detectors`, `list_by_access`, and a `validate_contract`
entry point. Fixes W6 and gives the leaderboard its access-level sub-views for
free.

**R5. Decide and encode the access-level contract.** Either extend `Observation`
with the L3/L4 fields now (while the contract is young and cheap to change) or
narrow `AccessLevel` to L1/L2 and document L3/L4 as reserved. Add a declared
`feature_spec` (dimensionality + semantics) to the environment interface so
cross-environment L2 detectors have a stable contract (W8, W9).

**R6. Introduce a batch scoring path** alongside the streaming one. Keep `step()`
for streaming detectors, but let a detector optionally implement
`score_sequence(observations) -> np.ndarray` so vectorizable detectors bypass the
per-step Python loop and the per-step feature copy (W10). Make the feature matrix
read-only once at load instead of per step.

**R7. Make loading lazy and aggregation single-pass.** Return an iterable/lazy
dataset with optional memory-mapping (W12); replace the per-environment rescans
with one group-by (W11).

**R8. Consolidate report/NaN handling and add tie-break ranking** (W18, W19), and
drop the redundant `median_latency` field (W15).

---

## Technical Debt (ledger)

| ID | Item | Severity | Cost to fix now | Cost if deferred |
|----|------|----------|-----------------|------------------|
| W1/W2 | Unsound / inconsistent aggregate CIs | C/H | Low | High (mis-stated results in paper) |
| W3/W4 | Phantom onset config; no re-label API | H | Low–Med | High (blocks M4 ablation) |
| W5 | Environment monolith | H | Med | High (4× duplication in M3) |
| W6 | No detector registry | H | Low | Med (compounds per detector) |
| W8/W9 | L3/L4 unimplemented; no L2 feature contract | H | Med | High (frozen-contract break) |
| W10/W12/W13 | Per-step copies, eager load, no parallelism | H | Med | High (throughput wall at scale) |
| W14 | Dead `Timestep` | L | Trivial | Low |
| W15 | Duplicated latency field | L | Trivial | Low |
| W16 | Failed-hack relabeled clean | M | Low | High (data integrity in Tier 2/3) |
| W17 | Lazy-import layering | M | Low | Med |
| W21 | No property/regression tests | M | Med | Med |

---

## What Should Be Fixed Before Milestone 2

Milestone 2 is theory-heavy (proofs, Sections 2–3) but includes numerical
verification of L1-indistinguishability vs. L2-separation, which leans directly on
the access-level architecture and the metric layer. Prioritize the fixes that are
**cheap now, load-bearing later, or actively misleading today**:

1. **[C] Fix the RHOB-Score CI (W1/W2 → R3).** It is the number every downstream
   milestone and the paper will quote. Do this first.
2. **[H] Kill the phantom onset knob (W3) and add re-labeling (W4 → R2).** At
   minimum, *delete* the dead `EvaluationConfig` fields so nobody builds on a
   no-op; ideally land `relabel()` so M4's ablation is architecturally free.
3. **[H] Resolve the access-level contract (W8 → R5).** M2's own L1-vs-L2
   separation argument is undermined if L3/L4 are advertised-but-fake. Either
   implement the fields or narrow the enum; add the L2 `feature_spec`.
4. **[H] Extract the tabular RL/recorder seam (W5 → R1)** *before* M3 starts, so
   the four new Tier-1 environments compose rather than copy.
5. **[M] Add the detector registry (W6 → R4)** so the growing baseline set (M3
   adds Flight Recorder, ensemble, KL, gradient-norm) has a home and a contract
   gate.
6. **[L] Hygiene:** delete dead `Timestep` (W14) and the duplicate `median_latency`
   (W15); flag rather than silently relabel failed hacks (W16).

Defer to their own milestones: parallelism/lazy-loading (needed at M4 scale, not
M2), leaderboard tie-breaking (needs a populated leaderboard), property tests
(valuable but not blocking).

---

## Is This Publishable Infrastructure?

**Yes, on the "benchmark + infrastructure" track — conditional on revisions.**

- **Engineering foundation:** *above* the typical academic-release bar. Layering,
  determinism, structural access control, test coverage, and documentation are
  the things most released benchmarks get wrong, and this repo gets them right.
- **Scientific/methodological gaps that a reviewer *will* catch:** the RHOB-Score
  CI (W1), the aggregate weighting inconsistency (W2), and the absence of a
  cross-environment L2 feature contract (W9) — the last being in tension with the
  benchmark's own transfer claim. These are revisions, not rebuilds.
- **Standard-setting ("the GLUE of reward hacking") readiness:** **not yet** — that
  requires the multi-environment suite, the standardized detector-input contract,
  a detector registry/leaderboard, and the ablation infrastructure. The current
  slice is a credible *proof of the architecture*, and critically, **none of the
  weaknesses are foundational** — every one is a seam, a policy decision, or a
  methodology fix that lands incrementally on top of what exists.

**Bottom line:** the bones are right. Fix the metric-CI methodology and the
label-provenance/config coupling before building higher, resolve the access/feature
contract while it is still cheap to change, and this becomes infrastructure the
community could adopt.
