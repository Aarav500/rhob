# RHOB Refactoring Roadmap

Derived from [`ARCHITECTURE_REVIEW.md`](ARCHITECTURE_REVIEW.md). This plan ranks
only the refactors that **materially improve the benchmark** — its scientific
validity, extensibility, or ability to scale to the full environment/detector
suite. Cosmetic and hygiene-only items are explicitly excluded (see the last
section).

**No code is changed by this document.** It is a sequencing and prioritization
plan.

---

## Ranking methodology

Every recommendation is scored on three axes (1–5):

- **Importance** — how much project success depends on it (5 = the benchmark is
  compromised without it).
- **Cost** — implementation effort (1 = a few hours, 5 = a week+). *Lower is
  cheaper.*
- **Research impact** — how directly it affects the paper's claims, the validity
  of reported numbers, or a headline experiment (5 = a reviewer would reject
  without it).

**Priority = Importance + Research-impact − ⌊Cost/2⌋**, then bucketed:

- **P0** — correctness/provenance. Do before building anything on top; otherwise
  results are wrong or misleading.
- **P1** — extensibility seams. Do before the environment/detector suite grows
  (M3), and to support the paper's transfer claim / M2 theory link.
- **P2** — scale. Do before M4 runs 9+ envs × 50 seeds × 6+ detectors.
- **Deferred** — real but not yet load-bearing.

---

## Master ranked table

| # | Refactor | Addresses | Import. | Cost | Res. impact | Priority | Unblocks |
|---|----------|-----------|:-------:|:----:|:-----------:|:--------:|----------|
| 1 | Trajectory-level RHOB-Score CI + consistent weighted aggregation | W1, W2, R3, nanmean bug | 5 | 2 | 5 | **P0** | every reported number |
| 2 | Separate labeling from generation; `relabel()` API; live onset config | W3, W4, R2 | 5 | 3 | 5 | **P0** | M4 onset ablation |
| 3 | Access-level + L2 feature contract (`feature_spec`; L3/L4 decision) | W8, W9, R5 | 5 | 3 | 5 | **P1** | transfer claim, M2 L1/L2 |
| 4 | Decompose environment into `MDP` / `TabularQLearner` / `Recorder` | W5, W17, R1 | 4 | 3 | 3 | **P1** | M3 (5 Tier-1 envs) |
| 5 | Detector registry + contract gate | W6, R4 | 4 | 2 | 3 | **P1** | 20+ detectors, sub-boards |
| 6 | Distinguish "missed hack" from "clean" (stop silent relabeling) | W16 | 4 | 1 | 4 | **P1** | Tier 2/3 integrity |
| 7 | Property + regression (gold-file) tests for metrics/invariants | W21 | 4 | 3 | 4 | **P1** | reviewer trust |
| 8 | Batch scoring path; eliminate per-step object/array copy | W10, R6 | 3 | 3 | 2 | **P2** | compute budget |
| 9 | Lazy/streamed loading + single-pass aggregation + parallel workers | W11, W12, W13, R7 | 3 | 4 | 2 | **P2** | M4/v2.0 scale |
| 10 | Environment plugin via entry-points (drop import-hub) | W7 | 2 | 3 | 1 | **Deferred** | external contributors |
| 11 | Leaderboard multi-key tie-break ranking | W19 | 2 | 1 | 2 | **Deferred** | populated leaderboard |

---

## P0 — Correctness & provenance (do first)

### 1. Trajectory-level RHOB-Score CI + consistent weighted aggregation
**Addresses:** W1, W2, R3, and the `Mean of empty slice` warning found in
verification (`runner.py:206`).
**Importance 5 · Cost 2 · Research impact 5.**

**Problem.** The primary metric's confidence interval is unsound: the single-env
path returns a per-*trajectory* AUROC CI as the score CI; the multi-env path
bootstraps over a handful of per-env points, ignoring trajectory variance and
tier weights (`runner.py:196–202`). Secondary aggregates use unweighted
`nanmean` (`:204–207`), inconsistent with the tier-weighted score, and warn on
all-NaN slices.

**Change (files):** `evaluation/runner.py`, `evaluation/metrics.py`.
Introduce one bootstrap that resamples **trajectories** (stratified within
environment), recomputes the tier-weighted RHOB-Score per resample, and takes
percentiles. Route all aggregate metrics through the same weighted, NaN-safe
reducer. Remove the single-vs-multi-env special case.

**Acceptance:** CI is a genuine interval on the score estimand; identical
aggregation path for 1 and N environments; no `RuntimeWarning`; a regression
test pins the corrected numbers.

**Risk:** Low. Self-contained in the evaluation layer; reported numbers will
shift slightly (expected and correct).

---

### 2. Separate labeling from generation; add `relabel()`; make onset config live
**Addresses:** W3, W4, R2.
**Importance 5 · Cost 3 · Research impact 5.**

**Problem.** The oracle runs inside `generate()` and the label is frozen into the
`Trajectory`/HDF5. `EvaluationConfig.lookback_k/significance_delta/onset_alpha`
are defined but read nowhere — a phantom knob. The M4 onset-sensitivity ablation
(vary k/δ/α, check ranking stability) would therefore require regenerating the
whole dataset per setting.

**Change (files):** `environments/oracle.py`, `environments/tier1/…`,
`evaluation/`, `config.py`, `data/storage.py`.
Make labeling a post-processing pass over stored `reward_true` (already
persisted): expose `rhob.relabel(trajectory, oracle) -> Trajectory` and let the
runner (optionally) re-label from the config's onset parameters before scoring.
Either wire the `EvaluationConfig` onset fields into that path or delete them —
no dead knob.

**Acceptance:** changing onset parameters changes labels/metrics without
regeneration; an M4-style ablation runs as a cheap re-label loop; `reward_true`
provenance documented.

**Risk:** Medium. Touches the generation→storage→evaluation seam; guard against
double-labeling. Keep the generation-time label as a cached default.

---

## P1 — Extensibility seams (before M3 / to support the transfer claim)

### 3. Access-level and L2 feature contract
**Addresses:** W8, W9, R5.
**Importance 5 · Cost 3 · Research impact 5.**

**Problem.** L3/L4 are advertised in the enum/docs but unimplementable —
`Observation`/`AccessFilter` carry only reward (L1) and `policy_features` (L2),
so declaring L3 silently yields L2 data with no error. And `policy_features` has
no declared schema/dimensionality, so a cross-environment L2 detector — the
benchmark's own transfer claim — has no stable input contract.

**Change (files):** `core/types.py`, `core/trajectory.py`, `core/access.py`,
`environments/base.py`. Add a declared `feature_spec` (dimensionality + named
semantics) to the environment contract, surfaced on `Observation`. **Decide
L3/L4 now:** either add the gradient/KL/internal-state fields to `Observation`
while the contract is young, or narrow the enum to L1/L2 and mark L3/L4
"reserved" with an explicit error if declared.

**Acceptance:** an L3 declaration either works or raises (never silently
degrades); every environment publishes a `feature_spec`; a cross-env L2 detector
can introspect feature semantics.

**Risk:** Medium. This is a *contract* change — cheaper now than after M3
environments and external detectors depend on the current shape. Sequence
**before** #4 so the extracted recorder bakes in `feature_spec`.

---

### 4. Decompose the environment (MDP / trainer / recorder)
**Addresses:** W5, W17, R1.
**Importance 4 · Cost 3 · Research impact 3.**

**Problem.** `gridworld_wireheading.py` fuses MDP, Q-learning, rollout, labeling,
assembly, and hashing (~130 LOC). M3's four new tabular envs will copy the
trainer/rollout. Layering is also propped up by function-local imports (W17).

**Change (files):** new `environments/tabular/{mdp,qlearner,recorder}.py` (or
similar); `gridworld_wireheading.py` becomes orchestration. Extract a reusable
`TabularQLearner` (curves + greedy rollout), a grid/transition model, and a
`TrajectoryRecorder` (assembles `Trajectory` from curves + features + label).

**Acceptance:** a second tabular environment can be written with no duplicated
training loop; `gridworld` behavior/numbers unchanged (pin via regression test
from #7).

**Risk:** Low–Medium. Pure refactor guarded by existing determinism tests; do it
**before** M3 starts, not after.

---

### 5. Detector registry + contract gate
**Addresses:** W6, R4.
**Importance 4 · Cost 2 · Research impact 3.**

**Problem.** Environments have a registry; detectors do not. Scaling to 20+
detectors through `__init__` exports gives no discovery, no per-access
sub-leaderboards, and no central contract validation.

**Change (files):** new `detectors/registry.py` mirroring the environment one
(`register_detector`, `list_detectors`, `list_by_access`, `validate_contract`);
register the baselines.

**Acceptance:** `rhob.list_detectors()` and access-level filtering work; contract
validation callable centrally; leaderboard can produce L1-only / L2-only views.

**Risk:** Low.

---

### 6. Distinguish "missed hack" from "clean"
**Addresses:** W16.
**Importance 4 · Cost 1 · Research impact 4.**

**Problem.** `is_hacking_run = is_hacking and onset_label is not None`
(`gridworld:265`) silently relabels a configured-hack that didn't manifest as a
clean negative — conflating two scientifically distinct populations. Latent at
today's 100% reliability, but a data-integrity hazard for Tier 2/3/adversarial
where reliability < 100% is expected and meaningful.

**Change (files):** `core/trajectory.py` (add an explicit run-outcome status,
e.g. `configured_hacking` vs `manifested`), `environments/*`, evaluation
filtering. Missed hacks should be countable and excludable, not silently merged
into clean negatives.

**Acceptance:** reports distinguish clean / hacked / missed-hack counts; miss-rate
math is unaffected for reliable envs; harder envs surface non-manifesting runs.

**Risk:** Low. Cheap now, expensive to retrofit once Tier 2/3 data exists.

---

### 7. Property + regression tests for the metric/invariant layer
**Addresses:** W21 (and would have caught the nanmean bug in #1).
**Importance 4 · Cost 3 · Research impact 4.**

**Problem.** All metric tests are example-based; the spec's invariants (bounds,
monotonicity `oracle > detector > random`, AUROC scale-invariance) and gold-file
regression pins are absent — below the trust bar for a metrics library others
will cite.

**Change (files):** `tests/property/…`, `tests/regression/…` using the
already-declared `hypothesis` dependency. Property tests for metric bounds,
determinism, and scale-invariance; a gold-file pin of the corrected M1 numbers
(couples with #1).

**Acceptance:** invariants hold under generated inputs; regression file fails if a
metric value drifts unexpectedly.

**Risk:** Low. Pure additive test work; best done alongside #1.

---

## P2 — Scale (before M4 / v2.0 data)

### 8. Batch scoring path; eliminate per-step allocation/copy
**Addresses:** W10, R6.
**Importance 3 · Cost 3 · Research impact 2.**

**Problem.** `iter_observations` allocates a frozen `Observation` per step and
`AccessFilter.filter` copies the feature vector every step (`access.py:53–54`).
No vectorized path exists; even CUSUM is forced through an O(T) Python loop.

**Change (files):** `detectors/base.py` (optional
`score_sequence(observations) -> np.ndarray`), `core/access.py` (make the feature
matrix read-only once, not per step), `evaluation/runner.py` (use the batch path
when available). Keep `step()` for streaming detectors.

**Acceptance:** vectorizable detectors bypass the per-step loop; streaming
semantics preserved; measurable throughput gain on long trajectories.

**Risk:** Low–Medium. Additive interface; must not weaken access enforcement.

---

### 9. Lazy loading + single-pass aggregation + parallel workers
**Addresses:** W11, W12, W13, R7.
**Importance 3 · Cost 4 · Research impact 2.**

**Problem.** `load_dataset` fully materializes every run (OOM at the spec's ~50 GB
v2.0); aggregation rescans results per environment (`runner.py:154`, O(envs ×
trajs)); `compare`/`run` are sequential and re-bootstrap per env per detector.

**Change (files):** `data/storage.py` (lazy/memory-mapped dataset iterator),
`evaluation/runner.py` (one group-by pass; optional `n_workers`).

**Acceptance:** a dataset larger than RAM evaluates; aggregation is single-pass;
detector×env evaluation parallelizes.

**Risk:** Medium. Concurrency + determinism must be preserved (seed the bootstrap
per worker).

---

## Deferred (real, but not yet load-bearing)

- **#10 Environment entry-point plugins (W7).** Replace the import-hub registry
  with entry-points so external packages contribute environments without editing
  core. Valuable for community scale; do once the internal suite stabilizes
  (post-M3).
- **#11 Leaderboard tie-break ranking (W19).** Multi-key sort (latency → miss →
  overhead → date). Implement when the leaderboard actually exists and has enough
  entries to tie.

---

## Explicitly excluded (cosmetic — per instructions)

These are not ranked; fix opportunistically only when already editing the file:

- Dead `Timestep` type (W14) — delete or wire in.
- Duplicate `median_latency`/`tfd` field (W15).
- NaN/format helper sprawl in `report.py` (W18).
- Wall-clock `generation_timestamp` in HDF5 (W20, reproducibility-of-file only).
- Schemaless `metadata` / `activation_episode` visibility (W20) — low risk since
  detectors never receive it; tighten if convenient.

---

## Recommended execution sequence

```
Phase A (P0, before any new features)      → #1, #2
Phase B (P1, before M3 environment suite)  → #3, then #4, #5, #6, #7
Phase C (P2, before M4 scale runs)         → #8, #9
Deferred                                   → #10, #11
```

**Rationale.** #1–#2 make the numbers correct and the onset ablation feasible;
everything downstream depends on them. #3 must precede #4 so the extracted
recorder bakes in the feature/access contract. #4–#6 remove the duplication and
integrity hazards *before* M3 multiplies environments. #7 locks trust in the
metric layer and pins the corrected numbers. #8–#9 buy the compute headroom M4
needs. #10–#11 wait until they have something to serve.

**Highest leverage per unit cost:** #6 (Cost 1, Research impact 4) and #1
(Cost 2, Research impact 5) — do these first.
