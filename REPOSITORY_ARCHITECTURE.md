# RHOB Repository Architecture

**Document:** REPOSITORY_ARCHITECTURE · **Spec version:** 0.1 · **Status:** Reference
Describes the repository as it exists at end of Milestone 1. Source of truth: the
tree under `src/rhob/`. No code is modified by this document.

---

## 1. Folder structure

```
Reward_hacking_benchmark/
├── src/rhob/
│   ├── __init__.py              # public API surface
│   ├── _version.py              # single source of version truth
│   ├── config.py                # EvaluationConfig (pydantic)
│   ├── core/                    # data models + type system (depends on nothing internal)
│   │   ├── types.py             # AccessLevel, HackingType, Tier, TIER_WEIGHTS, ids
│   │   ├── trajectory.py        # Trajectory, Timestep, Observation
│   │   ├── onset.py             # OnsetLabel
│   │   ├── access.py            # AccessFilter (structural leak prevention)
│   │   └── exceptions.py        # RHOBError hierarchy
│   ├── environments/
│   │   ├── base.py              # AbstractEnvironment, EnvironmentCard, ValidationReport
│   │   ├── oracle.py            # OnsetOracle (labeler)
│   │   ├── registry.py          # register/list/get + tier lookup
│   │   └── tier1/gridworld_wireheading.py
│   ├── detectors/
│   │   ├── base.py              # AbstractDetector, OverheadEstimate
│   │   └── baselines/{random_detector,cusum}.py
│   ├── evaluation/
│   │   ├── metrics.py           # pure metric functions
│   │   ├── runner.py            # EvaluationRunner, evaluate(), compare()
│   │   └── report.py            # EvaluationReport, EnvironmentMetrics, results_table
│   └── data/storage.py          # HDF5 save/load
├── scripts/generate_gridworld_data.py
├── examples/minimal_evaluation.py
├── tests/                        # 76 tests, ~96% coverage
├── docs/data_schema.md
├── data/                         # generated datasets (gitignored)
└── <spec docs>.md                # this suite
```

Absent by design in M1 (specified in the engineering spec, deferred): `cli/`,
`leaderboard/`, `integrations/`, `visualization/`, a `config/` package, and the
Tier 2/3/adversarial environment directories.

## 2. Module responsibilities

| Module | Responsibility | Stability |
|---|---|---|
| `core.types` | Enums, ids, tier weights | Stable |
| `core.trajectory` | Data model + access-filtered `Observation` view | Stable |
| `core.onset` | Ground-truth label model + (de)serialization | Stable |
| `core.access` | Structural access enforcement | Stable |
| `core.exceptions` | Typed error hierarchy | Stable |
| `config` | Validated evaluation configuration | Stable (subset) |
| `environments.base` | Environment contract, card, self-validation | Stable |
| `environments.oracle` | Onset labelling from curves | Stable |
| `environments.registry` | Discovery / instantiation / tier lookup | Provisional |
| `environments.tier1.*` | Concrete environments (data generators) | Growing |
| `detectors.base` | Detector contract | Stable |
| `detectors.baselines.*` | Reference detectors | Growing |
| `evaluation.metrics` | Pure metric functions | Stable |
| `evaluation.runner` | Orchestration, contract validation, aggregation | Provisional |
| `evaluation.report` | Report structures + rendering | Provisional |
| `data.storage` | HDF5 persistence | Stable (format 1.0) |

## 3. Dependency graph

```
                 ┌──────────────┐
                 │  evaluation  │  (runner, metrics, report)
                 └──┬───┬───┬───┘
        ┌───────────┘   │   └───────────┐
        ▼               ▼               ▼
 ┌────────────┐  ┌────────────┐  ┌────────────┐
 │environments│  │ detectors  │  │    data    │
 └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
       └──────────┬────┴───────────────┘
                  ▼
          ┌───────────────┐   ┌──────────┐
          │     core      │   │  config  │
          └───────────────┘   └──────────┘
```

**Rule:** dependencies flow downward only. `core` and `config` depend on nothing
internal. Verified by import inspection.

**Caveat (hidden coupling):** the clean layering is partly maintained by three
*function-local* imports that break latent cycles — `trajectory → access`,
`runner → data.storage`, `runner → environments.registry`. These work but signal
the true graph is slightly more entangled than the tree suggests
(ARCHITECTURE_REVIEW W17).

## 4. Public API

Everything a typical user needs is importable from `rhob` (see `__init__.__all__`):

```python
# evaluation
rhob.evaluate(detector, trajectories, config=None) -> EvaluationReport
rhob.compare([detectors], trajectories, config=None) -> [EvaluationReport]
rhob.EvaluationRunner, rhob.EvaluationReport, rhob.results_table

# base classes / extension
rhob.Detector (AbstractDetector),  rhob.Environment (AbstractEnvironment)
rhob.register_environment, rhob.get_environment, rhob.get_environment_card,
rhob.list_environments

# baselines / environments
rhob.RandomDetector, rhob.CUSUMDetector, rhob.GridWorldWireheading

# core types / data model
rhob.AccessLevel, rhob.HackingType, rhob.Tier
rhob.Trajectory, rhob.Timestep, rhob.Observation, rhob.OnsetLabel, rhob.AccessFilter

# data / config / metrics
rhob.load_dataset, rhob.save_dataset, rhob.load_trajectory, rhob.save_trajectory
rhob.EvaluationConfig, rhob.default_config, rhob.metrics

# errors
rhob.RHOBError, rhob.DetectorError, rhob.ContractViolationError, rhob.ScoreBoundsError
```

**API stability intent** (to freeze at 1.0): top-level functions, `core` types, the
detector/environment contracts, and the report structure are **Stable**;
`visualization`/`integrations`/`cli` will be **Unstable** when added.

## 5. Internal APIs (not part of the public contract)

- `EvaluationRunner._run_one / _validate_contract / _check_bounds /
  _trajectory_result / _aggregate_environments / _aggregate_overall` — pipeline
  internals; may change without notice.
- `AccessFilter.filter` — used by `Trajectory.iter_observations`.
- `OnsetOracle._effective_k / _significant / _confidence / _severity` — labelling
  internals.
- `storage._read_run / _write_run / _jsonable`, `runner._resolve_trajectories`,
  `report._fmt / _nan_to_none / _safe`.

These are intentionally private; external code should depend only on §4.

## 6. Extension points

| To add… | Mechanism | Touches core? |
|---|---|:---:|
| A detector | subclass `rhob.Detector`, pass to `evaluate` | **No** |
| An environment | subclass `rhob.Environment`, `register_environment(cls)` | **No** |
| A dataset | write HDF5 per `DATASET_SPEC`, `load_dataset` | **No** |
| A baseline (shipped) | add under `detectors/baselines/`, export | yes (intended) |
| A metric | add to `evaluation.metrics` | yes — a metric registry is **[Planned]** |
| A detector registry | **[Planned]** (`register_detector`) | — |
| Third-party plugins | **[Planned]** entry-points | — |

Verified: a custom detector **and** a custom environment can be added and
evaluated end-to-end from outside `src/`, modifying zero core files.

## 7. Areas requiring future refactoring

Cross-referenced to `ARCHITECTURE_REVIEW.md` (W#) and `REFACTOR_PLAN.md` (#):

| Area | Issue | Ref |
|---|---|---|
| Score aggregation | RHOB-Score CI unsound; unweighted secondary aggregates | W1/W2 · #1 |
| Label provenance | labels baked at generation; `EvaluationConfig` onset params inert; no `relabel()` | W3/W4 · #2 |
| Access contract | L3/L4 declared but unimplemented; no L2 `feature_spec` | W8/W9 · #3 |
| Environment monolith | MDP+trainer+rollout+labeling fused → M3 duplication | W5 · #4 |
| Detector registry | absent | W6 · #5 |
| Data integrity | failed-hack silently relabeled clean | W16 · #6 |
| Performance | per-step object/array copy; no batch path | W10 · #8 |
| Scale | eager loading; O(env×traj) aggregation; no parallelism | W11–13 · #9 |
| Layering | cycles broken by function-local imports | W17 |
| Registry coupling | import-hub, no entry-points; silent TIER1 fallback | W7 · #10 |
| Hygiene | dead `Timestep`; duplicate `median_latency` field | W14/W15 |

None are foundational; each is a seam, a policy decision, or a methodology fix.

## 8. Test & tooling architecture

- `tests/` — unit (`test_types/access/trajectory/onset/metrics/storage/detectors`),
  environment, integration (encodes M1 success criteria), and public-API tests;
  session-scoped fixtures generate data once. Property and regression/gold-file
  tests are **[Planned]** (`tests/property/`, `tests/regression/`).
- `scripts/generate_gridworld_data.py` — dataset generation CLI (click).
- `examples/minimal_evaluation.py` — runnable 10-line evaluation.
- `pyproject.toml` — src-layout, `pythonpath=["src"]`, ruff/mypy config; lint clean.
