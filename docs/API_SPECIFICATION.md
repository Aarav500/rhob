# RHOB v3 API Specification (Frozen for v3.2+)

## Overview
This document formalizes the v3 API contracts. All interfaces are frozen as of v3.2 release. Community contributions must conform to these signatures. Breaking changes require a major version bump.

---

## 1. Environment Family Interface

### `BaseFamily` (Abstract)
**Location:** `rhob.v3.base_family.BaseFamily`

**Stability:** FROZEN (v3.2+)

**Methods:**

```python
@property
@abstractmethod
def name(self) -> str:
    """Unique family identifier (e.g., 'gridworld_camping', 'novelty_farming').
    
    Convention: snake_case, max 32 chars.
    Used in leaderboard, CLI, and results files.
    """

@property
@abstractmethod
def mechanism(self) -> HackingMechanism:
    """Enum from rhob.v3.taxonomy.HackingMechanism.
    
    Must be one of: CAMPING_EXPLOIT, REWARD_SHAPING, SHORTCUT, EXPLORATION_EXPLOIT,
    PROXY_GAMING, GOAL_MISGENERALIZATION, DECEPTIVE_ALIGNMENT, REWARD_TAMPERING.
    """

@property
@abstractmethod
def complexity(self) -> EnvironmentComplexity:
    """Enum from rhob.v3.taxonomy.EnvironmentComplexity.
    
    Must be one of: TABULAR, CONTINUOUS_SIMPLE, CONTINUOUS_COMPLEX, SEQUENTIAL, MULTI_AGENT.
    """

@abstractmethod
def difficulty_range(self) -> tuple[float, float]:
    """Return (min_L2, max_L2) target behavioral separability range.
    
    Constraints:
    - Both in (0.5, 1.0)
    - min_L2 < max_L2
    - Single-point families return (x, x)
    
    Example: (0.60, 0.98) means this family can generate pairs at any L2 in [0.60, 0.98].
    """

@abstractmethod
def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
    """Generate one matched-proxy pair at target difficulty.
    
    Args:
        difficulty: Target L2-AUROC in range returned by difficulty_range()
        seed: RNG seed for reproducibility (optional, default 0)
    
    Returns:
        MatchedPair with rollout_hacking and rollout_legit callables
    
    Constraints:
        - Pair.difficulty must equal the requested difficulty
        - Both rollout callables must be deterministic given seed
        - Rollouts must complete in <60s on standard hardware
    """

def default_difficulties(self) -> list[float]:
    """List of difficulties to use when caller requests 'all'.
    
    Default implementation:
    - Return named DifficultyTier values that fall within difficulty_range()
    - Fall back to (lo, hi) endpoints if no tiers match
    
    Override only if you need custom default sweep.
    """
```

---

## 2. Matched Pair Interface

### `MatchedPair` (Dataclass)
**Location:** `rhob.v3.base_pair.MatchedPair`

**Stability:** FROZEN (v3.2+)

**Schema:**

```python
@dataclass(frozen=True)
class MatchedPair:
    family: str                                      # family.name
    mechanism: HackingMechanism                      # From taxonomy
    complexity: EnvironmentComplexity                # From taxonomy
    difficulty: float                                # Target L2-AUROC
    n_episodes: int                                  # Episode length
    rollout_hacking: Callable[[int], RolloutResult] # seed -> (RunData, onset)
    rollout_legit: Callable[[int], RolloutResult]   # seed -> (RunData, onset)
    params: dict                                     # Family-specific params
```

**Rollout Semantics:**
- `rollout_hacking(seed)` returns (RunData for hacking variant, onset_episode_index)
- `rollout_legit(seed)` returns (RunData for legitimate variant, onset_episode_index)
- onset = -1 means no onset (legitimate never hacks)
- onset >= 0 means episode where hacking behavior commits (zero-indexed)

---

## 3. Run Data Schema

### `RunData` (Dataclass)
**Location:** `rhob.detectors.posthoc.RunData`

**Stability:** FROZEN (v3.2+)

**Schema:**

```python
@dataclass
class RunData:
    proxy_rewards: np.ndarray          # [n_episodes], reward observable by detector
    true_rewards: np.ndarray           # [n_episodes], oracle only, NEVER read by L0/L1/L2 detectors
    state_counts: Optional[np.ndarray] # [n_episodes, n_states], state visitation histogram (L1+)
    behav_trace: Optional[np.ndarray]  # [n_episodes], anti-symmetric behavioral feature (L2)
```

**`behav_trace`'s orientation is randomized.** A family emits it with hacking positive in
its own coordinate, and `MatchedPair.rollout` then multiplies *both* variants by a
per-`(family, layout_seed)` sign of ±1 before any detector sees it. The pair's
separation is preserved exactly; the direction is not disclosed. A detector may not
assume it — see [`docs/l2_sign_randomization.md`](l2_sign_randomization.md).

**Access Restrictions (enforced at evaluate time by `rhob.v3.access.restrict`):**
- L0: ONLY proxy_rewards
- L1: proxy_rewards + state_counts
- L2: proxy_rewards + state_counts + behav_trace
- L3: ALL fields, including true_rewards

L3 detectors **are** scored and do appear in the committed leaderboard; they are a
ceiling measurement, not a production detector. (An earlier revision of this
document said "never scored in production", which described intent rather than
behavior.)

### Optional channels are N/A, not 0.5

`state_counts` and `behav_trace` are `Optional`, and families legitimately return
`None` for them — 25 of the 33 shipped families emit no `state_counts` at all
(high-dimensional continuous state, multi-agent joint spaces, token vocabularies).

**Detector contract:** a detector whose channel is `None` returns a neutral
constant (`0.5` from `classify`, `-1` from `detect_onset`). This is correct and
unchanged.

**Harness contract (new):** a neutral constant is *not a measurement*, and the
evaluation harness must not average it into an aggregate. When a family does not
emit the channel a detector's access level is defined over, that cell is recorded
as `null` with a reason and **excluded** from `overall_auroc`. It is never imputed
as 0.5. The same applies to the horizon: a detector that declares `min_episodes`
(`PosthocDetector.min_episodes`, default 0) is not scored on a family whose runs are
shorter, and that cell is `null` too, with the horizon as its reason. Reward Skewness
declares 100 and is `null` on the 59 cells of the 19 families that run 40 or 60
episodes.

The rule is one public function, `rhob.v3.benchmark.not_applicable_reason(detector,
runs, level=None)` (2026-11-27). It returns `None` when `detector` can be scored on
`runs` (already restricted to the level), and otherwise the reason string an artifact
records. It checks the channels first (`missing_channels` over
`required_channels(detector)`), then the shortest run against `min_episodes`.
`Benchmark.evaluate` applies it to every cell, and `scripts/cross_family_transfer.py`
applies it to its pooled fit and to every held-out cell (§7);
`external/hvta/score_detectors.py` still applies only the channel check.

The distinction is load-bearing, and it covers every detector on the board: each one
whose `classify` returns a constant below some run length declares that length
(`f83a152`), and `tests/test_detectors/test_min_episodes.py` holds each declaration to
the code, so a number in `per_family` is a measurement. The five external baselines
under `src/rhob/detectors/external_baselines/` are on no board (the HVTA bridge,
`external/hvta/score_detectors.py`, scores them), and since 2026-11-27 they declare
theirs too, held to the same rule by their own table in that test: Page-Hinkley Test 3,
Isolation Forest 2, AR(p) Residual `order + 1` (6 by default), PCA Reconstruction
`steady_window` (100 by default) and Bayesian Online Changepoint Detection 5. On the
board, until 2026-11-26 only Reward Skewness declared one, and the board regenerated
on 2026-11-19 (`52656cb`) published the fallback of eight L0 detectors (Reward Peak,
Reward Autocorrelation, Reward Trend and
Variance Window on the 19 short-horizon families; Max Plateau, Reward MLP, Gradient
Reversal and Reward CUSUM on the 7 that run 40), three L2 detectors (Feature
Consistency on the 19; Feature Magnitude and Reward-Feature Correlation on the 7) and
every L1 detector on `continuous_camping` (it emits `state_counts`, and all 8 L1
detectors read exactly 0.500 on it, but its 40 episodes are below each one's window) as
measured 0.5s. On the re-scored boards (`03d978e` for the single draw) those cells are
`null` with the horizon as the reason, and so they are on the re-drawn boards that
replaced them on 2026-11-27 (`b93c351`). Outside the L0 cells of the six DEGENERATE
families, where 0.5 is the tie convention, a `0.5` in an artifact means *"measured, and
at chance"*.

Imputing the fallback previously caused 88 of 123 cells (71.5%) of every L1
detector's published row to be a constant, which understated the suite: on the
pre-audit board (`git show c54a8cc:leaderboard/v5_leaderboard.json`, the board as it
stood before the 2026-11-19 regeneration) the best L1 detector read 0.621 over all 123
cells. The re-scored board (`03d978e`) records those 88 cells as `null`, and
`continuous_camping`'s 4 as `null` on the horizon, and scores that detector, State
Divergence, over the remaining 31 cells (7 families) at 0.982. The re-drawn board
(`b93c351`), the current single draw, records the same 92 `null`s and reads the same
0.982 over the same 31 cells; the 2026-11-19 board (`52656cb`), with
`continuous_camping`'s four fallback 0.5s still in, read 0.927 over 35.

---

## 4. Detector Interface

### `PosthocDetector` (Abstract)
**Location:** `rhob.detectors.posthoc.PosthocDetector`

**Stability:** FROZEN (v3.2+)

**Methods:**

```python
@property
@abstractmethod
def access_level(self) -> str:
    """Return 'L0', 'L1', 'L2', or 'L3'."""

@property
@abstractmethod
def name(self) -> str:
    """Human-readable name for leaderboard (e.g., 'Reward Threshold').
    
    Convention: Title case, max 40 chars.
    """

@abstractmethod
def classify(self, run: RunData) -> float:
    """Return P(hacking) score in [0.0, 1.0].
    
    0.5 = neutral (no discriminative signal)
    > 0.5 = leans hacking
    < 0.5 = leans legitimate
    """

@abstractmethod
def detect_onset(self, run: RunData) -> int:
    """Return episode index where hacking onset detected.
    
    Return -1 if no onset detected or onset undefined.
    """

def fit(self, runs_a: list[RunData], runs_b: list[RunData], ...) -> None:
    """Optional: Train detector on labeled runs (called during cross-validation).
    
    Only implement if detector requires supervision.
    Signature must accept runs_a, runs_b (lists of RunData).
    Additional kwargs optional (epochs, lr, etc.).
    Default: no-op (detector is unsupervised).
    """

@property
def is_trained(self) -> bool:
    """Return True iff detector is in trained state.
    
    Unsupervised detectors return False.
    Supervised detectors return False until fit() completes.
    Used during cross-validation to gate supervised fitting.
    """

@property
def min_episodes(self) -> int:
    """Optional: fewest episodes this detector can score. Default 0 (no minimum).

    The benchmark and the admission gate record a family whose runs are shorter
    as not applicable (null) instead of calling classify(). Override it whenever
    classify() falls back to a constant on short runs; every detector on the board
    that does so declares it, as does every external baseline since 2026-11-27, and
    tests/test_detectors/test_min_episodes.py checks each declaration against
    classify(): constant one episode below it, varying at it.
    """
```

**Constraints:**
- `classify()` and `detect_onset()` must be deterministic (given same RunData, return same result)
- Both must complete in <1s per run
- Must handle RunData with None fields appropriately (return 0.5 / -1)
- Accessing restricted fields (e.g., L0 detector reading true_rewards) is caught and raises AttributeError

---

## 5. Benchmark Evaluation Signature

### `Benchmark.evaluate()` (Core)
**Location:** `rhob.v3.benchmark.Benchmark`

**Stability:** FROZEN (v3.2+)

**Signature:**

```python
def evaluate(
    self,
    families: str | list[str] = "all",
    difficulties: str | list[float] = "all",
    detectors: list[PosthocDetector],
    n_seeds: int = 10,
    seed_base: int = 0,
) -> BenchmarkResults:
    """Evaluate detectors on family-difficulty pairs.
    
    Returns:
        BenchmarkResults with per-cell AUROC and onsetMAE, indexed by
        (family, mechanism, difficulty, n_seeds).
    """
```

**Guarantees:**
- Deterministic given seeds
- Stratified 5-fold CV for supervised detectors
- Out-of-fold scoring prevents leakage
- Access restrictions enforced
- Results cached by (family, difficulty, seed)

---

## 6. Admission Gate & Certification Contract

**Location:** `rhob.v3.admission_gate`

### Criterion outcomes are tri-state

```python
class CriterionOutcome(Enum):
    PASS = "PASS"                # measured, and the criterion is met
    FAIL = "FAIL"                # measured, and it is not
    DEGENERATE = "DEGENERATE"    # NOT MEASURABLE on this family
```

`DEGENERATE` is **not** a pass and **not** a fail. Both proxy criteria
(`proxy_matched`, `proxy_distribution_matched`) are TOST equivalence tests on an
AUROC. When the detector's scores are all tied across the two variants, the AUROC is
0.5 by the half-credit-per-tie convention, the cluster bootstrap has SE 0, and the
interval collapses to `[0.5000, 0.5000]` — which lies inside any margin. Two guards
run before the test's value is read, and either one produces `DEGENERATE`:

| Guard | Question | API |
|---|---|---|
| Resolution | could the statistic have taken a different value? A layout's AUROC is confined to `0.5 ± resolution/2` | `is_degenerate(resolution, margin)` |
| Informativeness | did the signal mean anything? Pooled proxy `SD / mean(\|proxy\|)`, scale-free | `proxy_informativeness(runs)`, `is_uninformative(v, floor=PROXY_INFORMATIVENESS_FLOOR)` (floor `1e-4`) |

The second exists because the first is bypassable: a constant proxy plus `N(0, 1e-7)`
jitter resolves at 0.961 and would otherwise certify ADMITTED. Informativeness is a
property of the family, not of a detector, so one measurement disqualifies every
equivalence test on that cell.

Consumers of the certificate and the ledger **must not** collapse the three outcomes
to a boolean. A cell is ADMITTED only when every criterion is `PASS`; cells with any
`DEGENERATE` criterion carry no matched-proxy claim in either direction and are
excluded from RHOB's L0-at-chance negative control. NOT ADMITTED cells are not
excluded: they stay in the L0 mean, and whether they should is an open question
(`ROADMAP.md`).

### `admission/admission_ledger.json` artifact schema

| Key | Meaning |
|---|---|
| `schema` | `"rhob.admission_ledger/1"` |
| `provenance` | git commit + dirty flag, Python, platform and CPU, package versions, `numerics` (§7; written since 2026-11-27, so the committed ledger has none), `argv`, scope, gate configuration; a ledger merged from single-cell shards sets `platform`, `cpu` and `numerics.cpu` to null and lists the shards' distinct platforms, CPU model strings and (since 2026-11-27) CPU vendors under `sharding` (shard count, time span, `platforms`, `cpus`, `cpu_vendors`, CI run); the other `numerics.cpu` fields (family, model number, stepping, ISA flags) are not kept. The merge refuses shards whose `numerics.env` differ |
| `design` | layouts, seeds/side, bootstrap resamples, margin, α, shape-detector panel, a-priori power |
| `summary` | `n_cells`, `n_admitted`, `n_degenerate`, `n_not_admitted`, `n_failed_by_criterion`, `n_degenerate_by_criterion`, `n_not_established_by_criterion`, `degenerate_families` |
| `results` | one record per (family, difficulty): `family`, `difficulty`, `passed`, `status`, `criteria`, `outcomes`, `degenerate_criteria`, `details`, `metrics`, `design`, `error` |
| `timing_seconds` | wall clock per family (in a merged ledger, the sum of its cells' wall clock on separate machines); not reproducible |

`results` reproduces byte-for-byte on the same commit and CPU model at the gate's fixed
root seed `12345`; a cell certified on a different CPU model can differ in its last
digits, and the committed ledger was merged from cells certified on seven
(`provenance.sharding.cpus`). Failing **and** degenerate cells are recorded, never
filtered — the ledger is the falsifiable form of the negative control.

"Last digits" understates it in two measured places, both re-certified on another CPU
model on the same lock: `eval_probe_sandbagging` moved a Reward KDE AUROC by 3.6e-4, and
`rlhf_reward_model_overopt`, whose proxy is constant to within rounding, moved its mean
L0 AUROC at 0.98 from 0.5000 to 0.4931, still DEGENERATE on informativeness.
`tests/test_v3/test_ledger_reproduces.py` records both, and leaves both families out
for that reason. *(2026-11-27: the ledger's four `continuous_camping` cells (0.98, 0.9,
0.8, 0.7) were certified unpinned with the DQN camper their runner trained, and that
training gives different weights on Intel and AMD CPUs, so their figures depend on which
CPU vendor certified them, and, unpinned, on the kernel level; by how much has not been
measured. The family has loaded a frozen camper since `ddc8ab5`; see `CHANGELOG.md`.)*

Since 2026-11-27 the test suite holds the committed `results` to the code.
`tests/test_v3/test_ledger_reproduces.py` re-certifies four registered families at every
tier the ledger scores (20 cells, on every push) and one more cell nightly
(`rlhf_feature_blindspot_gaming` at 0.80), with a gate built from the ledger's own
`provenance.gate`. Statuses, outcomes and the masked `details` text must match exactly,
and every number in `metrics` to within 5e-3 on rank statistics and 1e-9 relative
elsewhere, tolerances set from drift measured across the CI lock and the unpinned
Python 3.10 and 3.11 test legs. `tests/test_cannotfail_parity.py` re-certifies the six
DEGENERATE families (two on every push, four nightly).

The three per-criterion blocks in `summary` are **not** interchangeable and none of
them is the old `n_failing_by_criterion`, which was removed when the tri-state landed
because it silently merged the last two rows below:

| Block | Counts cells where the criterion… |
|---|---|
| `n_failed_by_criterion` | was measured and came back `FAIL` |
| `n_degenerate_by_criterion` | could not be measured (`DEGENERATE`) |
| `n_not_established_by_criterion` | did **not** come back `PASS` — the sum of the two above |

Only `n_failed_by_criterion` is evidence against a family; only
`n_cells - n_not_established_by_criterion` is evidence for one. Per-record, `status`
is the tri-state verdict (`"ADMITTED"` / `"DEGENERATE"` / `"NOT ADMITTED"`), `outcomes` is
the per-criterion tri-state, and `criteria` is the lossy boolean projection kept for
backward compatibility — `passed` is `status == "ADMITTED"`, so a `False` there means
"not admitted", never "refuted".

### Two strengths of check, and only one of them certifies

| | Smoke screen | Certification |
|---|---|---|
| Entry point | `tests/test_v3/admission_helpers.assert_smoke_admissible_at`, one case per scored tier | `AdmissionGate.certify_all_tiers` via `scripts/admission_ledger.py` |
| Design | 12 layouts × 4 seeds/side (96 rollouts/cell) | 12 × 24 (576 rollouts/cell) |
| Equivalence margin | `SMOKE_MARGIN` ≈ ±0.256 | `EQUIVALENCE_MARGIN` = ±0.10 |
| Runs in CI | yes | on demand only (`.github/workflows/regenerate.yml`, `workflow_dispatch`), one job per cell; not per push. Since 2026-11-27 every job there runs under the numerics pins (§7), after one `calibrate` job |
| Issues ADMITTED | **no** | yes |

`SMOKE_MARGIN` is `required_seeds_per_layout` inverted at the smoke design, not a
chosen tolerance, and `tests/test_v3/test_admission_smoke_design.py` asserts both that
identity and `SMOKE_MARGIN > EQUIVALENCE_MARGIN`. A green family test is a screen
against *large* leaks; where the screen and the ledger disagree, the ledger is
authoritative.

**Neither column is enforced over the registry.** 22 of the 33 registered families call
`assert_smoke_admissible_at`, 11 do not, and 8 never reach `AdmissionGate` in any test
(10 until 2026-11-27, when the ledger-reproduction test above began re-certifying
`proxy_correlation_gaming` on every push and the cannotfail parity test
`orbit_chirality` nightly). No test in `tests/` calls `certify_all_tiers` on a
*registered* family — the two that exercise it use synthetic fixtures — and the tests
that re-certify registered families only compare their certificates with the committed
ledger, so `scripts/admission_ledger.py` is the only caller whose certificates are
published. The ledger it writes covers all 33 since its 2026-11-19 regeneration on CI, as
a separate run rather than a test. Read a green suite as "the screens that exist
passed", never as "every family is admitted".

---

## 7. Leaderboard Schema

### `LeaderboardEntry` (Dataclass)
**Location:** `rhob.v3.leaderboard.board.LeaderboardEntry`

**Stability:** FROZEN (v3.2+)

**Schema:**

```python
@dataclass
class LeaderboardEntry:
    detector_name: str                    # From detector.name
    access_level: str                     # 'L0', 'L1', 'L2', 'L3'
    author: str                           # Detector author (for attribution)
    timestamp: str                        # ISO 8601, UTC (datetime.isoformat())
    overall_auroc: Optional[float]        # Mean AUROC over the MEASURED cells; None if none
    n_cells: int                          # Number of measured cells: overall_auroc's denominator
    family_auroc: dict[str, float]        # family -> mean AUROC (default {})
    mechanism_auroc: dict[str, float]     # mechanism -> mean AUROC (default {})
    tier_auroc: dict[str, float]          # DifficultyTier name ("MEDIUM") -> mean AUROC (default {})
```

This block listed `per_family_auroc`, `per_mechanism_auroc` and `per_difficulty_auroc`,
and no `n_cells`, until 2026-11-27; the dataclass's fields are the ones above.
`LeaderboardEntry.from_results` (and so `python -m rhob evaluate`) averages and counts
only measured cells: since 2026-11-27 `n_cells` excludes not-applicable cells, and a run
with no measured cell has `overall_auroc` `None` (`null`) and `n_cells` 0, never `NaN`.

**Validation.** `rhob.v3.leaderboard.board.submission_problems(data)` returns why `data`
is not a valid submission or board entry, and an empty list when it is. The four keys
`detector_name`, `access_level`, `overall_auroc` and `n_cells` are required;
`detector_name` is a non-empty string, `access_level` one of L0–L3, `overall_auroc` a
finite number in [0, 1] or `null`, `n_cells` a non-negative integer (`bool` is refused
for both numbers); `author` and `timestamp`, when present and not `null`, are strings;
`family_auroc`, `mechanism_auroc` and `tier_auroc`, when present, map names to finite
numbers in [0, 1], and the tier names are `DifficultyTier` names. Unknown keys are
ignored. `strict_json_loads` refuses `NaN`, `Infinity` and `-Infinity`, which are not
JSON. `python -m rhob validate` applies both (exit 1 on a problem), and so does
`.github/workflows/leaderboard_validate.yml`, which runs it on submission pull requests.

**The community board.** `Leaderboard.load(path)` reads an `{"entries": [...]}` board
(an empty board when `path` does not exist) and raises `ValueError` on anything else:
a file in another schema such as `leaderboard/v5_leaderboard.json`, invalid JSON (`NaN`
included), or an entry that does not match `LeaderboardEntry` or fails
`submission_problems`. It used to read a file in another schema as an empty board.
`Leaderboard.dumps()` serializes with `allow_nan=False`, and `save` writes nothing when
that raises. `python -m rhob submit SUBMISSION` records self-reported, single-draw
submissions in `leaderboard/community/` (`board.json` plus `README.md`,
`by_mechanism.md` and `by_difficulty.md` rendered from it), or in `--board-dir DIR`; it
never writes the benchmark's own boards in `leaderboard/`. It fills a missing `author`
(`"anonymous"`) and `timestamp` (the time of submission), applies `--name` and
`--author`, validates again, and renders the board and all three tables before writing
any of them. It refuses with exit 1, writing nothing, when the submission does not
validate, when `board.json` is not a valid entries-schema board, or when `README.md`,
`by_mechanism.md` or `by_difficulty.md` in the directory does not start with the
community header (`# RHOB community submissions (self-reported, single draw)`); other
files there are left alone. So `--board-dir leaderboard` is refused while the
hand-written index `leaderboard/README.md` is there.

### `leaderboard/v5_leaderboard.json` artifact schema (additive, non-breaking)

Top-level blocks:

| Key | Meaning |
|---|---|
| `provenance` | `generated_utc`, git commit / branch / dirty flag + dirty file list, Python and platform, tracked package versions, `numerics` (below; written since 2026-11-27, so an artifact stamped before then has no such key), `argv`, `script` |
| `sampling` | the draw: `n_seeds_per_variant`, `n_layouts`, `layout_seeds`, the explicit rollout seed lists, `n_replicates`, `single_draw`, `shared_draw_across_detectors`, `confidence_intervals` (null), and a `note` |
| `cell_semantics` | what a `null` per-family value means vs. a number |
| `families_evaluated` | explicit list of families in the run |
| `timestamp` | retained for backward compatibility with `rhob.v3.leaderboard.adapters` and `space/app.py`; mirrors `provenance.generated_utc` |

`provenance.numerics` (`rhob.v3.provenance.numerics_block`) is in every block
`provenance_block` stamps since 2026-11-27, in this board and in the other artifacts the
scripts write. It records what chose the floating-point kernels:

| Key | Meaning |
|---|---|
| `env` | each of `rhob.v3.calibration_cache.NUMERICS_ENV_VARS` (`OPENBLAS_CORETYPE`, `OPENBLAS_NUM_THREADS`, `MKL_CBWR`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`, `ATEN_CPU_CAPABILITY`), or `null` when unset. Every job of `.github/workflows/regenerate.yml` sets them to `Haswell`, `1`, `COMPATIBLE`, `1`, `1`, `avx2` |
| `cpu` | from `/proc/cpuinfo` where it exists: `vendor`, `model`, `family`, `model_number`, `stepping`, `isa` (the kernel-relevant ISA flags present), `logical_cpus`, `source`; elsewhere only `model`, from `platform.processor()` |
| `torch_imported` | whether the process had loaded torch when the block was stamped |
| `torch_cpu_capability` | the ATen kernel level torch dispatches to (`"AVX2"` under the pin), read on every torch 2.x; `null` when torch was not imported |
| `blas` | the BLAS and OpenMP libraries loaded, as threadpoolctl reports them (`user_api`, `internal_api`, `prefix`, `version`, `architecture`, `threading_layer`, `num_threads`); `architecture` is the kernel OpenBLAS selected (`"Haswell"` under the pin). `null` when threadpoolctl is unavailable |

Two runs with equal `env` and equal `blas` architectures picked the same kernels; equal
`cpu` is neither needed for that nor implied by it, and a claim that results agree across
CPU vendors needs two artifacts whose `cpu.vendor` differ. In one place equal recorded
kernels did not give equal results across vendors: under the pins, with ATen at `AVX2`
and OpenBLAS at `Haswell` on both, `continuous_camping`'s DQN camper trained to
different weights on Intel and AMD runners, and what made them differ is not among what
this block records (numpy without AVX-512, and oneDNN or MKL capped at AVX2, each tried
on AMD runners, changed nothing), so since 2026-11-27 (`ddc8ab5`) the family loads
frozen weights (`src/rhob/environments/continuous/frozen_camper.py`,
written by `scripts/freeze_camper.py`) instead of training them.
`scripts/check_redraw.py` refuses a re-draw unless every file records the pins and the
kernels show they took effect: ATen at `AVX2` wherever torch loaded, every OpenBLAS at
`Haswell`.

`scripts/v5_leaderboard_and_transfer.py --cells-out PATH` also writes the board's
per-cell AUROCs to a sidecar, `{"provenance": ..., "cells": {detector: [...]}}`, whose
`provenance` is the board's own block. It is not committed: `regenerate.yml` uploads it
beside the board in every mode that scores one (`board`, `both` and `redraw`), and in
`redraw` `scripts/check_redraw.py` requires replicate 0 of the re-draw to equal it cell
for cell (up to `--max-board-mismatch` cells, default 0). That check refused the first
re-draw under the pins (run 45995235365, at `d85a60c`): 16 `continuous_camping` cells
differed, the camper above. The committed re-draw (`b93c351`, run 46011019246 at
`54cd63c`) passed it with the board scored on an AMD runner and replicate 0 on an Intel
one: two artifacts whose `cpu.vendor` differ, equal cell for cell. Its 20 draws ran on
12 Intel and 8 AMD runners.

Per-detector keys under `results[detector]`:

| Key | Meaning |
|---|---|
| `cells` | cells attempted |
| `cells_measured` | cells scored: the channel is present and the horizon is at least the detector's declared `min_episodes`. Every detector on the board whose score is a constant below some run length declares that length (`tests/test_detectors/test_min_episodes.py`), so a cell counted here is a measurement |
| `cells_not_applicable` | cells skipped as not applicable: the family does not emit the detector's channel, or its horizon is below the detector's declared `min_episodes` (on the committed board, nine L0 detectors, every L1 detector and three L2 detectors have a minimum that some families fall below); the reason is in `not_applicable_reasons` |
| `not_applicable_families` | list of families skipped for this detector |
| `not_applicable_reasons` | `{reason string: count}` |
| `per_family` | float, **or `null`** when not applicable |
| `overall_auroc` | float, or `null` if nothing was measurable |

`NaN` is never written — a bare `NaN` token is invalid JSON and fails strict
parsers, including the Gradio Space. Not-applicable and unmeasurable both serialize
to `null`.

**Consumers must treat `null` as "no measurement exists", not as 0.5, and must not
count it in a denominator.**

### Cross-family transfer artifact schema

`scripts/cross_family_transfer.py` writes `leaderboard/cross_family_transfer_v5.json` by
default (`rhob.v3.leaderboard.transfer.CURRENT_TRANSFER_ARTIFACT`). It refuses to
overwrite `leaderboard/cross_family_transfer.json` (`HISTORICAL_TRANSFER_ARTIFACT`), the
2026-07-09 record of what was published before sign randomization and the
not-applicable rules. That file carries only the family lists, the design and each row's
AUROCs and gap, with no `provenance`, `sampling`, `configs` or not-applicable
bookkeeping.

Top-level keys:

| Key | Meaning |
|---|---|
| `train_families`, `test_families` | the 6 train families and the 8 held out |
| `n_seeds_train`, `n_seeds_test`, `n_trials` | the design actually run |
| `published_design` | `true` only for the published design: all four rows, both family lists, 10 train seeds, 20 test seeds, 5 trials. A reduced smoke run records `false` |
| `configs` | `{"shipped": {"randomize_behav_sign": true, "detector_infers_orientation": true}}`, so the loader reads the file as the shipped configuration |
| `provenance` | as above, with `wall_clock_s` per row |
| `sampling` | as above, plus `model_init_trials` (the weight-init replication axis — **distinct from** `n_replicates`, which is the environment draw and is 1), `test_seed_base` (50000), `n_seeds_test`, `rollout_seeds_hacking_test` and `rollout_seeds_legit_test` |
| `cell_semantics` | what a `null` and a number mean in each field |
| `results` | one row each for Reward MLP, State Divergence, Trajectory MLP and Ensemble (Top 5) |

Per-row keys. The three neural-net rows are repeated over `n_trials` model
initialisations and report `_mean` (and, for the AUROCs, `_std`) forms of the keys below
(`train_auroc_mean`, `per_family_transfer_mean`, `avg_transfer_auroc_mean`,
`generalization_gap_pct_mean`), plus `n_trials`, `all_trial_train_aurocs`,
`all_trial_transfer_aurocs` and `per_family_trials_not_applicable` (families not
applicable in some trials but not all). State Divergence is fitted once and reports the
plain forms.

| Key | Meaning |
|---|---|
| `access_level` | the level the row's detector is restricted to |
| `train_auroc` | 5-fold stratified cross-validated AUROC over the scored train cells, out of fold for every row, the ensemble included |
| `train_cells_measured`, `train_cells_not_applicable`, `train_families_not_applicable`, `train_not_applicable_reasons` | the train cells `not_applicable_reason` left out, and why |
| `train_families_excluded_from_pooled_fit`, `train_excluded_from_pooled_fit_reasons` | train families that contributed no runs to the frozen model's fit, and why |
| `pooled_fit_error` | why the pooled fit produced no frozen model (for a multi-trial row, each failing trial's reason), else `null`; a fit that failed scores no held-out cell |
| `per_family_transfer` | the frozen model's mean AUROC on each held-out family over its scored cells, or `null` |
| `test_families_not_applicable`, `test_not_applicable_reasons` | held-out families with no scored cell, and why |
| `per_family_cells_scored`, `per_family_cells_not_applicable`, `test_partial_not_applicable_reasons` | how many of each held-out family's difficulty cells its number is over and how many were left out, and the reasons for a family scored on some cells only (for a multi-trial row, the fewest scored and the most left out in any trial) |
| `per_family_incompatible_runs` | how many runs, per held-out family, the frozen model's `classify` raised on because it cannot consume that family's version of the channel (for a multi-trial row, the most in any trial); each cell with such a run is not applicable, not scored at 0.5, and a family with no other scored cell is in `test_families_not_applicable` |
| `avg_transfer_auroc` (the RTS), `generalization_gap_pct` | over the scored held-out families; `null` when none was scored |

`rhob.v3.leaderboard.transfer.load_transfer_results(path)` reads this file, the
historical one, and the `transfer` block of `docs/figures/sign_randomization_impact.json`.
Since 2026-11-27 `TransferRow` also carries `not_applicable_reasons`,
`incompatible_runs`, `pooled_fit_error` and `records_na_reasons` (the row was written
under the run-length and representational rules), and `TransferBoard` carries
`git_dirty`, `published_design` and `applies_na_rules` (every row was).

> `leaderboard/v5_leaderboard.json` carries `provenance`, `sampling` and
> `cell_semantics` since its 2026-11-19 regeneration (`52656cb`).
> `leaderboard/cross_family_transfer.json` predates `src/rhob/v3/provenance.py` and is
> kept unchanged as the historical record, so it does not carry them;
> `scripts/cross_family_transfer.py` writes all three into
> `leaderboard/cross_family_transfer_v5.json`, which carries them since its first CI run
> (`9c038f6`, 2026-11-27) and is the current transfer artifact. The committed file is
> `02c0475` (CI run 46011034429 at `54cd63c`, `published_design` true), its `results`
> identical to those of `9c038f6` and of `7d34989` (CI run 45995248987 at `d85a60c`);
> all three ran on Intel runners under the pins. `admission/admission_ledger.json`
> carries `provenance` and its own `design` block, not `sampling` or `cell_semantics`.

### Aggregation contract: duplicate detectors

A detector that duplicates another must not be counted twice in any
per-access-level aggregate. `rhob.detectors.redundancy.DUPLICATE_DIAGNOSTICS` maps
a duplicate's `name` to the name of the detector it duplicates;
`is_duplicate_diagnostic(name)` is the check every aggregate must apply. The
duplicate keeps its leaderboard row and its reported access level — artifacts and
downstream consumers key off them — but belongs to **no** level's aggregate, and
is not re-filed under the level it actually duplicates (that would just move the
double-count).

Current entry: `"Perfect Feature Oracle" -> "Behavioral Threshold"`.

---

## 8. Taxonomy Enums (Frozen)

### `HackingMechanism` (Enum)
```python
class HackingMechanism(str, Enum):
    CAMPING_EXPLOIT = "camping"
    REWARD_SHAPING = "shaping"
    SHORTCUT = "shortcut"
    EXPLORATION_EXPLOIT = "exploration"
    PROXY_GAMING = "proxy_gaming"
    GOAL_MISGENERALIZATION = "goal_misgen"
    DECEPTIVE_ALIGNMENT = "deceptive"
    REWARD_TAMPERING = "tampering"
```

### `EnvironmentComplexity` (Enum)
```python
class EnvironmentComplexity(str, Enum):
    TABULAR = "tabular"
    CONTINUOUS_SIMPLE = "cont_2d"
    CONTINUOUS_COMPLEX = "cont_hd"
    SEQUENTIAL = "sequential"
    MULTI_AGENT = "multi_agent"
```

### `DifficultyTier` (Enum, values only—names may be deprecated)
```python
class DifficultyTier(float, Enum):
    TRIVIAL = 0.98
    EASY = 0.90
    MEDIUM = 0.80
    HARD = 0.70
    EXTREME = 0.60
```

---

## 9. Deprecation Policy

**v3.2 Freeze:** No breaking changes to above interfaces until v4.0.

**Additive changes allowed:**
- New fields to dataclasses (with defaults)
- New detector access levels (e.g., L4) if truly novel
- New mechanisms or complexity classes

**Non-breaking extensions:**
- New methods on BaseFamily (optional, fallback defaults)
- New optional parameters (must have sensible defaults)
- New optional fields on RunData (must remain None-safe)

**Breaking changes require v4.0:**
- Removing fields or methods
- Changing field types
- Changing method signatures (without backward-compatible overload)

---

## 10. Registration & Discovery

### Family Registry
```python
@FamilyRegistry.register("family_name")
class MyFamily(BaseFamily):
    ...
```

Families auto-discover at import of `rhob.v3.families`.

### Detector Export
Detectors exported from `rhob.detectors.__all__` for CLI discovery.

### Validation
- Family names must be unique, snake_case, max 32 chars
- Detector names must be unique, title case, max 40 chars
- Mechanisms must be from enum (not free-form strings)

---

## 11. Community Contribution Template

**To add a new family:**
1. Create `src/rhob/v3/families/my_family.py`
2. Inherit from `BaseFamily`
3. Implement the abstract methods (name, mechanism, complexity, difficulty_range, generate_pair)
4. Register with `@FamilyRegistry.register("my_family")`
5. Emit `state_counts` if the environment has a natural fixed-bin state histogram;
   if not, leave it `None` and document why in the module docstring
6. Clear all **6** admission criteria at **every** difficulty
   (`AdmissionGate.certify_all_tiers`), not just `default_difficulties()[0]` — with
   no criterion `DEGENERATE`, which a constant proxy guarantees (see §6)
7. Add tests in `tests/test_v3/test_my_family.py`, including the reduced-power smoke
   screen (`assert_smoke_admissible_at`, one case per tier). It is what runs in CI; it does
   **not** certify the ±0.10 claim — only the ledger does
8. Submit PR with anti-symmetry screening audit

**To add a new detector:**
1. Create `src/rhob/detectors/lX_my_detector.py` (X = 0, 1, 2, 3)
2. Inherit from `PosthocDetector`
3. Implement 4 abstract methods (access_level, name, classify, detect_onset)
4. Optional: implement fit() if supervised
5. Add to `src/rhob/detectors/__init__.py` __all__
6. Add tests in `tests/test_detectors/test_lX_my_detector.py`
7. Submit PR

---

## 12. Version History

| Version | Date | Status | Changes |
|---------|------|--------|---------|
| v3.0 | 2026-06-15 | SHIPPED | Core infrastructure, 2 families, 6 detectors |
| v3.1 | 2026-07-07 | SHIPPED | 4 families, 23 detectors, gridworld leaderboard |
| v3.2 | TBD | FROZEN API | 8+ families, 40+ detectors, interactive leaderboard |
| v4.0 | TBD | PLANNED | Breaking changes allowed, multi-agent support, v4 mechanisms |

