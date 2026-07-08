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

**Access Restrictions (enforced at evaluate time):**
- L0: ONLY proxy_rewards
- L1: proxy_rewards + state_counts
- L2: proxy_rewards + state_counts + behav_trace
- L3: ALL fields (oracle only, never scored in production)

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

## 6. Leaderboard Schema

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
    timestamp: str                        # ISO 8601 (YYYY-MM-DDTHH:MM:SSZ)
    overall_auroc: float                  # Mean AUROC across all cells
    per_family_auroc: dict[str, float]    # family -> mean AUROC
    per_mechanism_auroc: dict[str, float] # mechanism -> mean AUROC
    per_difficulty_auroc: dict[str, float]# "0.60" (as string) -> mean AUROC
```

---

## 7. Taxonomy Enums (Frozen)

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

## 8. Deprecation Policy

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

## 9. Registration & Discovery

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

## 10. Community Contribution Template

**To add a new family:**
1. Create `src/rhob/v3/families/my_family.py`
2. Inherit from `BaseFamily`
3. Implement 6 abstract methods (name, mechanism, complexity, difficulty_range, generate_pair)
4. Register with `@FamilyRegistry.register("my_family")`
5. Add tests in `tests/test_v3/test_my_family.py`
6. Submit PR with anti-symmetry screening audit

**To add a new detector:**
1. Create `src/rhob/detectors/lX_my_detector.py` (X = 0, 1, 2, 3)
2. Inherit from `PosthocDetector`
3. Implement 4 abstract methods (access_level, name, classify, detect_onset)
4. Optional: implement fit() if supervised
5. Add to `src/rhob/detectors/__init__.py` __all__
6. Add tests in `tests/test_detectors/test_lX_my_detector.py`
7. Submit PR

---

## 11. Version History

| Version | Date | Status | Changes |
|---------|------|--------|---------|
| v3.0 | 2026-06-15 | SHIPPED | Core infrastructure, 2 families, 6 detectors |
| v3.1 | 2026-07-07 | SHIPPED | 4 families, 23 detectors, gridworld leaderboard |
| v3.2 | TBD | FROZEN API | 8+ families, 40+ detectors, interactive leaderboard |
| v4.0 | TBD | PLANNED | Breaking changes allowed, multi-agent support, v4 mechanisms |

