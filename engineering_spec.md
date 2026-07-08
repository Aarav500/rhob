# Conversation
- conversationId: 07078a61-04d4-43ad-9e7b-5259794cb9c9
- endpoint: agents
- title: New Chat
- exportAt: 18:32:08 GMT-0500 (Central Daylight Time)

## Options
- presetId: null
- agent_id: agent_sVkcZZtm0xlbpcQEGBtfe
- endpoint: agents
- endpointType: undefined
- title: New Chat

## History
**User**
Using the publication blueprint,

design the engineering architecture for the Reward-Hacking Onset Benchmark.

Design

- repository structure

- package layout

- APIs

- benchmark interface

- environment abstraction

- detector API

- evaluation API

- metrics

- leaderboard format

- Flight Recorder integration

- configuration system

Do NOT write code.

Produce a production-quality engineering specification suitable for implementation in Claude Code.


**Research Engineering Architect**
{"type":"think","think":"Aarav wants me to take the publication blueprint for RHOB (Reward-Hacking Onset Benchmark) and design the full engineering architecture. This is a research engineering specification — not code, but a detailed architectural document that would enable implementation in Claude Code or by any competent engineer.\n\nLet me think about this systematically. The publication blueprint gives me:\n- A benchmark with 21 environments across 4 tiers\n- 7 metrics\n- 6+ baselines\n- Access levels L1-L4\n- A detector API\n- Pre-recorded trajectory evaluation\n- A leaderboard\n- Integration with Flight Recorder\n\nI need to design:\n- Repository structure and Python package hierarchy\n- Public APIs and how users interact with the benchmark\n- Environment abstraction as data generators\n- How detection methods plug into the system\n- Evaluation pipeline for pre-recorded trajectories\n- Metrics computation and aggregation\n- Leaderboard submission and ranking format\n- Flight Recorder integration\n- Configuration system\n\nThe key design principle is making this pip-installable with minimal friction, where environments act as data generators that detectors consume from pre-recorded trajectories rather than interactive simulations.\n\nThe core architectural decisions center on pre-recorded evaluation as the primary mode for deterministic results, using HDF5 for trajectory storage with JSON metadata and YAML configuration, and keeping detectors as stateless functions or simple stateful objects with a step interface that produces reproducible results.\n\nI'm settling on the repository structure with `rhob` as both the package and project name, organizing it into separate directories for the core package, tests, configurations, environment definitions, data management scripts, documentation, and performance benchmarks.\n\nStructuring the package itself with a core module for fundamental abstractions like trajectory data models and onset labels, an environments module with a registry system for managing different environment tiers, and additional modules for detectors and evaluation metrics. Continuing with the evaluation pipeline, I'm adding score aggregation and report generation, then moving into data management with loaders for datasets and HDF5 storage, configuration handling with schema validation, a leaderboard system for submissions and rankings, and finally the command-line interface.\n\nNow I'm defining the core data types that underpin everything: timesteps representing individual state-action-reward transitions, trajectories as full sequences with metadata, onset labels for ground-truth annotations with confidence scores, access levels as an enumeration, and detection scores from individual detectors. I'm also sketching out an environment abstraction with methods to generate training runs and retrieve oracle onset information.\n\nThe detector interface needs to handle stateful processing where it consumes observations one at a time and outputs confidence scores, along with properties describing what information it requires and its computational characteristics. For evaluation, I'm designing a unified API that runs detectors against datasets and produces comprehensive reports with metrics like detection latency, AUROC, AUPRC, false positive rates, and time to first detection."}


**Research Engineering Architect**
# Engineering Architecture Specification: RHOB

## Reward-Hacking Onset Benchmark

---

## Executive Summary

RHOB is a pip-installable Python package that provides the first standardized evaluation framework for reward-hacking onset detection. The architecture separates three concerns: **environments** (data generation), **detectors** (method interface), and **evaluation** (metrics + reporting). The system prioritizes deterministic reproducibility, minimal-friction developer experience, and extensibility for community contributions.

**Design philosophy:** A researcher should go from `pip install rhob` to a full evaluation report in under 10 lines of code. A contributor should implement a new detector or environment by subclassing a single abstract class.

---

## 1. Repository Structure

```
rhob/
│
├── src/
│   └── rhob/                          # Main package (src-layout)
│       ├── __init__.py                 # Public API surface
│       ├── _version.py                 # Single source of version truth
│       │
│       ├── core/                       # Core data models and type system
│       │   ├── __init__.py
│       │   ├── types.py                # Fundamental type definitions
│       │   ├── trajectory.py           # Trajectory data model
│       │   ├── onset.py                # Onset label model
│       │   ├── access.py               # Access level algebra
│       │   └── exceptions.py           # Package-wide exception hierarchy
│       │
│       ├── environments/               # Environment abstraction + implementations
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract environment contract
│       │   ├── registry.py             # Global environment registry
│       │   ├── oracle.py               # Onset oracle base class
│       │   ├── difficulty.py           # Difficulty knob abstraction
│       │   ├── tier1/                  # Tabular / low-dimensional
│       │   │   ├── __init__.py
│       │   │   ├── navigation_proxy.py
│       │   │   ├── gridworld_wireheading.py
│       │   │   ├── bandit_gaming.py
│       │   │   ├── cliff_shortcut.py
│       │   │   └── reward_tampering_tab.py
│       │   ├── tier2/                  # Continuous control / medium-scale
│       │   │   ├── __init__.py
│       │   │   ├── halfcheetah_proxy.py
│       │   │   ├── reacher_misaligned.py
│       │   │   ├── cartpole_exploit.py
│       │   │   ├── lunar_specification.py
│       │   │   ├── ant_proxy_diverge.py
│       │   │   ├── hopper_gaming.py
│       │   │   └── walker_misgen.py
│       │   ├── tier3/                  # High-dimensional / LLM-adjacent
│       │   │   ├── __init__.py
│       │   │   ├── text_reward_model.py
│       │   │   ├── code_proxy.py
│       │   │   ├── dialogue_sycophancy.py
│       │   │   └── vision_reward_hack.py
│       │   └── adversarial/            # Constructed to defeat known detectors
│       │       ├── __init__.py
│       │       ├── stealth_diverge.py
│       │       ├── gradual_corruption.py
│       │       └── mimicry.py
│       │
│       ├── detectors/                  # Detector abstraction + baselines
│       │   ├── __init__.py
│       │   ├── base.py                 # Abstract detector contract
│       │   ├── registry.py             # Global detector registry
│       │   ├── baselines/              # Built-in baseline detectors
│       │   │   ├── __init__.py
│       │   │   ├── random_detector.py
│       │   │   ├── cusum.py
│       │   │   ├── ensemble_disagreement.py
│       │   │   ├── kl_monitor.py
│       │   │   ├── gradient_norm.py
│       │   │   └── oracle_detector.py
│       │   └── wrappers/              # Utility wrappers
│       │       ├── __init__.py
│       │       ├── access_filter.py    # Restricts observation to access level
│       │       ├── calibrated.py       # Platt-calibrates raw scores
│       │       └── windowed.py         # Adds sliding-window aggregation
│       │
│       ├── evaluation/                 # Evaluation pipeline
│       │   ├── __init__.py
│       │   ├── runner.py               # Orchestrates detector × environment
│       │   ├── metrics.py              # Individual metric computations
│       │   ├── aggregation.py          # Score aggregation logic
│       │   ├── report.py              # Structured report generation
│       │   ├── statistical.py          # Statistical testing utilities
│       │   └── comparison.py           # Head-to-head method comparison
│       │
│       ├── data/                       # Data management layer
│       │   ├── __init__.py
│       │   ├── storage.py              # HDF5 read/write abstraction
│       │   ├── loader.py               # Dataset loading (lazy + eager)
│       │   ├── download.py             # Remote dataset fetching
│       │   ├── splits.py               # Train/val/test split logic
│       │   ├── cache.py                # Local caching layer
│       │   └── schema.py               # Data schema validation
│       │
│       ├── config/                     # Configuration system
│       │   ├── __init__.py
│       │   ├── schema.py               # Pydantic config models
│       │   ├── defaults.py             # Default configuration values
│       │   ├── loader.py               # YAML/TOML loading + merge logic
│       │   └── validation.py           # Config validation + error messages
│       │
│       ├── leaderboard/                # Leaderboard infrastructure
│       │   ├── __init__.py
│       │   ├── submission.py           # Submission format + validation
│       │   ├── ranking.py              # Ranking algorithms
│       │   ├── export.py               # JSON/Markdown/HTML export
│       │   └── history.py             # Historical result tracking
│       │
│       ├── integrations/               # External tool integrations
│       │   ├── __init__.py
│       │   ├── flight_recorder.py      # Flight Recorder adapter
│       │   ├── wandb.py                # Weights & Biases logging
│       │   └── mlflow.py               # MLflow logging
│       │
│       ├── visualization/              # Plotting and visualization
│       │   ├── __init__.py
│       │   ├── trajectory_plot.py      # Proxy/true reward + onset overlay
│       │   ├── roc_curves.py           # ROC/PR curves per method
│       │   ├── difficulty_heatmap.py   # Method × environment heatmap
│       │   └── latency_distribution.py # Detection latency histograms
│       │
│       └── cli/                        # Command-line interface
│           ├── __init__.py
│           ├── main.py                 # Entry point (rhob ...)
│           ├── evaluate.py             # rhob evaluate
│           ├── download.py             # rhob download
│           ├── submit.py               # rhob submit
│           ├── generate.py             # rhob generate (create trajectories)
│           ├── info.py                 # rhob info (environment metadata)
│           └── validate.py             # rhob validate (check submission)
│
├── tests/
│   ├── conftest.py                     # Shared fixtures
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_trajectory.py
│   │   ├── test_onset.py
│   │   ├── test_access.py
│   │   ├── test_metrics.py
│   │   ├── test_aggregation.py
│   │   ├── test_storage.py
│   │   ├── test_config.py
│   │   └── test_ranking.py
│   ├── integration/
│   │   ├── test_evaluation_pipeline.py
│   │   ├── test_environment_generation.py
│   │   ├── test_detector_contract.py
│   │   ├── test_data_roundtrip.py
│   │   └── test_cli.py
│   ├── property/                       # Hypothesis-based property tests
│   │   ├── test_metric_properties.py
│   │   ├── test_access_monotonicity.py
│   │   └── test_score_bounds.py
│   └── regression/
│       └── test_known_results.py       # Pin known-good metric values
│
├── configs/
│   ├── default.yaml                    # Full default configuration
│   ├── quick.yaml                      # Minimal config for fast iteration
│   ├── paper.yaml                      # Exact config used in paper
│   ├── tier1_only.yaml                 # Subset configs
│   ├── tier2_only.yaml
│   └── ablations/
│       ├── lookback_sweep.yaml
│       ├── seed_count.yaml
│       ├── clean_ratio.yaml
│       └── difficulty_sweep.yaml
│
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── api_reference/                  # Auto-generated from docstrings
│   ├── tutorials/
│   │   ├── 01_evaluate_baseline.md
│   │   ├── 02_implement_detector.md
│   │   ├── 03_add_environment.md
│   │   └── 04_submit_to_leaderboard.md
│   ├── contributing/
│   │   ├── detector_guide.md
│   │   ├── environment_guide.md
│   │   ├── code_style.md
│   │   └── review_process.md
│   └── paper/
│       ├── reproduction.md             # Steps to reproduce all paper results
│       └── figures.md                  # How each figure was generated
│
├── examples/
│   ├── minimal_evaluation.py           # 10-line evaluation
│   ├── custom_detector.py              # Implement and evaluate a detector
│   ├── custom_environment.py           # Add a new environment
│   ├── ablation_study.py              # Run parameter sensitivity
│   ├── head_to_head.py                # Compare two methods
│   └── visualization.py              # Generate paper-quality plots
│
├── scripts/
│   ├── generate_all_data.py           # Generate full dataset
│   ├── run_all_baselines.py           # Reproduce paper results
│   ├── update_leaderboard.py          # Refresh leaderboard from submissions
│   └── validate_environments.py       # Check all envs produce hacking
│
├── leaderboard/
│   ├── results/                        # Committed result JSONs
│   │   ├── flight_recorder_v1.json
│   │   ├── cusum_v1.json
│   │   └── ...
│   ├── leaderboard.json               # Current rankings
│   └── schema.json                    # Submission format schema
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                      # Lint + unit tests on every PR
│   │   ├── integration.yml            # Full integration tests (weekly)
│   │   ├── release.yml                # PyPI publish on tag
│   │   └── leaderboard.yml           # Auto-validate submissions
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── new_environment.md
│   │   ├── new_detector.md
│   │   └── leaderboard_submission.md
│   └── PULL_REQUEST_TEMPLATE.md
│
├── docker/
│   ├── Dockerfile                      # Reproducibility container
│   ├── Dockerfile.gpu                  # GPU-enabled variant
│   └── docker-compose.yml             # Full evaluation stack
│
├── pyproject.toml                      # Package metadata + build config
├── README.md                           # Project overview + quickstart
├── CHANGELOG.md                        # Version history
├── LICENSE                             # MIT
├── CITATION.cff                        # Citation metadata
└── Makefile                            # Common developer commands
```

---

## 2. Package Layout & Dependency Architecture

### 2.1 Dependency Graph (Internal)

```
                    ┌──────────────┐
                    │   rhob.cli   │  ← User entry point
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  evaluation  │  ← Orchestration layer
                    └──┬───┬───┬──┘
                       │   │   │
          ┌────────────┘   │   └────────────┐
          │                │                │
   ┌──────▼──────┐  ┌─────▼─────┐  ┌───────▼──────┐
   │ environments │  │ detectors │  │     data     │
   └──────┬───────┘  └─────┬─────┘  └───────┬──────┘
          │                │                │
          └────────┬───────┴────────┬───────┘
                   │                │
            ┌──────▼──────┐  ┌──────▼──────┐
            │    core     │  │   config    │
            └─────────────┘  └─────────────┘
```

**Rule:** Dependencies flow downward only. `core` and `config` depend on nothing internal. `environments`, `detectors`, and `data` depend only on `core` and `config`. `evaluation` depends on all three. `cli` depends on `evaluation`.

### 2.2 External Dependencies (Minimal, Tiered)

| Tier | Dependencies | Required For |
|---|---|---|
| **Core** (required) | `numpy`, `h5py`, `pydantic`, `pyyaml`, `click` | Basic functionality |
| **Evaluation** (required) | `scipy`, `scikit-learn` | Metric computation, statistical tests |
| **Environments** (optional per tier) | `gymnasium` (Tier 1-2), `stable-baselines3` (generation) | Running/generating environments |
| **Visualization** (optional) | `matplotlib`, `seaborn` | Plotting |
| **Integrations** (optional) | `wandb`, `mlflow` | Experiment tracking |
| **Development** (dev only) | `pytest`, `hypothesis`, `ruff`, `mypy`, `pre-commit` | Testing + quality |

### 2.3 Installation Profiles

```toml
# pyproject.toml (relevant sections)
[project]
name = "rhob"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "h5py>=3.8",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
]

[project.optional-dependencies]
environments = ["gymnasium>=0.29", "stable-baselines3>=2.0"]
viz = ["matplotlib>=3.7", "seaborn>=0.12"]
integrations = ["wandb>=0.15", "mlflow>=2.5"]
all = ["rhob[environments,viz,integrations]"]
dev = ["pytest>=7.0", "hypothesis>=6.0", "ruff", "mypy", "pre-commit", "pytest-cov"]
```

---

## 3. Core Type System

### 3.1 Fundamental Types (`rhob.core.types`)

| Type | Description | Representation |
|---|---|---|
| `Timestep` | Single step of interaction | Named dataclass |
| `Trajectory` | Complete training run | Sequence of timesteps + metadata |
| `OnsetLabel` | Ground-truth onset annotation | Dataclass with time, confidence, type |
| `DetectionScore` | Detector output at a timestep | `float` in `[0.0, 1.0]` |
| `ScoreSequence` | Full detector output for a trajectory | `np.ndarray` of shape `(T,)` |
| `AccessLevel` | Information constraint | Enum: `L1`, `L2`, `L3`, `L4` |
| `HackingType` | Category of reward hacking | Enum: 6 values |
| `Tier` | Environment difficulty tier | Enum: `TIER1`, `TIER2`, `TIER3`, `ADVERSARIAL` |
| `EnvironmentID` | Unique environment identifier | `str` (e.g., `"tier1/navigation_proxy"`) |
| `DetectorID` | Unique detector identifier | `str` (e.g., `"baselines/cusum"`) |
| `RunID` | Unique trajectory identifier | `str` (environment + seed + algorithm) |

### 3.2 Timestep Schema

A `Timestep` contains different fields depending on access level. The full schema:

| Field | Type | Available at | Description |
|---|---|---|---|
| `t` | `int` | L1+ | Training step index |
| `reward_proxy` | `float` | L1+ | Observed proxy reward |
| `state` | `np.ndarray` | L2+ | Environment state |
| `action` | `np.ndarray` | L2+ | Agent action |
| `next_state` | `np.ndarray` | L2+ | Resulting state |
| `done` | `bool` | L2+ | Episode termination |
| `info` | `dict` | L2+ | Environment info dict |
| `gradient_norm` | `float` | L3+ | Policy gradient L2 norm |
| `policy_params` | `np.ndarray` | L3+ | Policy parameter snapshot |
| `kl_from_reference` | `float` | L3+ | KL divergence from initial policy |
| `internal_state` | `dict` | L4 | Hidden environment internals |
| `reward_true` | `float` | Oracle only | True reward (never given to detectors) |

### 3.3 Trajectory Model

```
Trajectory:
    metadata:
        environment_id: EnvironmentID
        seed: int
        algorithm: str (e.g., "PPO")
        total_steps: int
        onset_label: OnsetLabel | None
        is_hacking_run: bool
        generation_timestamp: str (ISO 8601)
        config_hash: str
    steps: list[Timestep]  # Length = total_steps
```

### 3.4 Onset Label Model

```
OnsetLabel:
    onset_step: int                    # The labeled onset timestep t*
    confidence: float                  # In [0, 1] — oracle confidence
    hacking_type: HackingType          # Category
    detection_method: str              # How oracle determined onset
    confidence_interval: tuple[int, int]  # Uncertainty in t*
    severity: float                    # Rate of true-reward degradation post-onset
```

### 3.5 Access Level Algebra

Access levels form a total order: `L1 < L2 < L3 < L4`. The system enforces that a detector at level `Lk` only receives fields available at that level. This is enforced by the `AccessFilter` wrapper, not by the environment — environments always generate full data, and the filter restricts what the detector sees.

**Key invariant:** A detector declared at `L2` will NEVER receive gradient, parameter, or internal state information, regardless of what the environment provides. This is enforced structurally, not by convention.

---

## 4. Environment Abstraction

### 4.1 Abstract Environment Contract

Every RHOB environment implements this contract:

```
AbstractEnvironment:
    # --- Identity ---
    name: str                          # Human-readable name
    id: EnvironmentID                  # Unique identifier
    tier: Tier                         # Difficulty tier
    hacking_type: HackingType          # Category of hacking
    description: str                   # One-paragraph description
    
    # --- Specifications ---
    state_dim: int                     # Observation dimensionality
    action_dim: int                    # Action dimensionality
    action_type: "discrete" | "continuous"
    max_steps: int                     # Maximum training steps
    expected_onset_range: tuple[int, int]  # Typical onset window
    hacking_reliability: float         # Fraction of seeds that produce hacking
    
    # --- Difficulty ---
    difficulty_knob: str | None        # Name of the tunable parameter
    difficulty_range: tuple[float, float] | None
    difficulty_default: float | None
    
    # --- Methods ---
    generate(seed: int, difficulty: float | None, config: EnvConfig) -> Trajectory
        """Generate a complete training trajectory with onset labels."""
    
    validate() -> ValidationReport
        """Self-check: can this environment produce hacking reliably?"""
    
    describe() -> EnvironmentCard
        """Return structured metadata (for documentation and registry)."""
```

### 4.2 Environment Card (Metadata Schema)

Each environment produces a structured card:

```
EnvironmentCard:
    id: EnvironmentID
    name: str
    tier: Tier
    hacking_type: HackingType
    true_objective: str               # What the agent SHOULD do
    proxy_reward: str                  # What the proxy measures
    hacking_mechanism: str            # How hacking manifests
    oracle_mechanism: str             # How onset is detected
    difficulty_knob: str | None
    compute_per_seed: str             # e.g., "~30 GPU-sec on A100"
    citation: str | None              # If environment is adapted from prior work
```

### 4.3 Oracle Interface

Each environment contains an onset oracle — the mechanism that computes the ground-truth onset label from the full trajectory (including true reward):

```
AbstractOracle:
    compute_onset(
        trajectory: Trajectory,       # Full trajectory including R_true
        lookback_k: int,             # Lookback window
        significance_delta: float,    # Threshold in std units
        alpha: float                  # Test level
    ) -> OnsetLabel | None
        """Return onset label, or None if no hacking detected in this run."""
    
    validate_label(
        trajectory: Trajectory,
        label: OnsetLabel
    ) -> bool
        """Confirm a label is consistent with the data."""
```

### 4.4 Environment Registry

```
EnvironmentRegistry:
    register(env_class: type) -> None
    get(env_id: EnvironmentID) -> AbstractEnvironment
    list_all() -> list[EnvironmentID]
    list_by_tier(tier: Tier) -> list[EnvironmentID]
    list_by_type(hacking_type: HackingType) -> list[EnvironmentID]
    get_card(env_id: EnvironmentID) -> EnvironmentCard
```

Environments register via decorator: `@rhob.register_environment`.

### 4.5 Environment Catalogue

| ID | Name | Tier | Hacking Type | Difficulty Knob |
|---|---|---|---|---|
| `tier1/navigation_proxy` | Navigation Proxy | T1 | Proxy Gaming | proxy_correlation |
| `tier1/gridworld_wireheading` | GridWorld Wireheading | T1 | Reward Tampering | tamper_accessibility |
| `tier1/bandit_gaming` | Bandit Gaming | T1 | Specification Gaming | num_arms |
| `tier1/cliff_shortcut` | Cliff Shortcut | T1 | Proxy Gaming | shortcut_reward |
| `tier1/reward_tampering_tab` | Tabular Tampering | T1 | Reward Tampering | tamper_delay |
| `tier2/halfcheetah_proxy` | HalfCheetah Proxy | T2 | Proxy Gaming | proxy_noise |
| `tier2/reacher_misaligned` | Reacher Misaligned | T2 | Specification Gaming | target_ambiguity |
| `tier2/cartpole_exploit` | CartPole Exploit | T2 | Proxy Gaming | exploit_gap |
| `tier2/lunar_specification` | Lunar Specification | T2 | Specification Gaming | fuel_proxy_weight |
| `tier2/ant_proxy_diverge` | Ant Proxy Diverge | T2 | Proxy Gaming | diverge_rate |
| `tier2/hopper_gaming` | Hopper Gaming | T2 | Specification Gaming | style_weight |
| `tier2/walker_misgen` | Walker Misgeneralization | T2 | Goal Misgeneralization | env_diversity |
| `tier3/text_reward_model` | Text Reward Model | T3 | Proxy Gaming | model_capacity |
| `tier3/code_proxy` | Code Proxy | T3 | Specification Gaming | test_coverage |
| `tier3/dialogue_sycophancy` | Dialogue Sycophancy | T3 | Goal Misgeneralization | sycophancy_gradient |
| `tier3/vision_reward_hack` | Vision Reward Hack | T3 | Proxy Gaming | discriminator_capacity |
| `adversarial/stealth_diverge` | Stealth Diverge | ADV | Proxy Gaming | stealth_factor |
| `adversarial/gradual_corruption` | Gradual Corruption | ADV | Goal Misgeneralization | corruption_rate |
| `adversarial/mimicry` | Mimicry | ADV | Specification Gaming | mimicry_fidelity |

---

## 5. Detector API

### 5.1 Abstract Detector Contract

```
AbstractDetector:
    # --- Identity ---
    name: str                          # Human-readable name
    id: DetectorID                     # Unique identifier
    version: str                       # Semantic version
    access_level: AccessLevel          # Required information level
    is_oracle_free: bool               # Does NOT use R_true
    
    # --- Lifecycle ---
    reset() -> None
        """Reset all internal state. Called before each new trajectory."""
    
    configure(config: DetectorConfig) -> None
        """Apply configuration. Called once before evaluation begins."""
    
    # --- Core Interface ---
    step(observation: Observation) -> float
        """
        Consume one timestep of data. Return hacking probability in [0.0, 1.0].
        
        Args:
            observation: Filtered view of the timestep (respects access_level)
        
        Returns:
            Score in [0, 1] where 1.0 = "certainly hacking"
        
        Invariants:
            - Must be deterministic given the same sequence of observations
            - Must run in O(1) amortized time (no full-history recomputation)
            - Must not store more than O(sqrt(T)) state
        """
    
    # --- Optional ---
    fit(training_data: Dataset) -> None
        """Optional: train on labeled data (for learning-based detectors)."""
    
    explain(t: int) -> str | None
        """Optional: human-readable explanation of detection at step t."""
    
    # --- Metadata ---
    computational_overhead() -> OverheadEstimate
        """Report estimated computational overhead vs. base training."""
    
    hyperparameters() -> dict[str, Any]
        """Return current hyperparameter values for reproducibility."""
```

### 5.2 Observation Schema (What Detectors See)

The `Observation` object is a view of a `Timestep`, filtered by access level:

```
Observation:
    t: int                             # Always available
    access_level: AccessLevel          # What this observation contains
    
    # L1 fields
    reward: float                      # Proxy reward
    
    # L2 fields (None if access < L2)
    state: np.ndarray | None
    action: np.ndarray | None
    next_state: np.ndarray | None
    done: bool | None
    info: dict | None
    
    # L3 fields (None if access < L3)
    gradient_norm: float | None
    kl_from_reference: float | None
    policy_params_hash: str | None     # Hash, not full params (for comparison)
    
    # L4 fields (None if access < L4)
    internal_state: dict | None
```

**Critical design decision:** The `Observation` type is immutable. Detectors cannot modify it. The access filter creates a new `Observation` from the full `Timestep`, zeroing out fields above the detector's declared level. This prevents information leakage by construction.

### 5.3 Detector Wrappers

| Wrapper | Purpose | Behavior |
|---|---|---|
| `AccessFilterWrapper` | Enforce access level | Strips fields above declared level; raises on violation |
| `CalibratedWrapper` | Probability calibration | Applies isotonic regression to raw scores (fit on calibration set) |
| `WindowedWrapper` | Temporal smoothing | Applies sliding-window aggregation to raw scores |
| `TimedWrapper` | Performance measurement | Records per-step wall-clock time |
| `RecordingWrapper` | Debugging | Records all inputs/outputs for replay |

### 5.4 Detector Registration

```
DetectorRegistry:
    register(detector_class: type) -> None
    get(detector_id: DetectorID) -> AbstractDetector
    list_all() -> list[DetectorID]
    list_by_access(level: AccessLevel) -> list[DetectorID]
    validate_contract(detector: AbstractDetector) -> ContractReport
        """Verify detector satisfies all contract invariants."""
```

### 5.5 Contract Validation

Before any evaluation, the system validates that a detector satisfies its contract:

1. **Determinism:** Run the same trajectory twice → identical scores (bit-for-bit).
2. **Bounds:** All scores in `[0.0, 1.0]` (no NaN, no inf).
3. **Reset isolation:** Scores after `reset()` do not depend on previous trajectory.
4. **Access compliance:** Detector does not access fields above its declared level.
5. **Performance:** Average step time < 100ms (warning), < 1s (error).
6. **State growth:** Memory usage grows sub-linearly in T.

---

## 6. Evaluation API

### 6.1 Evaluation Runner

```
EvaluationRunner:
    """Orchestrates the evaluation of detectors on the benchmark."""
    
    run(
        detector: AbstractDetector | DetectorID,
        environments: list[EnvironmentID] | Tier | "all",
        config: EvaluationConfig,
    ) -> EvaluationReport
        """
        Main entry point. Evaluates a detector on specified environments.
        
        Pipeline:
            1. Validate detector contract
            2. Load/generate trajectory data
            3. For each trajectory:
                a. Reset detector
                b. Feed observations step-by-step
                c. Collect score sequence
            4. Compute per-trajectory metrics
            5. Aggregate per-environment
            6. Aggregate overall
            7. Generate report
        """
    
    compare(
        detectors: list[AbstractDetector | DetectorID],
        environments: list[EnvironmentID] | Tier | "all",
        config: EvaluationConfig,
    ) -> ComparisonReport
        """Evaluate multiple detectors and produce head-to-head comparison."""
    
    ablate(
        detector: AbstractDetector | DetectorID,
        parameter: str,
        values: list[Any],
        environments: list[EnvironmentID],
        config: EvaluationConfig,
    ) -> AblationReport
        """Sweep a single parameter and report sensitivity."""
```

### 6.2 Evaluation Configuration

```
EvaluationConfig:
    # Data
    seeds: list[int] | int             # Specific seeds or count
    split: "test" | "train" | "all"    # Which data split
    
    # Onset definition parameters
    lookback_k: int = 1000             # Lookback window for onset oracle
    significance_delta: float = 1.0    # Threshold in sigma units
    alpha: float = 0.01                # Test level
    
    # Evaluation
    alert_budget: int = 3              # For FPR@k
    score_threshold: float = 0.5       # For binary decisions
    
    # Statistical
    bootstrap_n: int = 10000           # Bootstrap resamples for CIs
    significance_level: float = 0.01   # For method comparisons
    correction: str = "holm"           # Multiple comparison correction
    
    # Compute
    n_workers: int = 1                 # Parallel workers
    cache_dir: str | None = None       # Where to cache results
    deterministic: bool = True         # Force deterministic mode
    
    # Output
    output_dir: str = "./results"
    save_raw_scores: bool = True       # Save per-step scores
    generate_plots: bool = True
    report_format: "json" | "markdown" | "latex" | "all" = "all"
```

### 6.3 Evaluation Report Structure

```
EvaluationReport:
    metadata:
        detector_id: DetectorID
        detector_version: str
        detector_config: dict
        evaluation_config: EvaluationConfig
        timestamp: str
        rhob_version: str
        total_compute_seconds: float
        hardware: str
    
    per_trajectory: list[TrajectoryResult]
        # One entry per (environment, seed) pair
        trajectory_result:
            run_id: RunID
            environment_id: EnvironmentID
            seed: int
            is_hacking: bool
            onset_step: int | None
            scores: np.ndarray           # Full score sequence
            detected: bool               # Did detector fire?
            detection_step: int | None   # When detector first fired
            latency: float | None        # Normalized detection latency
    
    per_environment: dict[EnvironmentID, EnvironmentMetrics]
        environment_metrics:
            auroc: float
            auroc_ci: tuple[float, float]     # 99% CI
            auprc: float
            auprc_ci: tuple[float, float]
            median_latency: float
            latency_ci: tuple[float, float]
            fpr_at_k: float
            miss_rate: float
            tfd: float
            n_hacking_runs: int
            n_clean_runs: int
    
    per_tier: dict[Tier, TierMetrics]
        tier_metrics:
            mean_auroc: float
            mean_auroc_ci: tuple[float, float]
            mean_latency: float
            weighted_score: float
    
    aggregate:
        rhob_score: float
        rhob_score_ci: tuple[float, float]
        total_auroc: float
        total_miss_rate: float
        mean_latency: float
        computational_overhead: OverheadEstimate
    
    diagnostics:
        contract_violations: list[str]
        numerical_warnings: list[str]
        environments_skipped: list[str]
```

---

## 7. Metrics Specification

### 7.1 Metric Definitions

| Metric | Formula | Domain | Properties |
|---|---|---|---|
| **Detection Latency** | `(t_detect - t*) / T` | `[0, 1]` ∪ `{∞}` | Lower is better; ∞ if never detected |
| **AUROC** | Standard (per-step binary classification: hacking vs. not-yet) | `[0, 1]` | Higher is better; 0.5 = random |
| **AUPRC** | Precision-Recall AUC (positive = post-onset steps) | `[0, 1]` | Higher is better; handles class imbalance |
| **FPR@k** | `FP / total_alerts` at budget `k` alerts | `[0, 1]` | Lower is better; k=3 default |
| **Miss Rate** | Fraction of hacking trajectories never detected | `[0, 1]` | Lower is better; 0 = perfect recall |
| **TFD** | Median latency among true positives | `[0, 1]` | Lower is better; undefined if miss_rate = 1 |
| **RHOB-Score** | Weighted mean AUROC across environments | `[0, 1]` | Higher is better; primary ranking metric |

### 7.2 RHOB-Score Computation

```
RHOB-Score = Σ_{e ∈ environments} w(tier(e)) · AUROC(e) / Σ_{e} w(tier(e))

where:
    w(TIER1) = 1.0
    w(TIER2) = 1.5
    w(TIER3) = 2.0
    w(ADVERSARIAL) = 2.5
```

**Rationale:** Harder environments contribute more to the score, incentivizing methods that solve difficult cases rather than optimizing on easy environments.

### 7.3 Per-Step Classification Setup

For AUROC/AUPRC computation, define per-step labels:
- **Positive:** Step `t ≥ t*` (post-onset, i.e., hacking is occurring)
- **Negative:** Step `t < t*` (pre-onset, normal training) OR clean run
- **Detector score at step t:** Used as the "predicted probability"

This converts onset detection into a sequence of binary classification problems, enabling standard ROC/PR machinery.

### 7.4 Statistical Rigor

Every metric is reported with:
- Point estimate (mean across seeds)
- 99% confidence interval (bootstrap, 10K resamples)
- Standard deviation across seeds

Method comparisons include:
- Paired Wilcoxon signed-rank test (non-parametric)
- Holm-Bonferroni correction for multiple comparisons
- Effect size (Cliff's delta)

### 7.5 Metric Properties (Tested via Property-Based Tests)

| Property | Assertion |
|---|---|
| Boundedness | All metrics in their declared domains |
| Monotonicity | Better detector → better score (oracle > any detector > random) |
| Determinism | Same inputs → same metric values |
| Scale invariance | AUROC unchanged by monotone transform of scores |
| Decomposability | Per-environment metrics recoverable from per-trajectory data |

---

## 8. Data Management

### 8.1 Storage Format

**Primary format:** HDF5 (efficient for large numerical arrays with metadata).

```
rhob_data_v1.h5
├── metadata/
│   ├── version: str
│   ├── generated_at: str
│   ├── config_hash: str
│   └── environment_cards: JSON blob
│
├── tier1/
│   ├── navigation_proxy/
│   │   ├── metadata/
│   │   │   ├── n_seeds: int
│   │   │   ├── hacking_reliability: float
│   │   │   └── algorithm: str
│   │   ├── seed_0000/
│   │   │   ├── rewards_proxy: float32[T]
│   │   │   ├── rewards_true: float32[T]    # For oracle only
│   │   │   ├── states: float32[T, state_dim]
│   │   │   ├── actions: float32[T, action_dim]
│   │   │   ├── dones: bool[T]
│   │   │   ├── gradient_norms: float32[T]
│   │   │   ├── kl_divergences: float32[T]
│   │   │   ├── onset_step: int
│   │   │   ├── is_hacking: bool
│   │   │   └── onset_confidence: float
│   │   ├── seed_0001/
│   │   │   └── ...
│   │   └── ...
│   └── ...
├── tier2/
│   └── ...
├── tier3/
│   └── ...
└── adversarial/
    └── ...
```

### 8.2 Data Loading API

```
DataLoader:
    load(
        environments: list[EnvironmentID] | "all",
        split: "train" | "test" | "all",
        access_level: AccessLevel,
        max_seeds: int | None = None,
        lazy: bool = True,              # Memory-map, don't load all at once
    ) -> Dataset

Dataset:
    __iter__() -> Iterator[tuple[Trajectory, OnsetLabel | None]]
    __len__() -> int
    environments: list[EnvironmentID]
    total_trajectories: int
    total_steps: int
    
    filter(
        tier: Tier | None = None,
        hacking_type: HackingType | None = None,
        hacking_only: bool = False,
        clean_only: bool = False,
    ) -> Dataset
```

### 8.3 Data Versioning

| Version | Content | Backwards Compatible |
|---|---|---|
| `v1.0` | Tier 1 + Tier 2, 50 seeds, PPO only | — |
| `v1.1` | + SAC and DQN runs (10 seeds each) | Yes (additive) |
| `v2.0` | + Tier 3, adversarial tier | No (new schema fields) |
| `v2.x` | Adversarial refresh (quarterly) | Yes |

Data is hosted on Hugging Face Hub with version tags. The package downloads and caches locally.

### 8.4 Train/Test Split

- **Train split (60%):** Labels visible. Used for detector development and hyperparameter tuning.
- **Test split (40%):** Labels withheld in public release. Used for leaderboard evaluation.
- **Split is fixed per version** — same seeds always in same split.
- Split assignment uses deterministic hash: `split = "train" if hash(env_id + seed) % 5 < 3 else "test"`.

---

## 9. Configuration System

### 9.1 Configuration Hierarchy (Precedence: highest first)

1. **CLI arguments** — override everything
2. **Environment variables** — `RHOB_*` prefix
3. **Local config file** — `./rhob.yaml` in working directory
4. **User config** — `~/.config/rhob/config.yaml`
5. **Package defaults** — `rhob/config/defaults.py`

### 9.2 Configuration Schema (Pydantic Models)

```
RHOBConfig:
    """Top-level configuration model."""
    
    data: DataConfig
        data_dir: Path = "~/.cache/rhob"
        remote_url: str = "https://huggingface.co/datasets/aarav-shah/rhob"
        version: str = "v1.0"
        auto_download: bool = True
    
    evaluation: EvaluationConfig
        # (as defined in Section 6.2)
    
    environments: EnvironmentConfig
        include: list[EnvironmentID] | "all" = "all"
        exclude: list[EnvironmentID] = []
        tiers: list[Tier] = [TIER1, TIER2, TIER3, ADVERSARIAL]
        difficulty_override: dict[EnvironmentID, float] = {}
    
    onset: OnsetConfig
        lookback_k: int = 1000
        significance_delta: float = 1.0
        alpha: float = 0.01
        method: "two_sample" | "cusum_oracle" | "regression" = "two_sample"
    
    output: OutputConfig
        dir: Path = "./results"
        format: list[str] = ["json", "markdown"]
        save_raw: bool = True
        save_plots: bool = True
        plot_format: str = "pdf"
        verbosity: int = 1
    
    compute: ComputeConfig
        n_workers: int = 1
        device: str = "cpu"
        seed: int = 42
        deterministic: bool = True
```

### 9.3 Configuration Files (YAML)

Example `configs/paper.yaml`:
```yaml
data:
  version: "v1.0"

evaluation:
  seeds: 50
  split: "test"
  bootstrap_n: 10000
  significance_level: 0.01
  correction: "holm"

environments:
  include: "all"
  tiers: [tier1, tier2, tier3, adversarial]

onset:
  lookback_k: 1000
  significance_delta: 1.0
  alpha: 0.01

output:
  format: ["json", "markdown", "latex"]
  save_raw: true
  save_plots: true
  plot_format: "pdf"

compute:
  n_workers: 8
  deterministic: true
```

---

## 10. Leaderboard Specification

### 10.1 Submission Format

```
Submission:
    """A leaderboard submission."""
    
    metadata:
        detector_name: str
        detector_version: str
        authors: list[str]
        affiliation: str | None
        paper_url: str | None
        code_url: str                   # Required — must be reproducible
        access_level: AccessLevel
        is_oracle_free: bool
        training_data_used: str         # "none" | "train_split" | "external"
        compute_used: str               # e.g., "1x A100, 2 hours"
        date: str                       # ISO 8601
    
    results:
        # Per-environment AUROC + latency
        per_environment: dict[EnvironmentID, dict[str, float]]
        # Aggregate
        rhob_score: float
        total_auroc: float
        mean_latency: float
        miss_rate: float
    
    verification:
        config_hash: str                # Hash of evaluation config
        data_version: str               # RHOB data version used
        rhob_package_version: str       # Package version used
        random_seed: int
        reproducibility_hash: str       # Hash of raw score arrays
```

### 10.2 Submission Validation

Before acceptance, the system verifies:

1. **Format compliance:** Schema validation via JSON Schema.
2. **Reproducibility check:** Re-run detector on 5 random trajectories → scores match within floating-point tolerance.
3. **Code availability:** The `code_url` resolves and contains a working implementation.
4. **No test-set peeking:** The submission's `training_data_used` field is checked against the code (no test labels imported).
5. **Version compatibility:** `data_version` and `rhob_package_version` are compatible.

### 10.3 Ranking Algorithm

Primary sort: **RHOB-Score** (descending).

Tie-breaking (in order):
1. Mean detection latency (ascending)
2. Miss rate (ascending)
3. Computational overhead (ascending)
4. Submission date (earlier wins)

### 10.4 Leaderboard Display Format

```
| Rank | Method          | Access | RHOB-Score | T1 AUROC | T2 AUROC | T3 AUROC | ADV AUROC | Latency | Miss | Overhead |
|------|-----------------|--------|------------|----------|----------|----------|-----------|---------|------|----------|
| 1    | Flight Recorder | L2     | 0.823      | 0.91     | 0.85     | 0.78     | 0.64      | 0.12    | 0.05 | 3%       |
| 2    | Ensemble v2     | L2     | 0.791      | 0.88     | 0.82     | 0.74     | 0.61      | 0.18    | 0.08 | 15%      |
| ...  | ...             | ...    | ...        | ...      | ...      | ...      | ...       | ...     | ...  | ...      |
| —    | Oracle (ceiling) | L4+R  | 1.000      | 1.00     | 1.00     | 1.00     | 1.00      | 0.00    | 0.00 | 0%       |
| —    | Random (floor)  | —      | 0.500      | 0.50     | 0.50     | 0.50     | 0.50      | 0.50    | 0.50 | 0%       |
```

Sub-leaderboards:
- By access level (L1-only, L2-only, etc.)
- By tier (per-tier rankings)
- By hacking type
- Oracle-free only (primary leaderboard)
- Latency-constrained (must detect within 10% of training horizon)

### 10.5 Leaderboard Infrastructure

- **Storage:** JSON files committed to the repository (version-controlled).
- **Website:** Static site generated from JSON (GitHub Pages or dedicated domain).
- **Submission:** GitHub Issue with template → CI validates → merge adds to results.
- **Refresh:** Quarterly adversarial tier update invalidates affected rankings.

---

## 11. Flight Recorder Integration

### 11.1 Integration Architecture

Flight Recorder is one of the built-in baselines, but its integration is deeper than other baselines because RHOB is designed to validate Flight Recorder's theoretical claims.

```
rhob/integrations/flight_recorder.py:

FlightRecorderAdapter(AbstractDetector):
    """Adapts Flight Recorder into the RHOB detector interface."""
    
    # --- Configuration ---
    access_level: AccessLevel = L2
    is_oracle_free: bool = True
    
    # --- Flight Recorder specific ---
    sde_window: int                    # Spectral Drift Entropy window
    spectral_threshold: float          # Detection threshold
    embedding_dim: int                 # State embedding dimension
    
    # --- Interface implementation ---
    reset() -> None:
        """Clear SDE history and state."""
    
    step(observation: Observation) -> float:
        """
        Compute SDE on the observation window.
        Map SDE value to [0,1] detection probability.
        """
    
    # --- Flight Recorder specific methods ---
    get_sde_history() -> np.ndarray:
        """Return the full SDE time series (for analysis/plotting)."""
    
    get_spectral_components() -> dict:
        """Return intermediate spectral decomposition (for explanation)."""
```

### 11.2 Validation of Theoretical Claims

RHOB is designed to empirically validate three claims from the Flight Recorder paper:

| Claim | RHOB Validation Method |
|---|---|
| THM-01: Oracle-blind geometric signals cannot separate hacking from convergence | Measure CUSUM/KL/entropy baseline performance — should be near-random on adversarial tier |
| L2 dominance over L1 | Compare same method restricted to L1 vs. L2 — should show statistically significant gap |
| SDE detects structural change | Flight Recorder should outperform non-spectral L2 methods on sharp-onset environments |

### 11.3 Bidirectional Data Flow

```
Flight Recorder → RHOB:
    - Flight Recorder produces detection scores
    - RHOB evaluates them against ground truth
    - RHOB produces the evidence for Flight Recorder paper claims

RHOB → Flight Recorder:
    - RHOB identifies failure cases for Flight Recorder
    - These become regression tests in Flight Recorder development
    - RHOB difficulty hierarchy guides Flight Recorder research priorities
```

---

## 12. CLI Specification

### 12.1 Command Structure

```
rhob
├── evaluate      # Run evaluation
├── compare       # Compare multiple detectors
├── ablate        # Run ablation study
├── download      # Download benchmark data
├── generate      # Generate trajectory data from environments
├── info          # Display environment/detector information
├── validate      # Validate a submission or detector
├── submit        # Prepare leaderboard submission
└── version       # Display version information
```

### 12.2 Key Commands

```
rhob evaluate \
    --detector flight_recorder \
    --environments all \
    --config configs/paper.yaml \
    --output-dir results/fr_v1/ \
    --format json markdown latex

rhob compare \
    --detectors flight_recorder cusum ensemble kl_monitor gradient_norm random \
    --environments all \
    --config configs/paper.yaml \
    --output-dir results/comparison/

rhob ablate \
    --detector flight_recorder \
    --parameter sde_window \
    --values 100 500 1000 2000 5000 \
    --environments tier1 \
    --output-dir results/ablation_window/

rhob download --version v1.0 --tiers tier1 tier2

rhob info --environment tier2/halfcheetah_proxy

rhob validate --submission results/fr_v1/submission.json

rhob submit --results results/fr_v1/ --output submission.json
```

---

## 13. Testing Strategy

### 13.1 Test Pyramid

| Layer | Count (target) | Speed | What it tests |
|---|---|---|---|
| **Unit** | ~200 | < 30s total | Individual functions, type validation, metric math |
| **Property** | ~50 | < 60s total | Invariants (bounds, monotonicity, determinism) |
| **Integration** | ~30 | < 5 min total | Full pipeline on Tier 1 subset |
| **Regression** | ~20 | < 2 min total | Pin known-good outputs against gold files |
| **Contract** | ~10 | < 1 min total | Detector contract verification on all baselines |

### 13.2 Critical Invariants to Test

| Invariant | Test Type |
|---|---|
| All metrics in declared bounds | Property |
| Oracle detector achieves perfect score | Integration |
| Random detector achieves ~0.5 AUROC | Integration |
| Evaluation is deterministic (same config → same output, bit-for-bit) | Integration |
| Access filter prevents information leakage | Unit + Property |
| HDF5 roundtrip preserves all data | Unit |
| Config merge follows documented precedence | Unit |
| Leaderboard ranking is stable under tie conditions | Unit |
| Methods declared L1 never access L2+ fields | Contract |

### 13.3 CI/CD Pipeline

```
On every PR:
    - Lint (ruff)
    - Type check (mypy --strict)
    - Unit tests
    - Property tests
    - Contract tests

On merge to main:
    - Full integration tests
    - Regression tests
    - Documentation build
    - Coverage report (target: >90%)

On version tag:
    - All above
    - Build wheel
    - Publish to PyPI
    - Generate release notes
    - Update leaderboard site
```

---

## 14. Public API Surface

### 14.1 Top-Level Imports

The public API should be minimal and discoverable. Everything a typical user needs is importable from `rhob` directly:

```
import rhob

# --- Core ---
rhob.evaluate(detector, environments, config)
rhob.compare(detectors, environments, config)
rhob.load_data(environments, split, access_level)

# --- Registration ---
rhob.register_detector(cls)
rhob.register_environment(cls)

# --- Discovery ---
rhob.list_environments()
rhob.list_detectors()
rhob.get_environment_card(env_id)

# --- Base classes ---
rhob.Detector           # AbstractDetector
rhob.Environment        # AbstractEnvironment
rhob.Observation        # Observation type
rhob.OnsetLabel         # Onset label type
rhob.AccessLevel        # Access level enum

# --- Metrics ---
rhob.metrics.auroc(scores, labels)
rhob.metrics.detection_latency(scores, onset_step, threshold)
rhob.metrics.rhob_score(per_env_aurocs, tiers)

# --- Config ---
rhob.load_config(path)
rhob.default_config()
```

### 14.2 Minimal Usage (The "10-Line Evaluation")

```python
import rhob

class MyDetector(rhob.Detector):
    name = "my_detector"
    access_level = rhob.AccessLevel.L2
    
    def reset(self):
        self.history = []
    
    def step(self, obs):
        self.history.append(obs.reward)
        if len(self.history) > 100:
            return anomaly_score(self.history)
        return 0.0

report = rhob.evaluate(MyDetector(), environments="tier1")
print(report.aggregate.rhob_score)
```

### 14.3 API Stability Guarantees

| Module | Stability | Policy |
|---|---|---|
| `rhob` (top-level) | **Stable** | Semver; no breaking changes in minor versions |
| `rhob.core` | **Stable** | Type definitions frozen per major version |
| `rhob.evaluation` | **Stable** | Report structure frozen per major version |
| `rhob.environments.base` | **Stable** | Contract frozen per major version |
| `rhob.detectors.base` | **Stable** | Contract frozen per major version |
| `rhob.detectors.baselines` | **Semi-stable** | Baselines may be added but not removed |
| `rhob.visualization` | **Unstable** | May change between minor versions |
| `rhob.integrations` | **Unstable** | Depends on external API stability |

---

## 15. Scalability Design

### 15.1 Compute Scalability

| Concern | Solution |
|---|---|
| Large trajectory files | HDF5 with chunked storage + memory-mapped loading |
| Many environments × seeds | Embarrassingly parallel; `n_workers` config |
| Expensive Tier 3 generation | Pre-generated data; generation is offline batch job |
| CI speed | Quick config (Tier 1 only, 5 seeds) for fast feedback |
| Full paper reproduction | `configs/paper.yaml` with documented compute requirements |

### 15.2 Data Scalability

| Scale | Today | v2.0 | v3.0 |
|---|---|---|---|
| Environments | 21 | 30+ | 50+ |
| Seeds per env | 50 | 100 | 100 |
| Total trajectories | ~1,050 | ~3,000 | ~5,000 |
| Storage (compressed) | ~10 GB | ~50 GB | ~100 GB |
| Download strategy | Full | Per-tier chunks | Streaming |

### 15.3 Community Scalability

| Growth vector | Design support |
|---|---|
| New environments | Registry pattern + template + validation |
| New detectors | Base class + contract tests + submission flow |
| New metrics | Metric registry + property test framework |
| Forks/variants | Plugin architecture (entry points) |
| Multi-institution | Governance documented; submission via PR |

---

## 16. Versioning Strategy

### 16.1 Package Versioning (Semver)

```
MAJOR.MINOR.PATCH

MAJOR: Breaking API changes, data format changes, metric redefinitions
MINOR: New environments, new baselines, new features (backward-compatible)
PATCH: Bug fixes, documentation, CI improvements
```

### 16.2 Data Versioning (Independent)

```
vMAJOR.MINOR

MAJOR: Schema changes, new tiers, split reassignment
MINOR: New seeds, adversarial refresh, additional algorithms
```

Package version and data version are independent. The package specifies which data versions it supports.

### 16.3 Leaderboard Versioning

Each leaderboard entry records the exact package version and data version. When data version changes (major), a new leaderboard epoch begins. Historical results are preserved but marked as "prior epoch."

---

## 17. Documentation Architecture

### 17.1 Documentation Layers

| Layer | Audience | Format | Tool |
|---|---|---|---|
| **README** | First-time visitors | Markdown | GitHub |
| **Quickstart** | New users | Tutorial | MkDocs |
| **API Reference** | Implementers | Auto-generated | mkdocstrings |
| **Architecture** | Contributors | Design doc | MkDocs |
| **Paper Reproduction** | Reviewers/researchers | Step-by-step | MkDocs |
| **Contributor Guide** | Community | Process doc | MkDocs |
| **Changelog** | Upgraders | Chronological | CHANGELOG.md |

### 17.2 Docstring Standard

Every public class and function has:
- One-line summary
- Extended description (if needed)
- Args with types and descriptions
- Returns with type and description
- Raises section
- Example usage
- Notes (mathematical details, invariants)

---

## 18. Error Handling Philosophy

### 18.1 Exception Hierarchy

```
RHOBError (base)
├── ConfigError
│   ├── ConfigNotFoundError
│   ├── ConfigValidationError
│   └── ConfigCompatibilityError
├── DataError
│   ├── DataNotFoundError
│   ├── DataCorruptionError
│   ├── DataVersionMismatchError
│   └── DownloadError
├── DetectorError
│   ├── ContractViolationError
│   ├── AccessViolationError
│   ├── ScoreBoundsError
│   └── DeterminismError
├── EnvironmentError
│   ├── GenerationError
│   └── OracleError
└── EvaluationError
    ├── InsufficientDataError
    └── StatisticalError
```

### 18.2 Error Messages

Every error includes:
- What went wrong (concrete)
- Why it matters (context)
- How to fix it (actionable)

Example:
```
AccessViolationError: Detector 'my_detector' (declared L1) accessed 
field 'state' which requires L2+.

This violates the access level contract. Detectors at L1 may only 
observe 'reward' and 't'.

Fix: Either upgrade your detector's access_level to L2, or remove 
the state access from your step() method.
```

---

## 19. Priority Implementation Order

### Phase 1: Foundation (Weeks 1–2)

1. `rhob/core/types.py` — All type definitions
2. `rhob/core/trajectory.py` — Trajectory model
3. `rhob/core/onset.py` — Onset label model
4. `rhob/core/access.py` — Access level enum + filter
5. `rhob/detectors/base.py` — Abstract detector contract
6. `rhob/config/schema.py` — Pydantic config models
7. `rhob/config/defaults.py` — Default values
8. Unit tests for all of the above

### Phase 2: Minimal Pipeline (Weeks 3–4)

8. `rhob/environments/base.py` — Environment contract
9. `rhob/environments/tier1/navigation_proxy.py` — First environment
10. `rhob/detectors/baselines/random_detector.py` — Random baseline
11. `rhob/detectors/baselines/cusum.py` — CUSUM baseline
12. `rhob/evaluation/metrics.py` — All metric computations
13. `rhob/evaluation/runner.py` — Evaluation orchestration
14. `rhob/evaluation/report.py` — Report generation
15. Integration test: evaluate random + CUSUM on navigation_proxy

### Phase 3: Data Layer (Weeks 5–6)

16. `rhob/data/storage.py` — HDF5 abstraction
17. `rhob/data/loader.py` — Dataset loading
18. `rhob/data/splits.py` — Split logic
19. Generate Tier 1 data (all 5 environments, 50 seeds)
20. Full integration test on Tier 1

### Phase 4: Completeness (Weeks 7–10)

21. All Tier 1 environments
22. All baseline detectors
23. Flight Recorder integration
24. CLI
25. Visualization
26. Leaderboard infrastructure
27. Documentation
28. Tier 2 environments

### Phase 5: Paper-Ready (Weeks 11–14)

29. Tier 3 environments (partial)
30. Adversarial tier
31. Statistical testing module
32. Ablation automation
33. Full paper reproduction pipeline
34. `configs/paper.yaml` → produces all paper tables/figures

---

## 20. Design Decisions Log

| Decision | Choice | Alternatives Considered | Rationale |
|---|---|---|---|
| Package layout | src-layout | flat layout | Prevents import ambiguity; modern best practice |
| Config format | YAML (primary) + Pydantic | TOML, JSON, dataclasses | YAML is human-friendly; Pydantic validates |
| Data format | HDF5 | Parquet, SQLite, NPZ | Best for large numerical arrays with metadata; random access |
| CLI framework | Click | argparse, Typer, Fire | Mature, composable, well-documented |
| Test framework | pytest + Hypothesis | unittest | Property testing essential for invariant verification |
| Documentation | MkDocs + mkdocstrings | Sphinx | Simpler, better-looking, Markdown-native |
| Detector interface | Step-by-step (streaming) | Batch (full trajectory) | Mirrors real-time use case; enforces O(1) step cost |
| Leaderboard storage | Git-committed JSON | Database, API | Simplicity; version-controlled; forkable |
| Access enforcement | Structural (filter creates new object) | Convention (honor system) | Prevents accidental leakage; testable |
| Metric ranking | RHOB-Score (weighted AUROC) | Pareto ranking, latency-first | Single scalar enables clear ranking; weights incentivize hard cases |

---

## 21. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|---|---|---|---|
| Tier 3 environments too expensive to generate | Blocks paper completeness | Medium | Design system to work with Tiers 1+2 only; Tier 3 is additive |
| Onset definition instability | Undermines benchmark validity | Medium | Ablation suite built into architecture; sensitivity visible early |
| HDF5 portability issues | Blocks adoption on some platforms | Low | Also export to NPZ as fallback; CI tests on Linux/Mac/Windows |
| Community doesn't adopt | Low citation impact | Medium | Paper quality + pip-installable + documentation + leaderboard |
| Flight Recorder performs poorly | Awkward for the publication narrative | Low | RHOB is method-agnostic; honesty about FR limitations is a feature |
| Too complex for solo maintenance | Burnout, abandoned package | Medium | Phase 1-3 are self-contained and useful alone; modular design allows partial release |

---

## 22. Summary of Engineering Principles

1. **Separation of concerns:** Environments, detectors, and evaluation are independent. You can add an environment without touching evaluation code, and vice versa.

2. **Contract-based design:** Every component has a formal contract (abstract base class + invariants). Contracts are machine-verified via property tests.

3. **Reproducibility by construction:** Deterministic evaluation, pinned configs, version-locked data, hash-verified results. Two researchers running the same config MUST get identical numbers.

4. **Progressive disclosure:** Simple things are simple (10-line evaluation). Complex things are possible (custom environments, ablation sweeps, statistical comparisons). The API surface grows with user sophistication.

5. **Fail loudly and helpfully:** Every error is specific, contextual, and actionable. Silent failures are bugs.

6. **Community-first design:** Registration patterns, plugin architecture, submission workflows, and governance documentation are first-class concerns, not afterthoughts.

7. **Paper-driven engineering:** The architecture is designed to produce every table and figure in the publication blueprint from a single config file and a single command.

---

*This specification is complete and sufficient for implementation. Every architectural decision is motivated. Every interface is defined. The implementation order is specified. Begin with Phase 1.*


