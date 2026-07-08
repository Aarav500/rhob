# RHOB Configuration Specification

**Document:** CONFIG_SPEC · **Spec version:** 0.1 · **Status:** Draft for freeze
Source of truth: `src/rhob/config.py`.

Defines the configuration surface for experiments, detectors, environments,
dataset generation, and evaluation, with recommended YAML schemas. A run must be
fully reproducible from its resolved configuration.

**Current state:** M1 implements a single validated `EvaluationConfig` (pydantic,
frozen). The full multi-section, multi-source hierarchy below is the **[Planned]**
frozen target; sections are tagged accordingly.

---

## 1. Configuration hierarchy **[Planned]**

Resolution precedence, highest first:

1. CLI arguments
2. Environment variables (`RHOB_*`)
3. Local config file (`./rhob.yaml`)
4. User config (`~/.config/rhob/config.yaml`)
5. Package defaults

The resolved config is validated, hashed, and recorded in the report for
reproducibility. M1 resolves only an in-code `EvaluationConfig`; file/env layering
is planned.

## 2. Top-level experiment configuration **[Planned]**

```yaml
# rhob.yaml
experiment:
  name: "fr_v1_full"
  output_dir: "./results/fr_v1"
  report_formats: ["json", "markdown"]     # also: latex
  seed: 42

data:                # see §6
detector:            # see §3
environments:        # see §5
evaluation:          # see §4
```

An `ExperimentConfig` bundles the sections and is the single artifact a run is
launched from (`rhob evaluate --config rhob.yaml`).

## 3. Detector configuration **[Planned]**

```yaml
detector:
  id: "baselines/cusum"          # registry id, or an import path
  access_level: "L1"             # must match the detector's declared level
  hyperparameters:
    slack_k: 0.5
    threshold_h: 5.0
    warmup: 10
  training_data_used: "none"     # none | train_split | external
```

`hyperparameters` are passed to the detector's `configure(**kwargs)`; the detector
echoes effective values via `hyperparameters()` into the report.

## 4. Evaluation configuration **[Impl]**

Implemented today as the frozen pydantic `EvaluationConfig`:

```yaml
evaluation:
  score_threshold: 0.5      # [0,1]  binarization / firing threshold
  alert_budget: 3           # >=1    k for FPR@k
  bootstrap_n: 10000        # >=100  bootstrap resamples for CIs
  confidence: 0.99          # (0,1)  CI level
  seed: 42                  #        deterministic bootstrap seed

  # Onset-definition parameters (provenance today; see note)
  lookback_k: 20            # >=2
  significance_delta: 1.0   # >=0    threshold in sigma units
  onset_alpha: 0.01         # (0,1)  test level
```

- Validated by pydantic (`extra="forbid"`, frozen). Unknown keys are rejected.
- **Note / known issue:** the onset parameters are currently *inert* at evaluation
  time — labels are baked at generation. Once re-labelling lands (REFACTOR_PLAN
  #2) these become live; until then they document provenance only. Do not rely on
  them to change results.

## 5. Environment configuration **[Planned]**

```yaml
environments:
  include: "all"                 # or an explicit list of ids
  exclude: []
  tiers: ["tier1", "tier2", "tier3", "adversarial"]
  difficulty_override:           # per-environment difficulty knob
    tier1/gridworld_wireheading: 0.5
```

Selects which registered environments' datasets an evaluation runs over. In M1 the
environment set is passed directly as a trajectory list or dataset path.

## 6. Dataset-generation configuration **[Planned]**

Drives the generation scripts (today expressed as CLI flags on
`scripts/generate_gridworld_data.py`):

```yaml
data:
  version: "v1.0"
  output: "data/gridworld_wireheading.h5"
  environments: ["tier1/gridworld_wireheading"]
  seeds:
    n_hacking: 35
    n_clean: 15                  # 70/30 split
    seed_offset: 0
  difficulty: 0.5
  n_episodes: 500
  algorithm: "tabular_q_learning"
  onset:                         # oracle parameters applied at labelling
    lookback_k: 20
    significance_delta: 1.0
    alpha: 0.01
```

The generation `onset` block is where onset parameters are *authoritative* today
(they are applied by the environment oracle). Keeping them here and in
`evaluation` documents both the labelling-time and (future) re-labelling-time
values.

## 7. Recommended profile files **[Planned]**

Ship named profiles under `configs/`:

| Profile | Purpose |
|---|---|
| `default.yaml` | full defaults |
| `quick.yaml` | Tier 1, few seeds — fast iteration/CI |
| `paper.yaml` | exact configuration used for paper results |
| `tier1_only.yaml`, `tier2_only.yaml` | tier subsets |
| `ablations/*.yaml` | lookback/seed-count/clean-ratio/difficulty sweeps |

## 8. Validation & reproducibility requirements

- Every config is schema-validated; unknown keys are errors (fail loud).
- The resolved config is serialized into the evaluation report and hashed.
- Numeric ranges are enforced (e.g. `score_threshold ∈ [0,1]`,
  `onset_alpha ∈ (0,1)`).
- Two runs with identical resolved configs, data version, and seed MUST produce
  identical metrics.

## 9. Freeze guidance

- **`EvaluationConfig` field names and ranges are [Stable]** for the 1.x line;
  additions must be optional with defaults.
- The section structure (`experiment/data/detector/environments/evaluation`) is
  the frozen target; implement the loader and profiles before the leaderboard so
  submissions reference a canonical config.
