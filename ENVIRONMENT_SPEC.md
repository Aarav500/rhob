# RHOB Environment Specification

**Document:** ENVIRONMENT_SPEC · **Spec version:** 0.1 · **Status:** Draft for freeze
Source of truth: `src/rhob/environments/base.py`, `registry.py`, `oracle.py`.

An RHOB environment is a **data generator**, not a step-in/step-out simulator: it
produces complete, labelled training runs that detectors are later scored against.
This decoupling (training ≠ detection) is fundamental — it makes evaluation
deterministic and independent of any RL algorithm.

**Core requirement:** new environments are added by subclassing
`AbstractEnvironment` and registering via the public API — **the benchmark core is
never modified.**

---

## 1. The environment → benchmark contract

Every environment subclasses `rhob.Environment` (`AbstractEnvironment`) and
provides identity/spec attributes plus one method.

### 1.1 Required attributes **[Stable]**

```python
class MyEnv(rhob.Environment):
    # --- Identity ---
    id: str                 # unique, "tier<N>/<name>" or "<namespace>/<name>"
    name: str               # human-readable
    tier: Tier              # TIER1 | TIER2 | TIER3 | ADVERSARIAL
    hacking_type: HackingType
    description: str

    # --- Specifications ---
    state_dim: int
    action_dim: int
    action_type: str        # "discrete" | "continuous"
    max_steps: int          # trajectory length T (training steps)
    expected_onset_range: tuple[int, int]
    hacking_reliability: float   # fraction of hacking-configured seeds that hack

    # --- Difficulty ---
    difficulty_knob: str | None
    difficulty_range: tuple[float, float] | None
    difficulty_default: float | None
```

### 1.2 Required method **[Stable]**

```python
def generate(self,
             seed: int,
             difficulty: float | None = None,
             config: dict | None = None) -> Trajectory
```

**Determinism guarantee (mandatory):** identical `(seed, difficulty, config)`
MUST return a byte-identical `Trajectory`. This is the single most important
contract for reproducibility.

**Config conventions:** `config["hacking"] = False` requests a *clean* run (the
exploit is not accessible → no onset). Environments MAY define additional
`config` keys, but MUST default to a hacking-configured run.

### 1.3 Provided by the base class (do not override)

- `describe() -> EnvironmentCard` — structured metadata.
- `validate(n_seeds=10, min_reliability=0.6) -> ValidationReport` — self-check that
  the environment hacks reliably; **admission gate** for the suite.

### 1.4 Planned additions to the contract **[Planned]**

- `feature_spec: FeatureSpec` — a declared schema for L2 `policy_features`
  (dimensionality + named semantics). Required so cross-environment L2 detectors
  have a stable input contract. *(Tracked in REFACTOR_PLAN #3; specify before Tier 2.)*
- `l3_spec` / `l4_spec` — declarations for gradient/KL and internal-state fields.

---

## 2. Episode (training-run) lifecycle

RHOB's "episode" is one **training run** producing one `Trajectory` of length
`T = max_steps`. The generation lifecycle:

```
generate(seed, difficulty, config)
  │
  ├─ 1. Seed a private RNG from `seed`  (all stochasticity flows from it)
  ├─ 2. Resolve difficulty → environment parameters
  ├─ 3. Resolve hacking vs clean from config
  ├─ 4. Run the internal training process for T training steps:
  │        for t in range(T):
  │            train one step (internal MDP; algorithm-agnostic)
  │            record reward_proxy[t], reward_true[t], policy_features[t]
  ├─ 5. Label onset via the oracle over (reward_proxy, reward_true)  [hacking runs]
  └─ 6. Assemble and return a Trajectory (curves + features + label + metadata)
```

Steps 1 and the determinism of 4 are mandatory; the *internal* training process
(step 4) is unconstrained (tabular Q-learning, PPO, or any generator) as long as
the output contract and determinism hold.

## 3. Observation format

Two distinct notions of "observation" exist; keep them separate.

### 3.1 Trajectory per-step record (environment output) **[Stable]**

Each training step `t` contributes:

| Field | Type | Level | Meaning |
|---|---|---|---|
| `reward_proxy[t]` | `float64` | L1 | Observable proxy return of the policy at step `t` |
| `reward_true[t]` | `float64` | oracle | True return (labels only; never shown to detectors) |
| `policy_features[t]` | `float64[F]` | L2 | Behavioural summary (e.g. state-visitation) |

`reward_proxy` and `policy_features` are the detector-visible signals;
`reward_true` is oracle-only.

### 3.2 Detector-visible `Observation` (access-filtered view) **[Stable]**

Downstream, the runner presents each step to a detector as an immutable
`Observation` filtered to the detector's declared access level (see
`DETECTOR_API`). Environments do **not** construct `Observation`s — the access
filter does.

## 4. Action format

RHOB does not mandate a specific action space at the benchmark boundary — actions
are an *internal* concept of the environment's training process. The declared
`action_dim` and `action_type` (`"discrete" | "continuous"`) are **metadata** for
documentation and future L2 feature interpretation, not an enforced interface.
This keeps the contract general enough for tabular, continuous-control, and
token-space (LLM) environments.

## 5. Reward interface

An environment internally defines two reward channels:

- **Proxy reward** `R_proxy` — observable; drives (and can be gamed by) the agent.
- **True reward** `R_true` — the real objective; used only for labelling.

The environment records the *policy's return* under each channel per training
step (recommended: the greedy policy's return, an estimate of `E_{π_t}[R]`, which
matches the onset definition). The gap between the two channels is what the oracle
detects. Reward *shaping/misspecification* — how the two channels diverge — is the
essence of each environment and is documented in its `EnvironmentCard`.

## 6. Onset generation interface **[Stable]**

Labelling is performed by an `OnsetOracle` over the recorded curves:

```python
oracle.compute_onset(reward_proxy: np.ndarray,
                     reward_true:  np.ndarray,
                     hacking_type: HackingType) -> OnsetLabel | None

oracle.validate_label(reward_true: np.ndarray, label: OnsetLabel) -> bool
```

Parameters `(lookback_k, significance_delta, alpha)` implement the onset
definition (`BENCHMARK_SPEC` §6). An environment owns an oracle instance and calls
it in `generate` step 5. Because `reward_true` is persisted in the dataset,
**re-labelling with different oracle parameters without regeneration** is a
target capability (**[Planned]**, REFACTOR_PLAN #2).

`OnsetLabel` fields: `onset_step`, `confidence`, `hacking_type`,
`detection_method`, `confidence_interval`, `severity`.

## 7. Registration mechanism **[Stable]**

Environments register through the public API — **no core files are edited**:

```python
import rhob

@rhob.register_environment          # decorator form
class MyEnv(rhob.Environment): ...

# or explicitly
rhob.register_environment(MyEnv)

rhob.list_environments()            # discovery
rhob.get_environment("ns/my_env")   # instantiate by id
rhob.get_environment_card("ns/my_env")
```

The registry also provides `get_environment_tier(id)` used by the RHOB-Score.
Built-in environments self-register on import.

- **Third-party extension is verified:** a custom environment defined and
  registered outside `src/` evaluates end-to-end with zero core changes.
- **Known gaps (see ARCHITECTURE_REVIEW):** the registry is an import-hub (no
  entry-point plugin system yet, **[Planned]**), and an *unregistered* environment
  silently falls back to `TIER1` in scoring — register before evaluating.

## 8. Environment metadata (`EnvironmentCard`) **[Stable]**

`describe()` returns an `EnvironmentCard` for documentation and the registry:

| Field | Meaning |
|---|---|
| `id`, `name`, `tier`, `hacking_type` | Identity |
| `true_objective` | What the agent *should* do |
| `proxy_reward` | What the proxy measures |
| `hacking_mechanism` | How hacking manifests |
| `oracle_mechanism` | How onset is detected |
| `difficulty_knob` | Tunable difficulty parameter |
| `compute_per_seed` | Cost estimate |
| `citation` | Prior-work provenance (optional) |

Additionally, each `Trajectory` carries free-form `metadata` (generation
parameters, config hash). **Note:** metadata may currently contain
generation-internal fields (e.g. activation timing) that are *not* exposed to
detectors via the official path but should be schema-controlled in future
(ARCHITECTURE_REVIEW W20).

---

## 9. Admission checklist for a new environment

1. Subclasses `AbstractEnvironment`; all identity/spec attributes set.
2. `generate` is deterministic in `(seed, difficulty, config)`.
3. Supports both hacking and clean (`config["hacking"]=False`) runs.
4. `validate()` reports `hacking_reliability ≥ 0.6`.
5. Onset labels pass `validate_label` (true return lower after onset).
6. Registered via `register_environment`; appears in `list_environments()`.
7. `EnvironmentCard` fields populated.
8. *(When available)* declares `feature_spec` for L2 features.

Meeting this checklist is sufficient for inclusion — no benchmark-core change is
required.
