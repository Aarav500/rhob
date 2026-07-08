# RHOB Detector API

**Document:** DETECTOR_API · **Spec version:** 0.1 · **Status:** Draft for freeze
Source of truth: `src/rhob/detectors/base.py`, `core/access.py`.

Every detector — a two-line statistical rule or a deep sequence model — implements
the same small interface and is scored identically. A detector consumes a stream
of access-filtered observations and emits, per step, a hacking probability in
`[0, 1]`.

---

## 1. The interface **[Stable]**

```python
class MyDetector(rhob.Detector):          # AbstractDetector
    # --- Identity (class attributes) ---
    name: str = "my_detector"
    id: str = "namespace/my_detector"
    version: str = "0.1.0"
    access_level: AccessLevel = AccessLevel.L1
    is_oracle_free: bool = True            # must be True for the main leaderboard

    def reset(self) -> None: ...           # REQUIRED
    def step(self, observation) -> float: ...   # REQUIRED, returns [0,1]

    # optional
    def configure(self, **kwargs) -> None: ...
    def hyperparameters(self) -> dict: ...
    def computational_overhead(self) -> OverheadEstimate: ...
```

### 1.1 Required methods

| Method | Contract |
|---|---|
| `reset()` | Clear **all** internal state. Called once before each trajectory. After `reset()`, behaviour must be independent of any prior trajectory. |
| `step(obs) -> float` | Consume one observation; return a hacking probability in `[0.0, 1.0]`. Must be finite. |

### 1.2 Optional methods

| Method | Purpose | Default |
|---|---|---|
| `configure(**kwargs)` | Apply configuration once before evaluation | sets matching attributes |
| `hyperparameters()` | Return current hyperparameters (for reproducibility) | `{}` |
| `computational_overhead()` | Report relative cost vs. base training | `0.0` |

## 2. Initialization

Two-phase, to separate construction from benchmark-controlled configuration:

1. **`__init__`** — the detector's own constructor sets hyperparameters
   (`CUSUMDetector(slack_k=0.5, threshold_h=5.0)`).
2. **`configure(**kwargs)`** — called once by the harness before evaluation to
   inject config-file values; overrides matching attributes.

`hyperparameters()` should return the effective values so a run is fully
reproducible from the report.

## 3. Inputs — the `Observation` **[Stable]**

`step` receives an immutable, access-filtered `Observation`:

| Field | Type | Available at | Notes |
|---|---|---|---|
| `t` | `int` | always | Training-step index |
| `access_level` | `AccessLevel` | always | What this observation contains |
| `reward` | `float` | L1+ | Proxy reward |
| `policy_features` | `float64[F] \| None` | L2+ | Behavioural features; `None` below L2 |
| *(gradients, KL)* | — | L3+ | **[Planned]** |
| *(internal state)* | — | L4 | **[Planned]** |

**Access enforcement is structural.** The harness constructs a fresh, frozen
`Observation` with fields above the detector's declared `access_level` set to
`None` and feature arrays marked read-only. A detector declared L1 **cannot**
receive L2 fields, by construction. The oracle's `reward_true` is *never* present
in any `Observation`.

## 4. Outputs **[Stable]**

- **Per-step:** a `float` in `[0.0, 1.0]` — the probability that hacking is active
  at step `t`. `1.0` = "certainly hacking".
- The sequence of scores over a trajectory is the detector's output; the harness
  derives all metrics (AUROC, latency, FPR@k, …) from it and the ground-truth
  onset. Detectors do **not** compute metrics or emit binary alerts themselves;
  binarization uses the config's `score_threshold`.

**Bounds are enforced.** Out-of-range or non-finite scores raise
`ScoreBoundsError` with an actionable message. Squash raw statistics into `[0,1]`
(e.g. `1 - exp(-S/h)`).

## 5. State handling & contract invariants

The harness validates these before scoring (bounds + determinism on a sample; full
certification is **[Planned]**):

| Invariant | Requirement |
|---|---|
| **Determinism** | Identical observation sequences → identical scores, bit-for-bit. Seed any RNG in `reset()`. |
| **Bounded output** | Every score in `[0, 1]`, finite. |
| **Reset isolation** | Post-`reset()` behaviour independent of previous trajectories. |
| **Access compliance** | Never read fields above the declared level (guaranteed by the filter). |
| **Streaming cost** | `step()` should be ~O(1) amortized and sub-linear in memory. |

## 6. Online vs. offline detectors

The **streaming `step()` interface is the canonical form** and models the target
use case (real-time monitoring during training). Both detector families map onto
it:

- **Online / causal** (CUSUM, KL monitor): update internal state per step; the
  score at `t` depends only on observations `≤ t`. This is the natural fit.
- **Offline / non-causal**: a detector that wants the full trajectory before
  scoring should implement the **[Planned]** batch method (§7) rather than
  buffering inside `step()`. Note: scores that peek at future steps are permitted
  by the metric math but are *not* deployable and should be reported as offline.

Detectors SHOULD declare causality in metadata (**[Planned]** field) so the
leaderboard can separate deployable (online) from analysis-only (offline) methods.

## 7. Batch evaluation support **[Planned]**

To support vectorized statistical detectors and deep models efficiently, an
optional batch entry point will be added (REFACTOR_PLAN #8):

```python
def score_sequence(self, observations: Sequence[Observation]) -> np.ndarray
```

- Returns one score per step; the harness uses it when present, else falls back to
  the per-step loop.
- Preserves access filtering (the sequence is already filtered).
- Lets a detector run its model once over the whole trajectory (a single forward
  pass for a neural detector) instead of `T` Python calls.

## 8. Learning-based (deep) detectors **[Planned]**

Detectors that train on the labelled train split implement:

```python
def fit(self, dataset) -> None      # train on the TRAIN split only
```

Requirements to keep results comparable:

- **No test-set peeking.** `fit` may use only the train split; the leaderboard
  verifies `training_data_used` (see `LEADERBOARD_SPEC`).
- **Determinism after training.** Once fit, `reset()`/`step()` must be
  deterministic; ship weights or a seed for reproduction.
- **Declared inputs and access level.** A neural detector consuming L2 features
  declares `access_level = L2` and reads only `policy_features`.

The same `step`/`score_sequence` interface serves inference, so statistical and
deep detectors are scored identically.

## 9. Reference baselines (implemented)

| Detector | id | Access | Kind | Role |
|---|---|:---:|---|---|
| `RandomDetector` | `baselines/random` | L1 | online | metric floor (AUROC ≈ 0.5) |
| `CUSUMDetector` | `baselines/cusum` | L1 | online | classical change-point baseline |

**[Planned]:** Flight Recorder (L2, structural), ensemble disagreement (L2),
KL monitor (L3), gradient-norm (L3), and an oracle ceiling (labels-only, non-submittable).

## 10. A detector registry **[Planned]**

Mirroring the environment registry (REFACTOR_PLAN #5): `register_detector`,
`list_detectors`, `list_by_access`, and a central `validate_contract` — enabling
discovery, access-level sub-leaderboards, and one contract gate for the full
detector suite.

---

## Minimal example

```python
import rhob

class SlopeDetector(rhob.Detector):
    name = "slope"; id = "user/slope"; access_level = rhob.AccessLevel.L1
    def reset(self):
        self.prev, self.s = None, 0.0
    def step(self, obs):
        if self.prev is not None and obs.reward > self.prev:
            self.s = min(1.0, self.s + 0.05)
        self.prev = obs.reward
        return self.s

report = rhob.evaluate(SlopeDetector(), trajectories)   # no registration needed
```

A detector is plug-and-play: subclass, declare an access level, pass to
`evaluate`. No benchmark-core change is required.
