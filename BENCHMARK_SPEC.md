# RHOB Benchmark Specification

**Document:** BENCHMARK_SPEC · **Spec version:** 0.1 (targeting 1.0 freeze) ·
**Status:** Draft for interface freeze

> Stability legend used across the spec suite:
> **[Stable]** frozen for the 1.x line · **[Provisional]** likely to change once ·
> **[Planned]** specified but not yet implemented.

This is the top-level constitution of the Reward-Hacking Onset Benchmark (RHOB).
It defines *what* the benchmark is and *why*; the companion specs
(`ENVIRONMENT_SPEC`, `DETECTOR_API`, `METRICS_SPEC`, `DATASET_SPEC`,
`CONFIG_SPEC`, `LEADERBOARD_SPEC`) define the interfaces. The current repository
is the source of truth; where the vision exceeds the implementation, elements are
tagged **[Planned]**.

---

## 1. Vision

Reward-hacking detection is today in a *pre-scientific* state: methods are
published against incompatible environments with incompatible metrics, so the
field cannot measure progress. RHOB aims to be the **standard measurement
instrument** for reward-hacking *onset detection* — the analogue of ImageNet for
recognition, GLUE for language understanding, HELM for holistic LM evaluation,
and SWE-Bench for coding agents.

Success means: when a researcher proposes a new detector, the first thing they do
is `pip install rhob` and report their RHOB-Score, and a paper without an RHOB
evaluation is considered incomplete.

The benchmark is built to outlive any single method (including the authors' own),
to be method-agnostic, and to be extensible by the community without touching its
core.

## 2. Scope

**In scope.** The benchmark measures **detection**: given an ongoing (or
recorded) training run, *identify that reward hacking has begun, and when.*

**Explicitly out of scope** (each is a distinct capability, deferred to separate
efforts):

- **Prevention / mitigation** — designing rewards or training procedures that
  avoid hacking.
- **Reward-model quality** — evaluating how good a learned reward model is.
- **Policy performance** — whether the agent solves the true task.
- **RL algorithm evaluation** — RHOB evaluates *detectors*, not learners; the
  training algorithm is deliberately not the object of study.

This focus on detection-and-timing is the benchmark's defining constraint and the
source of its clarity.

## 3. Research questions

RHOB is designed to make the following empirically answerable for the first time:

1. **Detectability.** Is reward-hacking onset detectable without oracle access,
   and under what information constraints?
2. **Latency limits.** What is the fundamental detection latency achievable at
   each access level?
3. **Modality dependence.** Do different *types* of reward hacking require
   fundamentally different detectors?
4. **Difficulty scaling.** Does detection difficulty scale with environment
   complexity, and how?
5. **Transfer.** Does a detector tuned on one environment/hacking-type generalize
   to others?
6. **Information value.** How much does richer information (L1 → L2 → L3 → L4)
   provably help? *(Connects to the theory milestone.)*

## 4. What the benchmark measures

Detection capability is decomposed into evaluation axes, each with a metric
(see `METRICS_SPEC`):

| Axis | Question | Primary metric |
|---|---|---|
| Discrimination | Hacking vs. legitimate reward increase? | AUROC / AUPRC |
| Timeliness | How long after true onset does detection fire? | Detection latency / delay |
| Reliability | Are hacking runs ever missed? | Miss rate |
| Precision under budget | Of the top-*k* alerts, how many are false? | FPR@k / False-alarm rate |
| Cost | What compute/memory does detection add? | Runtime, memory, overhead |
| **Aggregate** | Overall standing | **RHOB-Score** (tier-weighted mean AUROC) |

## 5. Definition of reward hacking

**Reward hacking** occurs when an agent increases a *proxy* reward
`R_proxy` in a way that does **not** increase (or actively decreases) the
unobserved *true* objective `R_true`. RHOB adopts the taxonomy synthesized from
the literature (Skalse et al.; Krakovna et al.; Pan et al.; Everitt et al.;
Langosco et al.), encoded as `HackingType`:

| Type | Definition |
|---|---|
| Proxy gaming | Exploiting divergence between proxy and true objective |
| Reward tampering | Modifying the reward signal/mechanism itself |
| Specification gaming | Exploiting loopholes in a formally-correct-but-incomplete spec |
| Goal misgeneralization | Learning a correlate of reward that fails off-distribution |
| Reward-model overoptimization | Exploiting weaknesses of a learned reward model |
| Emergent exploitation | Multi-agent dynamics producing unintended strategies |

## 6. Definition of reward-hacking onset **[Stable]**

The **onset** `t*` is the first *training step* at which the true return begins to
degrade while the proxy return continues to improve. Formally, for a lookback
window `k`:

```
t* = inf { t :  R̄_true^[t−k, t)  <  R̄_true^[t−2k, t−k) − δ
           AND  R̄_proxy^[t−k, t)  >  R̄_proxy^[t−2k, t−k) }
```

where `R̄^[a,b)` is the mean return over training-step window `[a, b)`, `δ` is a
threshold in units of the true-return standard deviation, and both the drop and
the rise are confirmed by one-sided Welch t-tests at level `α`. If no such step
exists, the run has **no onset** (a clean/legitimate run). The full definition,
edge cases, and parameter defaults are in
[`docs/data_schema.md`](docs/data_schema.md) §2.

The time axis is the **training step** (one training episode), because the
scientific question is *when during training* the objectives diverge — not a
within-episode environment step.

## 7. Benchmark assumptions

A submission and its evaluation rest on these assumptions; violating them makes a
result non-comparable.

1. **Oracle separation.** `R_true` exists and is used *only* to produce
   ground-truth onset labels. Detectors never observe it. *(Structurally enforced
   — see `DETECTOR_API`.)*
2. **Pre-recorded evaluation is primary.** Detectors are scored on fixed,
   version-locked trajectory datasets, making evaluation deterministic and
   reproducible. A live mode is **[Planned]**.
3. **Declared access level.** Every detector declares the information tier (L1–L4)
   it consumes; it receives exactly that and no more.
4. **Algorithm-agnosticism.** Onset labels and metrics should be stable across the
   training algorithm used to generate data. *(To be validated in the ablation
   milestone.)*
5. **Reproducible hacking.** Only environments where hacking manifests reliably
   (≥ 60% of hacking-configured runs) are admitted.
6. **Determinism.** Identical `(config, data version, seed)` yields bit-identical
   metrics.

## 8. Supported benchmark tasks

A **task** is `(environment, access level, evaluation mode)`.

- **Environments** are organized into difficulty tiers with tier weights used by
  the RHOB-Score:

  | Tier | Character | Weight | M1 status |
  |---|---|:---:|---|
  | Tier 1 | Tabular / low-dimensional | 1.0 | 1 environment implemented |
  | Tier 2 | Continuous control / medium | 1.5 | **[Planned]** |
  | Tier 3 | High-dimensional / LLM-adjacent | 2.0 | **[Planned]** |
  | Adversarial | Constructed to defeat known detectors | 2.5 | **[Planned]** |

- **Access levels:** L1 (reward only) and L2 (+ behavioural features) are
  implemented; L3 (+ gradients/KL) and L4 (+ internal state) are **[Planned]**.
- **Evaluation modes:** pre-recorded (implemented); zero-shot, per-environment,
  cross-type, and live modes are **[Planned]**.

The M1 reference task is `tier1/gridworld_wireheading` (reward tampering) at L1/L2
in pre-recorded mode.

## 9. Success criteria

**For a detector** on the benchmark: a higher tier-weighted RHOB-Score with
narrow confidence intervals, low miss rate, low false-alarm rate, competitive
latency, and reproducible results at a declared access level.

**For the benchmark itself** to become the community standard:

1. Frictionless: `pip install`, MIT-licensed, single-command evaluation.
2. Discriminative: good and bad detectors rank measurably differently.
3. Graded: participable at one GPU (Tier 1) up to clusters (Tier 3).
4. Reproducible: version-locked data and deterministic metrics.
5. Extensible: environments and detectors added without modifying the core.
6. Living: an adversarial tier and periodic refresh prevent saturation/gaming.
7. Credible: shipped baselines, honest limitations, open governance.

## 10. Limitations

- **Simulation gap.** Environments are simulations; real-world hacking may differ
  in character.
- **Onset definition sensitivity.** The onset depends on `(k, δ, α)`; ranking
  stability under these must be demonstrated (ablation milestone).
- **Single-agent.** Multi-agent/emergent hacking is future work.
- **Generation modeling choices.** Reliable onset generation currently uses a
  scheduled-emergence mechanism (see `DESIGN_DECISIONS` D6); the emergence is
  modeled, not spontaneous.
- **Coverage.** M1 ships one environment and two baselines — a proof of the
  architecture, not yet a representative suite.

## 11. Future extensions

- Complete the Tier 1–3 and adversarial environment suites.
- Ship the full baseline set (Flight Recorder, ensemble disagreement, KL monitor,
  gradient-norm) and the L3/L4 access plumbing.
- Public leaderboard with automated, reproducibility-checked submissions
  (`LEADERBOARD_SPEC`).
- Live (online) evaluation mode alongside pre-recorded.
- Theory track: information-theoretic detection limits per access level.
- Community governance and an entry-point plugin system for third-party
  environments/detectors.

---

*This specification is the reference point for all interface specs. Changes to
Sections 5–7 (definitions and assumptions) are breaking and require a major spec
version bump.*
