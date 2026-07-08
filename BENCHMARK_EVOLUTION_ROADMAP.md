# RHOB Benchmark Evolution Roadmap: v1.0 → v2.0

**Document:** BENCHMARK_EVOLUTION_ROADMAP · **Status:** Planning · **No code here.**
Describes how RHOB evolves from the current architecture proof (v0.1) to a credible
first public standard (v1.0) and on to the multimodal community standard (v2.0),
**maintaining backwards compatibility wherever possible.** It operationalizes
[`CRITICISM_RESPONSE_PLAN.md`](CRITICISM_RESPONSE_PLAN.md) as a versioned release
line.

The organizing principle: **make the v1.0 contracts supersets with optional,
defaulted fields, so v2.0 is additive rather than a breaking epoch.** The expensive
mistake — the one this roadmap exists to prevent — is shipping a narrow scalar/RL
contract, letting an ecosystem lock onto it, and then breaking it for RLHF/agents.

---

## 1. Versioning model (three independent axes)

| Axis | Scheme | Meaning |
|---|---|---|
| **Package** | SemVer `MAJOR.MINOR.PATCH` | code/API compatibility |
| **Dataset format** | `MAJOR.MINOR` (e.g. `1.1`) | on-disk schema (`DATASET_SPEC`) |
| **Spec** | `MAJOR.MINOR` | interface/definition contracts |

These are decoupled (a package may support several dataset formats). A **leaderboard
epoch** opens only on a dataset **MAJOR** bump (per `LEADERBOARD_SPEC`), so scores stay
comparable across a major line.

## 2. Backwards-compatibility principles (the contract)

Within a major line (all of 1.x), the following are **guaranteed**:

1. **Additive-only.** New dataset fields and API methods are optional with defaults;
   nothing required is removed or renamed.
2. **Superset contracts.** v1.0 schemas already contain the *shape* for v2.0 concepts
   (vector objective signals, modality, time axis, provenance, multi-agent), populated
   with backward-safe defaults. v2.0 fills them in; it does not invent new required
   structure.
3. **Graceful degradation.** A reader encountering an unknown optional field ignores
   it; a reader encountering a *missing* optional field applies the documented default
   (scalar reward, `time_axis=training_step`, `provenance=oracle`, single agent, full
   observability).
4. **Scalar as degenerate vector.** The generalized objective-signal set has scalar
   `{proxy, true}` as its degenerate case, so every v1.0 trajectory is a valid v2.0
   trajectory unchanged.
5. **Stable metric definitions.** AUROC, latency, miss rate, FPR@k, and RHOB-Score
   keep their definitions across 1.x; new metrics are added, not redefined. (The one
   sanctioned change is the CR2 CI *computation* fix, which corrects — not redefines —
   the estimator.)
6. **Detector API stability.** An L1/L2 detector written against v1.0 runs unchanged on
   every 1.x and, by policy, on v2.0 (new capabilities are opt-in).

What may change at **2.0** (and only there): introduction of genuinely new *required*
capabilities for new modalities, behind capability flags, with a published migration
path and a shim that reads 1.x data. Even here the goal is a **minimal breaking
surface** — ideally zero for existing scalar/RL datasets and detectors.

## 3. Release timeline

### v0.1 — Architecture proof (current, Milestone 1)
- **Contents:** 1 environment (reward tampering), Random + CUSUM baselines, metrics,
  HDF5 format 1.0, deterministic pipeline.
- **Status:** internal; not a public standard. Honestly labeled as a vertical slice.

### v1.0 — First public release ("minimum credible standard")
- **Gate:** all Critical items in `CRITICISM_RESPONSE_PLAN` (CR1–CR6).
- **Contents:**
  - **Validity:** ≥1 matched-difficulty environment + discriminability metric + a
    change-point control that provably fails there (CR1); best baseline < ~0.85 on ≥1
    env (CR4).
  - **Coverage:** ≥5 environments across ≥3 hacking types (CR3).
  - **Correct statistics:** trajectory-level RHOB-Score CI (CR2).
  - **Extensibility hooks (dataset format `1.1`, additive):** objective-signal set,
    modality-typed `feature_spec`, `time_axis` tag, `OnsetLabel.provenance`, reserved
    `agent_id` / `partial_observability` fields (CR5).
  - **Governance:** independent co-maintainer, pre-registered designs, honest loss
    reporting (CR6).
- **Scope statement (honest):** "onset detection under controlled reward divergence in
  single-agent RL." No overclaim of RLHF/agent coverage.
- **Compatibility:** format `1.1` is a strict superset of `1.0`; v0.1 datasets remain
  readable.

### v1.1–v1.x — Pre-paper hardening ("credible standard")
All **Important** items (CR7–CR13), all **additive** (dataset stays format `1.x`,
detectors unaffected):
- **v1.1** — realistic noisy monitored signal alongside greedy (CR7); onset-uncertainty
  latency bands (CR12).
- **v1.2** — externally-anchored labels (human/gold-RM) + label-agreement metric on ≥1
  env (CR8); `OracleProtocol` implementations for gold-RM/human grounding (CR9).
- **v1.3** — RLHF reward-model-overoptimization environment via a held-out gold RM
  (CR10); this is the flagship pre-paper task and the first partial escape from the
  oracle problem.
- **v1.4** — completed metric suite: gradual/sharp stratification, calibration, measured
  runtime/memory; holistic reporting as the primary artifact (CR11).
- **v1.5** — hardened public/private test split + nuisance randomization (CR13);
  leaderboard launch on the withheld split.
- **Compatibility:** every step optional/additive; a v1.0 detector still runs; a v1.0
  dataset is still valid.

### v2.0 — The multimodal standard ("the field standard")
All **Future** items (CR14–CR15), implemented **on the v1.x hooks**:
- **Modalities:** LLM/RLHF-at-scale, LLM agents (interaction-time trajectories),
  continuous control (robotics), multi-agent, partial observability.
- **Grounding:** semi-real and real traces (`provenance` ≠ oracle); optional live/online
  evaluation mode.
- **Format `2.0`:** the reserved v1.x fields become *populated* for new modalities.
  Because the shape existed since `1.1`, **v1.x scalar/RL datasets are valid v2.0
  datasets unchanged**, and **v1.x detectors run on v2.0** (new observation fields are
  opt-in). New leaderboard epoch opens (per `LEADERBOARD_SPEC`) so cross-modality scores
  aren't naively pooled.
- **Minimal break:** any unavoidable breaking change is gated behind a capability flag
  with a reader shim for `1.x` data and a documented migration.

## 4. Dataset-format evolution

| Format | Introduced | Adds (all optional unless noted) | Reads older? | Older tools read it? |
|:---:|:---:|---|:---:|:---:|
| `1.0` | v0.1 | scalar `rewards_proxy/true`, `policy_features`, `onset_label` | — | — |
| `1.1` | v1.0 | objective-signal set (scalar default), modality-typed feature spec, `time_axis`, `provenance`, reserved `agent_id`/`partial_observability` | ✓ | ✓ (ignore optional) |
| `1.2` | v1.x | noisy observed-signal channel; external-label agreement fields | ✓ | ✓ |
| `2.0` | v2.0 | populated multi-agent/interaction/modality data; real-trace provenance | ✓ | partial (structure parses; new-modality eval needs v2 tools) |

**Key guarantee:** a `1.0` file is valid under `1.1`, `1.2`, and `2.0`; a `2.0`
scalar/RL file is valid under `1.x` tooling. Only *new-modality* `2.0` data requires
`2.0` tooling to evaluate.

## 5. Detector-API evolution

| Stage | Adds | Existing detectors affected? |
|---|---|:---:|
| v1.0 | frozen L1/L2 streaming contract; contract-validation gate | — |
| v1.x | optional `score_sequence` batch path; optional `fit` for learning-based; L3 fields | No (all opt-in) |
| v2.0 | modality-typed observations (tokens/sensors), interaction-time axis, per-agent views, L4 | No for L1/L2/L3; new fields are opt-in |

An L1 CUSUM written for v1.0 keeps producing identical scores through v2.0.

## 6. Compatibility matrix (summary)

| Consumer ↓ / Producer → | v1.0 data | v1.x data | v2.0 scalar data | v2.0 modality data |
|---|:---:|:---:|:---:|:---:|
| **v1.0 tooling** | ✓ | ✓ (ignores new opt.) | ✓ | parse-only |
| **v1.x tooling** | ✓ | ✓ | ✓ | parse-only |
| **v2.0 tooling** | ✓ | ✓ | ✓ | ✓ |

| Consumer ↓ / Detector → | L1/L2 (v1.0) | L3 (v1.x) | modality (v2.0) |
|---|:---:|:---:|:---:|
| **v1.x harness** | ✓ | ✓ | n/a |
| **v2.0 harness** | ✓ | ✓ | ✓ |

## 7. Deprecation & migration policy

- **Deprecation window.** Any field or method slated for change is marked deprecated
  for ≥1 MINOR before a MAJOR may alter it; deprecations are listed in `CHANGELOG.md`.
- **No silent breaks.** Removing/renaming a required field requires a MAJOR bump, a
  migration note, and a reader shim for the prior format.
- **Reproducibility pin.** Every result records `(package_version, dataset_format,
  dataset_version, config_hash)`; re-running a pinned tuple reproduces numbers within a
  major line.
- **Score comparability.** RHOB-Scores are comparable within a dataset major line; a
  dataset MAJOR opens a new epoch and historical entries are preserved, not re-ranked.

## 8. Release cadence & governance

- **Cadence:** v1.0 after the Critical gate; v1.x MINORs as Important items land;
  v2.0 after modalities are validated on the hooks. Adversarial-tier refresh on a fixed
  cadence to prevent saturation.
- **Governance:** pre-registered environment/adversarial design; independent
  co-maintainers; a shared task/competition to seed adoption and neutralize COI; a
  documented 3-year maintenance horizon moving toward community governance.

## 9. One-line strategy

Ship a **correct, honest, extensible v1.0** (validity + coverage + headroom + hooks +
governance), harden it additively through v1.x (realism + external labels + RLHF +
holistic metrics), then grow into v2.0 **on the hooks** so the multimodal standard is
additive — not a break — for everyone who adopted v1.0.
