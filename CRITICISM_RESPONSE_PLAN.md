# RHOB Criticism Response Plan

**Document:** CRITICISM_RESPONSE_PLAN · **Status:** Planning · **No code in this document.**
Converts the concerns in [`BENCHMARK_REVIEW.md`](BENCHMARK_REVIEW.md) into a
concrete, triaged research + engineering plan. Each criticism is analyzed as:
scientific stake · what it requires · minimal architectural change · effort ·
research impact · release timing.

**Guiding objective:** maximize *scientific credibility and long-term adoption*, not
feature count. The plan is explicitly designed so that fixing the Critical items is a
prerequisite for making *any* "this is the standard" claim.

Effort scale: **S** < 3 days · **M** ≈ 1–2 wk · **L** ≈ 3–6 wk · **XL** > 6 wk.

---

## Master triage table

| ID | Criticism (review ref) | Tier | Requires | Effort | Impact |
|----|------------------------|:----:|----------|:------:|:------:|
| CR1 | Change-vs-hacking confound (R1, Rec 1) | **Critical** | envs, metric, protocol | L | Critical |
| CR2 | Unsound RHOB-Score CI (Rec 9) | **Critical** | evaluation | S | High |
| CR3 | Coverage/saturation — 1 env, 1 type (W4/5, R4) | **Critical** | envs, tasks | L–XL | Critical |
| CR4 | No headroom at launch (Rec 10) | **Critical** | env calibration | S–M | High |
| CR5 | Data-model rigidity — extensibility *hooks* (Rec 4) | **Critical** | API, dataset | M | Critical |
| CR6 | COI / governance (R5, Rec 8) | **Critical** | protocol (process) | S + ongoing | High |
| CR7 | Unrealistic monitored signal (W6, Rec 5) | **Important** | dataset, protocol | M | High |
| CR8 | Self-referential labels / construct validity (W1/2, Rec 7) | **Important** | dataset, metric, protocol | L | High |
| CR9 | Oracle-dependence of onset (R3, Rec 3) | **Important** | API, dataset, envs | M–L | High |
| CR10 | RLHF reward-model overoptimization env (Rec 2) | **Important** | env, task | L–XL | High |
| CR11 | Missing metrics beyond discriminability (Rec 7) | **Important** | metrics | M | Medium |
| CR12 | Onset point-estimate → fragile latency (§1.1) | **Important** | metric, protocol | S | Medium |
| CR13 | Test-set gaming via public generator (§2) | **Important** | dataset, protocol | M | Medium |
| CR14 | Full multimodal reach — LLM/robotics/multi-agent/POMDP (Persp. 4) | **Future** | envs, API, dataset | XL | High |
| CR15 | Real-world / live-mode grounding (R2/R3) | **Future** | envs, protocol | XL | High |

---

## CRITICAL — must fix before the first public release (v1.0)

### CR1. Prove it measures *hacking*, not *change*
- **Scientific stake.** The central validity threat: a trivial change-point detector
  (CUSUM) scores ≈0.99 on the only environment, so RHOB has not shown it measures
  reward hacking rather than generic change-point detection. Without this, the
  benchmark's headline claim is unsupported.
- **Requires:** ☑ new environments · ☑ new metric · ☑ evaluation protocol.
- **Minimal change.** Add ≥1 **matched-difficulty** environment where legitimate
  learning-change and hacking-change have *statistically indistinguishable* proxy
  dynamics (the planned "Mimicry"/"Correlated-Proxy" designs). Add a **discriminability
  metric** measuring separation between legitimate-change and hacking score
  distributions. Protocol: promote a pure change-point detector to a **named control
  baseline** that MUST fail on matched-difficulty environments while a hacking-aware
  detector succeeds — the benchmark's internal-validity certificate.
- **Effort:** L (the environment design is the hard, novel part).
- **Research impact:** Critical — this is the difference between a valid benchmark and
  a change-point demo.
- **Release timing:** **Before release.** At least one matched-difficulty environment
  and the discriminability metric gate v1.0.

### CR2. Fix the primary-metric statistics
- **Scientific stake.** The RHOB-Score confidence interval is computed over too few
  points (per-env, not per-trajectory); every reported and leaderboarded number
  inherits the flaw.
- **Requires:** ☑ evaluation (only).
- **Minimal change.** Trajectory-level stratified bootstrap that recomputes the
  tier-weighted score per resample (REFACTOR_PLAN #1); route all aggregates through one
  weighted, NaN-safe reducer.
- **Effort:** S.
- **Research impact:** High — correctness of the headline number.
- **Release timing:** **Before release.**

### CR3. Reach a minimum credible suite (coverage + headroom prerequisite)
- **Scientific stake.** One environment and one of six declared hacking types is a
  demonstration, not a benchmark; "SOTA on RHOB" is meaningless with a single task.
- **Requires:** ☑ new environments · ☑ new tasks.
- **Minimal change.** Using the extracted RL seam (foundation milestone), reach a
  **v1.0 floor of ≥5 environments spanning ≥3 hacking types**, including: the existing
  reward-tampering env, ≥1 non-tampering type (proxy gaming, spec gaming, or goal
  misgeneralization), the matched-difficulty env (CR1), and one gradual-onset env.
- **Effort:** L–XL.
- **Research impact:** Critical — turns a demo into a benchmark.
- **Release timing:** **Before release** (the floor). Full 21-environment vision is
  Important/Future.

### CR4. Create headroom
- **Scientific stake.** A benchmark solved at 0.99 by a trivial baseline at launch has
  no room to reward progress — the opposite of what made ImageNet productive.
- **Requires:** ☑ environment calibration (difficulty).
- **Minimal change.** Calibrate difficulty knobs (and add gradual-onset / matched-
  difficulty cases) so the **best baseline is < ~0.85 on ≥1 launch environment** and
  the aggregate is not saturated. Report the oracle ceiling and random floor to frame
  headroom.
- **Effort:** S–M (calibration, once CR1/CR3 envs exist).
- **Research impact:** High — adoption depends on having something to beat.
- **Release timing:** **Before release.**

### CR5. Add the extensibility *hooks* while the contract is young
- **Scientific stake.** The scalar-reward / training-time / single-agent data model
  cannot reach RLHF/LLM/robotics/multi-agent. Adding *additive, optional* contract
  hooks now is cheap; changing the contract after external adoption is a breaking
  epoch. This is the decisive long-term-vs-now tradeoff.
- **Requires:** ☑ API · ☑ dataset (both additive/backwards-compatible).
- **Minimal change (all optional, scalar/RL remain the defaults):**
  1. **Objective-signal generalization** — allow a named/vector objective-signal set
     (`{"proxy": …, "true": …}` today) so multi-objective/vector rewards fit later;
     scalar is the degenerate case.
  2. **Modality-typed `feature_spec`** — declare feature dimensionality *and* modality
     semantics, not a bare vector.
  3. **Abstract time axis** — a trajectory tag `time_axis ∈ {training_step,
     interaction_step}` (default `training_step`).
  4. **Label provenance** — `OnsetLabel.provenance ∈ {oracle, gold_rm, human, weak}`
     (default `oracle`); pairs with CR9.
  5. **Reserved fields** — `agent_id` / multi-agent container and a
     `partial_observability` flag, unused in v1.0 but present so v2.0 is additive.
- **Effort:** M (contract only; no modality *implementations* here).
- **Research impact:** Critical for long-term reach; near-zero cost now.
- **Release timing:** **Before release** — precisely because deferring it forces a
  backwards-incompatible v2.0.

### CR6. Neutralize the conflict-of-interest perception
- **Scientific stake.** The benchmark is authored by the proponent of the reference
  baseline (Flight Recorder), and an adversarial tier "designed to defeat known
  detectors" reads as tunable. Standards are not credibly set by a party with a
  competing entry; this blocks independent adoption regardless of technical merit.
- **Requires:** ☑ protocol / governance (process, not code).
- **Minimal change.** Recruit ≥1 independent co-maintainer without a competing method;
  **pre-register** environment and adversarial-tier designs before running any method;
  prominently report environments where the authors' method *loses*; and plan a
  shared-task/competition with independent organizers.
- **Effort:** S to establish + ongoing.
- **Research impact:** High for adoption; near-zero scientific cost.
- **Release timing:** **Before release** — at minimum the governance framing and
  pre-registration must exist before "standard" is claimed.

---

## IMPORTANT — should fix before paper submission

### CR7. Add realistic, noisy monitored signals
- **Scientific stake.** Detectors are scored on clean greedy-*expected* returns a real
  monitor never sees; this is likely why CUSUM reaches 0.99 and is the core realism
  gap driving non-transfer.
- **Requires:** ☑ dataset · ☑ protocol.
- **Minimal change.** Record a noisy **behaviour-return** channel alongside the greedy
  estimate (backwards-compatible optional field, subsumed by CR5's objective-signal
  set); make the detector-visible signal noisy by default and report on it; keep the
  greedy signal for oracle/analysis only.
- **Effort:** M.
- **Research impact:** High — closes the realism gap and de-trivializes the task.
- **Release timing:** Before paper; **pull earlier if resourced**, since it changes the
  headline "CUSUM 0.99" narrative.

### CR8. Externally anchor the labels (construct validity)
- **Scientific stake.** Labels currently grade against the definition that generated
  them; a high score must mean more than "recovers our synthetic pattern."
- **Requires:** ☑ dataset · ☑ metric · ☑ protocol.
- **Minimal change.** On ≥1 environment, collect independent onset labels (human
  annotators and/or a gold reward model), record `provenance` (CR5), and report a
  **label-agreement metric** (e.g., Krippendorff's α, oracle-vs-human onset
  correlation). This anchors the operational definition to something exogenous.
- **Effort:** L (human labeling is the cost).
- **Research impact:** High — the strongest available answer to "self-referential."
- **Release timing:** Before paper.

### CR9. Decouple onset ground truth from oracle true reward
- **Scientific stake.** Defining onset on `E[R_true]` structurally blocks real data and
  RLHF-at-scale; it also caps external validity.
- **Requires:** ☑ API · ☑ dataset · ☑ environments.
- **Minimal change.** Introduce an `OracleProtocol` abstraction so onset can be grounded
  by {true-reward two-sample (today), gold-RM divergence, human labels}; make
  `reward_true` optional in the schema (present for synthetic, absent for human-labeled
  real data); use `OnsetLabel.provenance` (CR5).
- **Effort:** M (contract) + environment cost accrues to CR10.
- **Research impact:** High — unlocks CR8, CR10, and any future real data.
- **Release timing:** Contract hook is Critical (folded into CR5); the gold-RM/human
  *implementations* are Important (before paper).

### CR10. Ship the RLHF reward-model-overoptimization environment
- **Scientific stake.** The single most important real reward-hacking setting today
  (Gao et al.); it also *partially escapes the oracle problem* because a held-out gold
  reward model can serve as ground truth. This is the likely killer task for adoption.
- **Requires:** ☑ new environment · ☑ new task (depends on CR5, CR9).
- **Minimal change.** An environment where a policy overoptimizes a proxy reward model
  while a held-out **gold RM** measures true reward; onset labeled by gold-RM divergence
  (`provenance=gold_rm`). Can start in a controlled/small-model regime.
- **Effort:** L–XL.
- **Research impact:** High — coverage, realism, and adoption pull in one task.
- **Release timing:** Before paper; a minimal version before release would materially
  raise credibility.

### CR11. Complete the metric suite
- **Scientific stake.** A single dominant scalar invites Goodharting; several
  scientifically-load-bearing metrics are missing.
- **Requires:** ☑ metrics · ☑ protocol.
- **Minimal change.** Add: **gradual-vs-sharp onset stratification** (report metrics
  split by onset sharpness); **score calibration**; and **measured** (not estimated)
  runtime/memory. Adopt **holistic reporting** — the per-metric × per-tier ×
  per-hacking-type profile is the primary artifact; RHOB-Score is a convenience index,
  not the sole ranking.
- **Effort:** M.
- **Research impact:** Medium–High — de-risks Goodharting and deepens the analysis.
- **Release timing:** Before paper.

### CR12. Make latency honest under onset uncertainty
- **Scientific stake.** `t*` shifts with `k/δ/α`, so latency measured against a point
  estimate carries false precision.
- **Requires:** ☑ metric · ☑ protocol.
- **Minimal change.** Report detection latency relative to the onset **confidence
  interval** (already present on `OnsetLabel.confidence_interval`) — an uncertainty band,
  not a point; derive the band from the onset-sensitivity ablation.
- **Effort:** S (the CI field already exists).
- **Research impact:** Medium — corrects a fragile headline metric.
- **Release timing:** Before paper.

### CR13. Harden the test split against gaming
- **Scientific stake.** A fully deterministic, public generator lets a submitter
  reverse-engineer the onset/activation distribution — a memorizable "test set."
- **Requires:** ☑ dataset · ☑ protocol.
- **Minimal change.** Split generation parameters into **public (train)** and **private
  (test)** sets; add nuisance randomization that breaks exploitable structure; withhold
  test labels *and* the test generation parameters. Aligns with `LEADERBOARD_SPEC`.
- **Effort:** M.
- **Research impact:** Medium — necessary before a competitive leaderboard.
- **Release timing:** Before the leaderboard (Important); partial mitigation before
  release.

---

## FUTURE WORK — can be deferred (but enabled now via CR5 hooks)

### CR14. Full multimodal reach
- **Scientific stake.** RLHF-at-scale, LLM agents, multi-agent RL, POMDPs, and
  continuous control are where reward hacking most matters; reaching them is the
  difference between a niche RL benchmark and the field standard.
- **Requires:** ☑ envs · ☑ API · ☑ dataset — but **additively**, on the CR5 hooks.
- **Minimal change.** Implement modalities against the v1.0 contract extensions:
  interaction-time trajectories (agents), vector/per-agent objective signals
  (multi-agent), modality-typed features (tokens/sensors), partial-observability flag
  (POMDP). No new *contract* if CR5 shipped.
- **Effort:** XL (per modality).
- **Research impact:** High — the long-term reach.
- **Release timing:** After release (v2.0), gated on CR5 being done first.

### CR15. Real-world grounding and live mode
- **Scientific stake.** SWE-Bench-level credibility requires real traces; live/online
  evaluation tests deployment realism.
- **Requires:** ☑ envs · ☑ protocol.
- **Minimal change.** Ingest semi-real/real RL/RLHF traces with gold-RM/human labels
  (`provenance` ≠ oracle); add a live evaluation mode alongside pre-recorded.
- **Effort:** XL.
- **Research impact:** High — but only after the core is validated.
- **Release timing:** Future (v2.x).

---

## v1.0 release gate (the minimum credible release)

The first public release is **blocked** until every Critical item is met:

- ☐ **CR2** trajectory-level RHOB-Score CI (correct headline statistics).
- ☐ **CR1** ≥1 matched-difficulty environment + discriminability metric + change-point
  control baseline that provably fails there.
- ☐ **CR3** ≥5 environments spanning ≥3 hacking types.
- ☐ **CR4** best baseline < ~0.85 on ≥1 environment (headroom).
- ☐ **CR5** additive extensibility hooks in the frozen v1.0 contract (objective-signal
  set, modality-typed features, time-axis tag, label provenance, reserved multi-agent /
  partial-observability fields).
- ☐ **CR6** independent co-maintainer + pre-registered designs + honest reporting of
  losses.

**Rationale.** Releasing v1.0 without CR1/CR3/CR4 ships a change-point demo branded as a
hacking benchmark; without CR2 the numbers are wrong; without CR5 v2.0 becomes a
breaking epoch; without CR6 the standard claim is not credible. Everything in
*Important* strengthens the paper; everything in *Future* is unlocked cheaply once CR5
lands.

## Effort roll-up

| Tier | Items | Aggregate effort |
|---|---|---|
| Critical | CR1–CR6 | ~8–11 person-weeks (CR3/CR1 dominate) |
| Important | CR7–CR13 | ~10–14 person-weeks (CR8/CR10 dominate) |
| Future | CR14–CR15 | XL, per-modality, deferred to v2.x |

The Critical tier is the load-bearing investment: it converts the artifact from
"promising demo" to "defensible v1.0 standard candidate."
