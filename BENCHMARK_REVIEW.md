# RHOB Benchmark Review — Committee Assessment

**Framing.** This reviews the *benchmark as a proposed community standard*, not the
implementation. It simulates a joint committee with four lenses: **TMLR** (formal
depth, correctness of claims), **ICML** (methodological rigor, construct
validity), **NeurIPS Datasets & Benchmarks** (utility, coverage, reproducibility),
and **ICLR** (generality of the representation/abstraction). The bar is deliberately
high: *would every future reward-hacking-detection paper evaluate against this?*

Verdict up front: **a genuinely important idea sitting on a promising but
under-validated and, today, toy artifact.** The conceptual contribution (temporal
onset detection with an access-level hierarchy) is real; the benchmark's current
scientific standing is limited by self-referential labels, a realism gap, and — most
damagingly — a single synthetic environment that a generic change-point detector
already solves at AUROC ≈ 0.99. The architecture must generalize its core data
model *now*, while young, or it will not reach RLHF/LLM/robotics.

---

## Perspective 1 — Scientific Validity

### 1.1 Is reward-hacking onset well defined? Partly — and that is the problem.
The onset is defined operationally: the first training step where mean `R_true`
significantly decreases while mean `R_proxy` increases, over a window `k`, at
significance `δ, α`. This is *precise* but it is a **definition, not a discovery**.
Three validity concerns follow:

- **Point-estimate false precision.** Reward hacking is frequently gradual; forcing
  a single `t*` imposes a sharp boundary on a continuous phenomenon. Because the
  headline **detection-latency metric is measured relative to `t*`**, and `t*` moves
  with `k/δ/α`, the latency metric inherits the definition's uncertainty. If `t*`
  varies by ±30 steps across reasonable parameters, a reported latency of "0.02·T"
  is not meaningfully distinguishable from "0.08·T." The planned onset-sensitivity
  ablation tests *ranking* stability but not whether *latency itself* is well-posed.

- **Oracle-dependence limits external validity.** The definition requires
  `E[R_true]`, which exists only in simulation. In any real deployment `R_true` is
  unknown — that is the entire motivation. So the benchmark's ground truth is
  *constitutively* unavailable outside a simulator, and RHOB can never (under the
  current definition) incorporate a real-world reward-hacking trace. This caps how
  "real" the standard can ever become.

### 1.2 Are the labels scientifically meaningful? Yes, but self-referentially.
The environment is *built* to produce a proxy/true divergence (via scheduled exploit
activation), and the oracle *labels* that divergence with the same rule that defines
onset. The detector is then scored against those labels. This is a **closed loop**:
the benchmark defines hacking, synthesizes exactly that, and grades against its own
definition. Contrast ImageNet, whose labels are exogenous human semantics. RHOB's
labels are endogenous. This does not make the benchmark useless — a definition-driven
benchmark can still measure "can a method recover this defined signal" — but it means
**a high RHOB-Score certifies recovery of a synthetic divergence pattern, not
detection of reward hacking in the wild.** Construct validity (does RHOB performance
predict real detection performance?) is currently **unestablished and, given 1.1,
hard to establish.**

### 1.3 Could different definitions change conclusions? Almost certainly, at the margins.
Two-sample vs. CUSUM-oracle vs. regression definitions will disagree on `t*` and on
*which runs count as hacking* near threshold. The spec designates the two-sample
definition primary and the others "ablations," but the community has not converged
on any of them. A committee will read the choice of primary definition as an
unforced degree of freedom that the authors could (even unintentionally) tune.

### 1.4 Hidden assumptions (several, and load-bearing).
1. **A scalar, observable `R_proxy` and a scalar `R_true` exist.** False for LLM
   agents, most RLHF settings, and multi-objective tasks.
2. **The time axis is training-time (policy evolution).** Excludes deployment-time /
   within-episode hacking, which is exactly the RLHF/agent case.
3. **The monitored signal is the greedy policy's *expected* return** — a clean,
   near-noise-free quantity a real monitor never sees. This makes detection
   artificially easy and is the likely reason CUSUM reaches 0.99.
4. **Hacking emerges as a scheduled discontinuity.** Real hacking arises from
   continuous optimization pressure; a scheduled "exploit turns on at `E_activate`"
   injects a structural artifact a detector could exploit (see 1.5).
5. **One agent, full observability of the training curve.** Baked into the contract.

### 1.5 Internal consistency — the sharpest ICML-style criticism.
**The benchmark does not yet demonstrate that it measures *reward-hacking* detection
rather than *generic change-point* detection.** CUSUM — a textbook change detector
with no notion of "hacking" — scores 0.99 on the only environment. The intended
control is the clean run (which also contains a change: goal-learning), but if the
hacking-phase proxy jump is larger or sharper than the legitimate-learning jump, any
change detector wins. To show it measures hacking *specifically*, RHOB needs
environments where legitimate change and hacking change are **statistically matched**
(the planned "Mimicry" adversarial environment) — and those do not exist yet. Until
they do, an honest reading is: **RHOB currently measures change-point detection on a
proxy reward, with hacking as the narrative.**

**Scientific-validity summary:** the formal apparatus is coherent and the temporal
framing is a genuine contribution, but self-referential labels, oracle-dependence,
an unrealistic signal, and the change-vs-hacking confound leave construct validity
unproven. **The concept is sound; the current evidence that it measures what it
claims is not.**

---

## Perspective 2 — Benchmark Design vs. the Standards

| Dimension | ImageNet | GLUE | HELM | SWE-Bench | **RHOB (today)** |
|---|---|---|---|---|---|
| Scale | 1.4M images | 9 tasks | many scenarios | 2k+ real issues | **1 environment** |
| Label source | human semantics | human/existing | mixed | real PRs (executable) | **algorithmic (self-defined)** |
| Real-world grounding | high | medium | medium | **very high** | **none (synthetic)** |
| Ceiling/headroom | large at launch | saturated fast | ongoing | large | **already saturated (0.99)** |
| Multi-metric | top-1/5 | task avg | holistic | resolve rate | single-dominant (weighted AUROC) |
| Community buy-in at launch | competition | consortium | institution | practitioner pull | **none yet** |

**Strengths (design).** Fills a real, articulated gap (no onset-detection benchmark
exists); the access-level hierarchy is an elegant, novel axis that enables the
information-value question; difficulty tiers with score weighting are sensible;
determinism/reproducibility are excellent; the API is genuinely extensible.

**Weaknesses (design).**
- **Coverage is the existential weakness.** One environment, one hacking type (reward
  tampering) out of a declared taxonomy of six. A one-task "benchmark" is not a
  benchmark; it is a demonstration.
- **Saturation at launch.** The reference environment is already solved (0.99) by a
  trivial baseline, so there is no headroom to reward a better detector — the
  opposite of what made ImageNet productive.
- **Single dominant metric.** RHOB-Score = tier-weighted AUROC. HELM's lesson is that
  a standard benefits from *holistic* multi-metric reporting; a single scalar invites
  Goodharting (ironic for a reward-hacking benchmark).
- **Synthetic-only.** SWE-Bench's success came precisely from real, executable tasks.
  RHOB has the opposite posture and no path (under its current onset definition) to
  real data.

**Missing tasks / environments.** RLHF reward-model overoptimization (the single most
important real setting today, per Gao et al.); continuous control (MuJoCo proxy
reward); image-observation tasks; multi-agent emergent exploitation; goal
misgeneralization; and any adversarial/mimicry environment (needed for §1.5).

**Missing metrics.** Label-quality/human-agreement (the blueprint's Krippendorff-α is
not a required output); a gradual-vs-sharp-onset stratifier; an explicit
legitimate-change-vs-hacking discriminability metric; calibration; and *measured*
(not estimated) compute/memory.

**Reproducibility.** A genuine strength — but note the double edge: perfectly
deterministic, public, synthetic generation invites **test-set gaming** (a detector
can reverse-engineer the `E_activate` distribution). A "reproducible" benchmark whose
generator is fully known is also a *memorizable* one.

**Extensibility.** Strong at the API level (plug-and-play verified), weak at the
*representation* level (the scalar/training-curve model does not obviously extend —
see Perspective 4).

---

## Perspective 3 — Research Value: would a detector author choose it?

**Today: mostly no.** A researcher publishing a new detector gains little by
reporting RHOB numbers, because:

- **No coverage, no story.** With one saturated environment, you cannot claim "SOTA
  on RHOB" or show improvement over baselines; there is nothing to beat.
- **Toy transfer risk.** A detector reviewer will ask "does this work on real
  training runs / RLHF?" RHOB cannot answer that, so it does not de-risk the
  detector paper.
- **Perceived conflict of interest.** The benchmark is authored by the proponent of a
  specific method (Flight Recorder), which is also the reference baseline, and the
  design choices that make detection easy/hard (L2 emphasis, sharp onset, structural
  signal, adversarial tier "designed to defeat known detectors") could be read as
  favorable to that method. GLUE/ImageNet/SWE-Bench were not built by a party with a
  competing entry. This is a real adoption barrier and must be managed with
  independent co-maintainers and pre-registered environment design.

**Barriers to adoption.** Toy/synthetic; saturated; single environment; no external
results; no leaderboard yet; COI perception; and the construct-validity gap (§1).

**Missing capabilities.** LLM/RLHF support; live/deployment-time mode; measured cost;
a populated, independently-verified leaderboard; more than two baselines.

**Opportunities for ecosystem growth.** (a) An **RLHF reward-model-overoptimization
environment** would be the killer task — enormous current interest, and it partially
escapes the oracle problem (a held-out gold reward model can serve as ground truth).
(b) **Semi-real traces** from public RL/RLHF runs with human-labeled or gold-RM onsets
would buy SWE-Bench-style credibility. (c) A **shared-task/competition** with
independent organizers would break the COI perception and seed community buy-in.

---

## Perspective 4 — Long-Term Vision (5-year extensibility)

Can the current architecture naturally expand to robotics, RLHF, LLM agents,
multi-agent RL, POMDPs, continuous control, and real deployments?

**What helps (keep):** pre-recorded trajectory evaluation, the access-level filter,
the streaming detector contract, and oracle-separation are all modality-agnostic in
spirit.

**What blocks (must change now, while young):**

1. **Scalar `reward_proxy[t]` / `reward_true[t]`.** RLHF/LLM/multi-objective/multi-
   agent settings have vector, distributional, or *nonexistent* scalar true reward.
   → **Generalize the reward channel to a named, possibly multi-dimensional "objective
   signal" set**, with scalar as the degenerate case.
2. **Flat, env-specific, schemaless `policy_features`.** For LLMs the behavioural
   signal is token/logit/activation structure; for robotics, sensor streams.
   → **Make `feature_spec` modality-typed** (already flagged for cross-env transfer;
   the long-term need is stronger than the M2 need).
3. **Training-episode time axis.** A deployed LLM agent has no training episode;
   hacking occurs at *interaction* time over a task/conversation.
   → **Abstract the time axis** (training-step *or* interaction-step) in the trajectory
   contract.
4. **Onset grounded in oracle `E[R_true]`.** Precludes real data and RLHF-at-scale.
   → **Decouple "onset ground truth" from "oracle true reward":** allow label
   provenance ∈ {oracle, gold-reward-model, human, weak-supervision}, so real and LLM
   settings can enter the benchmark.
5. **Single-agent, fully-observed training curve.** Multi-agent needs per-agent
   trajectories/labels; POMDP monitoring needs partial detector views distinct from
   access levels.
   → **Reserve a multi-agent trajectory container and a partial-observability flag**
   in the schema now (cheap), even if unused until later.
6. **No realism/provenance taxonomy.** → **Tag every environment `synthetic |
   semi-real | real`** so the suite can visibly grow toward real-world grounding and
   so scores can be reported stratified by realism.

**Assessment:** the *services* (evaluation, access control, reporting) will extend;
the *data model* will not, without breaking changes. The five-year outcome hinges on
generalizing the reward channel, the feature schema, the time axis, and the
label-provenance model **before** an ecosystem locks in on the scalar/training
assumptions. These are ~2 focused weeks of contract work now versus a painful major
version bump later.

---

## Strengths

1. Addresses a real, well-articulated gap: no temporal onset-detection benchmark
   exists. Genuine novelty.
2. The onset *formalization* (temporal, not post-hoc) is a real conceptual
   contribution.
3. The **access-level hierarchy** is elegant and enables the "how much does
   information help?" question — a durable scientific axis.
4. Determinism and reproducibility engineering are excellent.
5. Extensible API (plug-and-play environments and detectors verified).
6. Difficulty tiers + weighting incentivize hard cases.
7. Unusually honest self-documentation of limitations and debt.

## Weaknesses

1. **Construct validity unproven** — RHOB performance is not shown to predict
   real-world detection (and may be hard to show).
2. **Self-referential labels** — the benchmark grades against its own definition.
3. **Change-vs-hacking confound** — a generic change detector scores 0.99; hacking-
   specific measurement is not yet demonstrated.
4. **One synthetic environment, one hacking type** — not yet a benchmark.
5. **Saturation at launch** — no headroom on the reference task.
6. **Unrealistic monitored signal** (clean greedy-expected returns).
7. **Scalar/training-time data model** limits extension to RLHF/LLM/multi-agent.
8. **Single dominant metric** invites Goodharting.
9. **COI perception** (author's method is the reference baseline).

## Major Risks

- **R1 — It measures change-point detection, not reward hacking.** If the matched-
  difficulty (Mimicry) environments never materialize or fail, the central claim
  collapses.
- **R2 — Non-transfer.** Methods that win on RHOB fail on real RLHF/agent hacking,
  discrediting the benchmark after adoption.
- **R3 — Never escapes simulation.** The oracle-`R_true` definition structurally
  blocks real-world data, capping credibility.
- **R4 — Coverage never arrives.** Solo maintenance stalls at Tier 1; a one-task
  benchmark is abandoned.
- **R5 — COI capture.** Perceived tuning toward the authors' method prevents
  independent adoption.

## Minor Risks

- Onset point-estimate ill-posedness makes latency numbers fragile.
- Deterministic public generator enables test-set gaming.
- Single-scalar RHOB-Score gamed by optimizing easy tiers (mitigated by weights, not
  eliminated).
- `activation_episode` and other generation internals stored alongside data (leak-
  adjacent).
- Difficulty knob currently only shifts onset timing (thin difficulty axis).

## Top 10 Recommendations

1. **Build matched-difficulty "legitimate-change vs. hacking" environments first**
   (Mimicry/Correlated-Proxy). Without them, the benchmark cannot show it measures
   hacking rather than change. *This is the highest-priority scientific fix.*
2. **Ship an RLHF reward-model-overoptimization environment** using a held-out gold
   reward model as ground truth — the killer task and a partial escape from the
   oracle problem.
3. **Decouple onset ground truth from oracle `R_true`;** add label provenance
   {oracle, gold-RM, human} so semi-real and real data can enter.
4. **Generalize the data model now:** vector/structured objective signals,
   modality-typed `feature_spec`, an abstract (training vs. interaction) time axis,
   and a reserved multi-agent/partial-observability schema.
5. **Add realistic, noisy monitored signals** (behaviour returns / logged metrics)
   alongside greedy-expected returns, and report both — close the realism gap that
   makes CUSUM trivially win.
6. **Report holistically (HELM-style):** never rank on a single scalar alone; publish
   the per-metric, per-tier, per-hacking-type profile as the primary artifact.
7. **Add a discriminability metric** that directly measures separation between
   legitimate-change and hacking distributions, and a label-agreement metric.
8. **Establish independent governance:** co-maintainers without a competing detector,
   pre-registered environment/adversarial-tier design, and a shared-task/competition
   to neutralize COI and seed adoption.
9. **Fix the primary-metric statistics** (trajectory-level RHOB-Score CI) before any
   number is published or leaderboarded.
10. **Create headroom deliberately** — calibrate at least one launch environment so
    the best baseline is well below ceiling, giving future detectors something to
    beat.

## Potential Reviewer Criticisms → Suggested Responses

| Criticism | Suggested response |
|---|---|
| "Labels are self-defined; this measures your own definition." | Concede it is a definition-driven benchmark; add human/gold-RM-validated labels on ≥1 environment to anchor the definition externally; report label-agreement. |
| "A trivial CUSUM gets 0.99 — you're measuring change points." | Ship matched-difficulty environments where legitimate learning and hacking are statistically indistinguishable to a change detector; show baselines separate there. |
| "It's a toy; does it transfer to real RLHF/agents?" | Deliver the RLHF/gold-RM environment and, ideally, a semi-real trace set; report a transfer study. Until then, scope claims to "onset detection under controlled divergence." |
| "Oracle `R_true` makes it impossible to include real data." | Decouple onset grounding from oracle reward (provenance model); demonstrate on gold-RM-labeled data. |
| "Conflict of interest — your method is the yardstick." | Independent co-maintainers, pre-registered design, competition, and cases where the authors' method loses, reported prominently. |
| "One environment isn't a benchmark." | Correct today; present M1 as architecture validation and commit to the coverage roadmap with an admission gate. |
| "Onset is a fuzzy point; latency is ill-defined." | Report latency with an onset-uncertainty band derived from the `k/δ/α` ablation; treat `t*` as an interval, not a point. |
| "Single scalar invites Goodharting." | Adopt holistic reporting; keep RHOB-Score as a convenience index, not the sole ranking. |

## Overall Recommendation

**Major revision before it can claim standard status — but continue; the idea is
worth it.** As a *research contribution* (temporal onset formalization + access-level
framework + reference architecture), this is publishable at a venue like TMLR with
the statistics fixed and claims scoped honestly. As *the community standard for
reward-hacking detection*, it is **not there yet** and will not get there on the
current synthetic/scalar/single-environment trajectory. The path is clear and the
foundation is unusually clean; execution on coverage, realism, construct validity,
and governance — not more engineering polish — determines the outcome.

The single most important thing to do next is **not** to add features, but to prove
the benchmark measures reward hacking rather than change (Recommendation 1) and to
generalize the data model before an ecosystem locks in (Recommendation 4).

## Scores (1–10, current state; ceiling in parentheses if roadmap executed)

| Axis | Score | Notes |
|---|:---:|---|
| **Scientific Soundness** | **6** (8) | Coherent formalism and a novel temporal framing, undercut by self-referential labels, oracle-dependence, and the unproven change-vs-hacking distinction. |
| **Engineering Design** | **8** (9) | Clean, deterministic, extensible, well-documented; held from 9 by the metric-CI and data-model debts. |
| **Benchmark Quality** | **3** (8) | One saturated synthetic environment and one hacking type. A proof of architecture, not yet a benchmark. |
| **Community Adoption Potential** | **4** (8) | Real gap and good ergonomics, but toy coverage, saturation, and COI perception block adoption today. |
| **Long-Term Research Impact** | **6** (9) | The problem matters and the access-level idea is durable; realized impact hinges on generalizing the data model and reaching RLHF/real settings. |

*Scores are deliberately harsh, per the committee's mandate. The gap between current
and ceiling is almost entirely coverage, realism, construct validity, and
governance — not code quality.*
