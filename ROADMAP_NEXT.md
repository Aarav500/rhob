# RHOB Next-Milestone Roadmap

**Document:** ROADMAP_NEXT · **Status:** Recommendation
Recommends the next milestones in priority order, optimized for **long-term
scientific value and community adoption** — not for adding features quickly. Each
milestone lists objective, why it matters, dependencies, effort, risks, and
expected research impact.

## Sequencing principle

The order deliberately puts **correctness and interface freeze first, coverage
second, theory third, validity fourth, adoption fifth, publication last.** You do
not scale on an unsound metric, you do not publish a benchmark without coverage
and robustness evidence, and adoption infrastructure only pays off once there is
something worth adopting. This *reprioritizes* the original project plan (which
front-loaded theory) because the architecture review showed the foundation fixes
are cheap and load-bearing, and because for a *benchmark* the primary scientific
value is coverage + validity + adoption, with theory as a high-value differentiator
that can proceed in parallel.

| # | Milestone | Focus | Effort | Research impact |
|---|-----------|-------|:------:|:---------------:|
| 1 | Interface Freeze & Foundation Hardening | correctness | 2–3 wk | Foundational |
| 2 | Environment Suite + Baselines | coverage | 4–6 wk | High |
| 3 | Theory Track | depth | 2–3 wk (parallel) | High |
| 4 | Ablations & Scale | validity | 3–4 wk | High |
| 5 | Leaderboard & Adoption | adoption | 3–4 wk | High (adoption) |
| 6 | Paper & v1.0 Release | publication | 3–4 wk | Capstone |

---

## Milestone 1 — Interface Freeze & Foundation Hardening

**Objective.** Land the P0/P1 refactors (`REFACTOR_PLAN.md` #1–#7) and freeze the
interface specs so every later milestone builds on stable, correct foundations:
trajectory-level RHOB-Score CI + consistent weighted aggregation; separate
labeling from generation with a `relabel()` API (kill the phantom onset knob);
freeze the access-level + L2 `feature_spec` contract; decompose the environment
into MDP/trainer/recorder; add a detector registry; distinguish missed-hacks from
clean; add property + regression tests.

**Why it matters.** An unsound primary-metric CI and baked labels would propagate
into every downstream result and the leaderboard; the access/feature contract must
be frozen *before* external detectors depend on it; the extraction seams prevent
the environment suite from duplicating code four times. This is the cheapest
possible time to fix all of it.

**Dependencies.** This spec suite (done). No new science required.

**Estimated effort.** 2–3 weeks (mostly refactor + tests; low external risk).

**Risks.** Over-engineering / scope creep. *Mitigation:* scope strictly to the
P0/P1 items; each has an acceptance criterion in `REFACTOR_PLAN.md`. Reported
numbers will shift slightly when the CI is corrected — expected and desirable.

**Expected research impact.** Foundational rather than headline: it makes every
future number correct and every future extension cheap. Skipping it compounds
error and duplication through M2–M6.

---

## Milestone 2 — Environment Suite + Baselines

**Objective.** Implement the five Tier 1 environments (reusing the extracted RL
seam) and begin Tier 2; ship the full baseline set — Flight Recorder (L2,
structural), ensemble disagreement (L2), KL monitor (L3), gradient-norm (L3) — plus
the oracle ceiling and the existing Random/CUSUM. Wire L3 plumbing per the frozen
access contract.

**Why it matters.** Coverage across hacking *types* and difficulty *tiers* is what
turns a demo into a benchmark: it is the substrate for the headline empirical
claims — *no single method dominates*, the *difficulty hierarchy*, and
*cross-environment transfer*. A credible, diverse baseline set makes RHOB
immediately useful to others (the GLUE/HELM adoption pattern).

**Dependencies.** Milestone 1 (env decomposition, `feature_spec`, L2/L3 access
contract, detector registry).

**Estimated effort.** 4–6 weeks (environment design + generation + baseline
integration + validation; some GPU for Tier 2).

**Risks.** (a) Some environments may not hack reliably — *mitigation:* the
`validate()` ≥60% admission gate plus 2 backup designs per slot. (b) Flight
Recorder / ensemble integration friction — *mitigation:* write the L2 adapter
against the frozen contract first and test on M1 data.

**Expected research impact.** High — this is the empirical core of the paper and
the reason the benchmark is worth citing.

---

## Milestone 3 — Theory Track (detection limits)

**Objective.** Formalize and prove the fundamental detection-limit results: an L1
impossibility bound (oracle-blind reward-only detection has Ω(T) latency under
near-perfect proxy–true correlation) and an L2-over-L1 separation; verify them
numerically against the M2 environments.

**Why it matters.** The theory is what differentiates RHOB from a "resource paper"
at TMLR and gives it durable scientific depth — it explains *why* access levels
matter and establishes the first formal limits of the problem. It also validates
the access-level design empirically.

**Dependencies.** Milestone 1 (a clean access-level contract for the L1/L2
numerical link) and Milestone 2 (environments to ground the constructions). Can run
**in parallel** with M2's compute-heavy generation.

**Estimated effort.** 2–3 weeks of focused theory (bounded — do not exceed; fall
back to a conditional/empirical result if a proof won't close).

**Risks.** Proofs may not close cleanly (endogenous, non-stationary change point).
*Mitigation:* weaken to a conditional impossibility or an empirically-characterized
limit — still publishable.

**Expected research impact.** High — the theoretical differentiator that lifts the
paper from "benchmark" to "formalization + benchmark."

---

## Milestone 4 — Ablations & Scale

**Objective.** Run the validity ablations — onset-definition sensitivity (via the
new `relabel()` path across k/δ/α), seed-count, clean-ratio, algorithm-agnosticism
(PPO vs. tabular), and access-level degradation — and land the scale
infrastructure (lazy/streamed loading, single-pass aggregation, parallel workers)
needed to evaluate the full suite × 50 seeds.

**Why it matters.** A benchmark's credibility rests on demonstrating that method
*rankings are stable* under its definitional choices and training algorithm.
Reviewers will demand exactly these ablations; the onset-sensitivity study is only
feasible because Milestone 1 made re-labeling cheap.

**Dependencies.** Milestone 1 (relabel, lazy load, parallelism), Milestone 2
(environments/baselines), Milestone 3 (hierarchy predictions to validate).

**Estimated effort.** 3–4 weeks (analysis + substantial compute; budget waiting
time).

**Risks.** Rankings may prove unstable under some parameter — *mitigation:* that is
itself a real, reportable finding; recalibrate the definition and re-run (cheap,
post-relabel). Compute overruns — *mitigation:* parallelism from this milestone.

**Expected research impact.** High — the robustness evidence that makes the
benchmark trustworthy and the ranking defensible.

---

## Milestone 5 — Leaderboard & Adoption Infrastructure

**Objective.** Stand up the public leaderboard per `LEADERBOARD_SPEC` (submission
format, CI reproducibility checks, git-committed results, static site), the `rhob`
CLI, a documentation site, a versioned dataset release on a public hub, and
contribution/governance docs.

**Why it matters.** Adoption is what makes RHOB *the standard*. Frictionless
submission, a living leaderboard, and a versioned dataset are the flywheel that
turned ImageNet/GLUE/SWE-Bench into community defaults. Low friction and automatic
verification are the whole game.

**Dependencies.** Milestone 2 (a suite worth ranking), Milestone 4 (withheld test
split + reproducibility hashing), `LEADERBOARD_SPEC` / `CONFIG_SPEC`.

**Estimated effort.** 3–4 weeks.

**Risks.** Maintenance burden and submission abuse — *mitigation:* git-committed
JSON + automated CI validation, an adversarial-tier refresh cadence, and a
documented 3-year maintenance horizon moving to community governance.

**Expected research impact.** High for adoption and long-run citations; medium for
direct science. This is the milestone that determines whether RHOB becomes
infrastructure or stays a paper.

---

## Milestone 6 — Paper & v1.0 Release

**Objective.** Write and submit the TMLR paper (formalization + benchmark +
baseline evaluation + ablations), post the arXiv preprint, and cut the v1.0 package
and dataset release.

**Why it matters.** The paper is the canonical reference every future
reward-hacking-detection method will cite, and the catalyst for adoption.

**Dependencies.** Milestones 1–5.

**Estimated effort.** 3–4 weeks.

**Risks.** Writing surfaces evidence gaps — *mitigation:* budget 2–3 days of
fill-gap experiments (cheap now that generation, relabeling, and parallelism
exist).

**Expected research impact.** The capstone — converts the infrastructure into a
citable standard.

---

## What NOT to do next (anti-priorities)

- **Do not add more environments before Milestone 1.** Building on the current
  metric/label foundation multiplies error and duplication.
- **Do not build the leaderboard before there is a suite to rank** (Milestone 2/4).
- **Do not chase L4 / live-mode / multi-agent** until the L1–L3 core, theory, and
  ablations are solid — they widen scope without deepening the core claim.
- **Do not optimize performance prematurely** beyond what Milestone 4's scale run
  actually requires.

## One-line rationale

Freeze and correct the foundation → build coverage → add theoretical depth →
prove validity → ship adoption infrastructure → publish. Each step is a
prerequisite for the scientific value of the next.
