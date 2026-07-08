# Conversation
- conversationId: 1af8a011-445a-4ec6-9815-e754d844a63c
- endpoint: agents
- title: New Chat
- exportAt: 18:35:52 GMT-0500 (Central Daylight Time)

## Options
- presetId: null
- agent_id: agent_qFKewAVHfqqlvkw4jZFFv
- endpoint: agents
- endpointType: undefined
- title: New Chat

## History
**User**
Take the engineering specification.

Break it into milestones.

Each milestone should

- have a clear deliverable

- estimate effort

- define dependencies

- define success criteria

- specify tests

Optimize for a solo researcher over approximately 8–12 weeks.


**Research Project Manager**
{"type":"think","think":"Aarav wants me to take the RHOB (Reward-Hacking Onset Benchmark) publication blueprint and break it into concrete engineering milestones optimized for a solo researcher over 8–12 weeks. I need to think like a PI who understands the reality of solo research execution.\n\nLet me analyze what needs to be built:\n\n1. **Theory**: Onset definition formalization, Theorem 1 (L1 impossibility), Theorem 2 (L2 dominates L1)\n2. **Infrastructure**: Detector API, environment base classes, metrics, evaluation harness\n3. **Environments**: Tier 1 (5 environments), Tier 2 (7 environments), potentially Tier 3\n4. **Baselines**: 6 detection methods plus Oracle and Random variants\n5. **Experiments**: Main results table, ablations, statistical significance testing\n6. **Writing**: Full paper draft\n\nFor solo execution, I need to front-load the highest-risk theoretical and infrastructure work, create tight feedback loops by building one environment at a time and testing everything before scaling, scope aggressively to avoid getting stuck, and make sure each milestone produces something testable. The key is identifying what's truly MVP versus what can wait.\n\nLet me map out the optimal sequencing starting with Week 1-2 on the foundation—nailing down the onset definition, establishing the detector API contract, and building one proof-of-concept environment, since everything downstream depends on getting these right. Then Week 3-4 focuses on core infrastructure: metrics, evaluation harness, and getting CUSUM plus Random baselines working end-to-end on that proof-of-concept, which gives me one complete vertical slice by the end. Week 5-6 scales to Tier 1 environments, integrates the Flight Recorder baseline, and produces initial results showing discrimination. Week 7-8 tackles the theory—proving Theorem 1 rigorously or figuring out what weakened version is actually provable, then attempting Theorem 2. Week 9-10 adds Tier 2 environments, the full baseline suite, and ablations. Week 11-12 is writing, polish, and any remaining experiments.\n\nBut I'm reconsidering whether theory should be partially front-loaded because the impossibility result could reshape the paper's positioning, the proof sketch informs which hard instances to construct in the environments, and formalizing early forces clarity. Yet engineering still needs to come first enough to have actual environments for validating the theory against.\n\nSo I'm restructuring around more natural milestones: Foundation in Week 1 locks definitions and the API contract. Weeks 2-3 deliver a vertical slice with one environment, two baselines, and the full metrics pipeline. Weeks 3-5 tackle the theory core—Theorem 1 proof and formalization that's paper-ready. Weeks 5-6 complete Tier 1 with all five environments and all baselines. Then Milestone 4 is cut off but presumably covers the remaining work.\n\nNow I'm mapping out the critical dependencies: the onset definition feeds into environment implementation, labeling, metrics, and results. It also feeds into the Theorem 1 proof. The detector API enables baseline implementations which produce results. And the full pipeline of environments, baselines, and metrics flows into experiments, ablations, and writing.\n\nThe real risk is whether the impossibility result actually holds. If it doesn't, I still have a benchmark paper, but it's weaker for TMLR, so I need to start the theory attempt early enough to pivot if necessary."}


**Research Project Manager**
# RHOB Engineering Milestones

---

## Overview

**Scope**: Solo researcher, 8–12 weeks, targeting TMLR submission.

**Critical Path**: Onset Definition → Detector API → One Environment End-to-End → Theory → Scale → Write

**Design Principle**: Every milestone produces a testable artifact. No milestone exceeds 2 weeks. Each milestone is independently valuable (if you stop at Milestone 3, you still have a submittable paper sketch).

---

## Milestone 0 — Foundation Lock

**Duration**: 5 days (Week 1)

**Deliverable**: A 3-file specification that everything else builds against — frozen before any code is written.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Formalize onset definition mathematically | `docs/onset_definition.tex` — 2 pages, rigorous | 8 |
| Define edge cases + resolution rules | Table of 6+ edge cases with binary decisions | 4 |
| Specify detector API contract | `rhob/api.py` — abstract base class, type hints, docstrings | 4 |
| Specify metrics API | `rhob/metrics.py` — function signatures, formulas, edge-case behavior | 4 |
| Specify environment base class | `rhob/envs/base.py` — interface every environment must implement | 4 |
| Define data format (HDF5 schema) | `docs/data_schema.md` — field names, types, shapes | 3 |
| Write acceptance tests for the API contracts | `tests/test_api_contract.py` — tests pass on dummy implementations | 5 |

**Total estimated effort**: ~32 hours

### Dependencies

- None (this is the root).

### Success Criteria

1. Onset definition handles all 6 identified edge cases with no ambiguity.
2. `Detector` abstract class is implementable in < 20 lines for a trivial detector.
3. Metrics function signatures are complete — given predictions + ground truth, every metric is computable.
4. All acceptance tests pass with dummy/mock implementations.
5. A second reader (or future-you in 2 weeks) can implement against these specs without asking questions.

### Tests

```python
# test_api_contract.py
def test_detector_interface_minimal():
    """A random detector conforms to the API."""
    det = RandomDetector()
    score = det.step(obs=np.zeros(4), action=0, reward=1.0, next_obs=np.zeros(4))
    assert 0.0 <= score <= 1.0

def test_metrics_on_perfect_detector():
    """Oracle detector achieves AUROC=1.0, latency=0."""
    preds = oracle_predictions(ground_truth)
    m = compute_metrics(preds, ground_truth)
    assert m.auroc == 1.0
    assert m.latency == 0.0

def test_metrics_on_random_detector():
    """Random detector achieves AUROC ≈ 0.5."""
    preds = random_predictions(len(ground_truth))
    m = compute_metrics(preds, ground_truth)
    assert 0.4 < m.auroc < 0.6  # within noise

def test_onset_label_deterministic():
    """Same trajectory → same onset label every time."""
    label1 = label_onset(trajectory, k=1000, delta=1.0)
    label2 = label_onset(trajectory, k=1000, delta=1.0)
    assert label1 == label2
```

### Risk

- **Medium**: Onset definition may need revision after seeing real training curves. Mitigation: build one quick prototype environment (Milestone 1) within the same week to sanity-check.

### Decision Gate (End of Week 1)

> "Can I explain the onset definition, the API, and the data format to someone in 5 minutes with no ambiguity?"

If no → spend 2 more days. Do not proceed with ambiguous foundations.

---

## Milestone 1 — Vertical Slice

**Duration**: 8 days (Week 2 – early Week 3)

**Deliverable**: One complete end-to-end pipeline: one environment, two baselines, all metrics, reproducible result.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Implement `GridWorldWireheading` environment | `rhob/envs/gridworld_wireheading.py` | 10 |
| Implement programmatic onset labeler for this env | Oracle access to true reward → label | 4 |
| Generate 10 training runs (PPO, SB3) | 10 `.hdf5` trajectory files with onset labels | 6 |
| Implement `RandomDetector` | Baseline floor | 1 |
| Implement `CUSUMDetector` | Classical change-point on proxy reward | 6 |
| Implement full metrics pipeline | AUROC, AUPRC, latency, FPR@k, miss rate, TFD | 8 |
| Implement evaluation harness | `rhob/evaluate.py` — runs detector on trajectory, computes all metrics | 6 |
| Produce first results table | 2 methods × 1 env × 10 seeds → table with CIs | 4 |
| Validate: CUSUM > Random on AUROC | Discrimination confirmed | 2 |

**Total estimated effort**: ~47 hours

### Dependencies

- Milestone 0 (all specs locked)

### Success Criteria

1. `GridWorldWireheading` reliably produces hacking in ≥ 60% of runs (at least 6/10).
2. Onset labels are correct when manually inspected (plot proxy + true reward, mark onset, visually confirm for all 10 runs).
3. CUSUM achieves AUROC > 0.6 (must beat random meaningfully).
4. Random achieves AUROC ∈ [0.45, 0.55].
5. Full pipeline runs in < 5 minutes on a laptop (excluding agent training).
6. `pytest tests/` passes, including integration test for the full pipeline.

### Tests

```python
def test_gridworld_hacking_rate():
    """≥ 60% of runs exhibit hacking."""
    results = [train_and_check(seed=i) for i in range(10)]
    assert sum(results) >= 6

def test_onset_labels_agree_with_manual():
    """Spot-check 3 trajectories against hand-labeled onsets."""
    for traj_id in [0, 4, 7]:
        auto = label_onset(load_trajectory(traj_id))
        manual = MANUAL_LABELS[traj_id]
        assert abs(auto - manual) < 50  # within 50 steps

def test_cusum_beats_random():
    """CUSUM AUROC significantly > 0.5."""
    cusum_aurocs = evaluate_all(CUSUMDetector(), trajectories)
    assert np.mean(cusum_aurocs) > 0.55
    # One-sample t-test against 0.5
    _, p = ttest_1samp(cusum_aurocs, 0.5)
    assert p < 0.05

def test_full_pipeline_deterministic():
    """Same inputs → same metrics."""
    m1 = run_pipeline(detector, trajectories, seed=42)
    m2 = run_pipeline(detector, trajectories, seed=42)
    assert m1 == m2
```

### Risk

- **High**: GridWorld may not reliably hack with default PPO. Mitigation: have a backup environment (NavigationProxy, simpler to force hacking via reward shaping).
- **Medium**: CUSUM may need careful tuning to show any signal at all. Mitigation: if AUROC < 0.55 after tuning, the environment's onset signal is too subtle for L1 — this is actually a useful finding (validates theory).

### Decision Gate (End of early Week 3)

> "Does the pipeline discriminate between a good detector and a bad one?"

If CUSUM ≈ Random → the environment or the onset labels are broken. Debug before scaling.

---

## Milestone 2 — Theory Core

**Duration**: 10 days (Week 3–4, overlapping with Milestone 1 tail)

**Deliverable**: Paper-ready proof of Theorem 1 (L1 impossibility) + formal write-up of the problem formalization (Sections 2–3 of the paper).

### Work Items

| Item | Output | Hours |
|---|---|---|
| Construct the Le Cam indistinguishability pair | Two MDPs identical under $\mathcal{F}_1$, one hacks | 12 |
| Prove detection latency lower bound | Full proof with all steps | 16 |
| Verify proof: check every inequality, test on concrete example | Hand-computed example matching the bound | 6 |
| Write Section 2 (Problem Formalization) to paper quality | ~4 pages LaTeX | 10 |
| Write Section 3 (Theoretical Results) to paper quality | ~4 pages LaTeX including proof sketch + appendix proof | 12 |
| Attempt Theorem 2 (L2 separates from L1) | Either: full proof OR documented obstruction | 10 |
| Connect theory to empirical predictions | "Theorem 1 predicts CUSUM fails; Theorem 2 predicts Flight Recorder succeeds" | 4 |

**Total estimated effort**: ~70 hours

### Dependencies

- Milestone 0 (onset definition)
- Milestone 1 (partially — having one real environment helps ground the construction)

### Success Criteria

1. Theorem 1 proof is complete, all steps verified, no gaps.
2. The constructed pair of MDPs is concretely instantiable (not just existential — you can build them as environments).
3. Section 2 reads as a standalone contribution: a reader unfamiliar with reward hacking can understand the problem after reading it.
4. Theorem 2 is either proved OR the obstruction is clearly documented as an open problem.
5. A knowledgeable reader (imagine a TMLR reviewer) cannot find a gap in the proof.

### Tests

```python
# Numerical verification of Theorem 1
def test_theorem1_construction():
    """The two MDPs are L1-indistinguishable for the predicted duration."""
    mdp_hack, mdp_clean = construct_le_cam_pair(epsilon=0.1, T=10000)
    
    # Run both, collect L1 observations
    obs_hack = simulate_l1(mdp_hack, T=10000)
    obs_clean = simulate_l1(mdp_clean, T=10000)
    
    # KS test should FAIL to distinguish until t > T/log(1/eps)
    threshold = int(10000 / np.log(1/0.1))  # ≈ 4343
    for t in range(100, threshold, 100):
        _, p = ks_2samp(obs_hack[:t], obs_clean[:t])
        assert p > 0.01, f"Distinguished at t={t}, too early"
    
    # Should be distinguishable by T
    _, p_final = ks_2samp(obs_hack, obs_clean)
    assert p_final < 0.01, "Never distinguishable — construction is wrong"

def test_l2_separates():
    """Under L2, Flight Recorder detects onset on mdp_hack before the L1 bound."""
    mdp_hack, _ = construct_le_cam_pair(epsilon=0.1, T=10000)
    traj = simulate_l2(mdp_hack, T=10000)
    detection_time = FlightRecorderDetector().detect(traj)
    l1_bound = int(10000 / np.log(1/0.1))
    assert detection_time < l1_bound, "L2 did not beat L1 bound"
```

### Risk

- **High**: Theorem 1 may not work with Le Cam cleanly. The reward hacking change-point is endogenous (policy-dependent), which complicates the standard hypothesis testing lower bound machinery. Mitigation: if Le Cam fails, fall back to a Fano-inequality-based argument or weaken to a conditional impossibility (conditioning on a specific policy class).
- **Medium**: Theorem 2 may require assumptions that feel unnatural. Mitigation: if full generality is impossible, prove it for a specific MDP family and conjecture generality.

### Decision Gate (End of Week 4)

> "Is Theorem 1 proved, or do I have a clear alternative that's publishable?"

Options:
- **A**: Full proof works → proceed as planned.
- **B**: Proof works under assumptions → state assumptions explicitly, verify they hold in benchmark environments.
- **C**: Proof has fundamental obstruction → pivot to empirical-only impossibility characterization (weaker paper, but still publishable).

**Do not spend more than 10 days on this.** If stuck at day 10, take option B or C and move on.

---

## Milestone 3 — Tier 1 Complete

**Duration**: 10 days (Week 5–6)

**Deliverable**: All 5 Tier 1 environments implemented, labeled, and evaluated with all baselines.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Implement `NavigationProxy` | Agent navigates; proxy rewards shortcut, true rewards destination | 8 |
| Implement `RewardTampering` | Agent modifies its own reward signal | 8 |
| Implement `CoinRunSimplified` | Spec gaming in procedural env | 10 |
| Implement `CartPoleExploit` | Exploit physics boundary for proxy reward | 6 |
| (Already done) `GridWorldWireheading` | From Milestone 1 | 0 |
| Generate 50 seeds × 5 environments | 250 trajectory files | 12 (compute) |
| Onset labeling + manual validation (spot-check 5 per env) | Validated labels | 8 |
| Implement `FlightRecorderDetector` | SDE-based, L2 access | 10 |
| Implement `EnsembleDisagreementDetector` | Reward model ensemble | 8 |
| Implement `KLMonitorDetector` | Policy KL from reference | 6 |
| Implement `GradientNormDetector` | Gradient magnitude anomaly | 5 |
| Run full evaluation: 6 methods × 5 envs × 50 seeds | Results table with CIs | 8 (compute + analysis) |
| Statistical tests (Wilcoxon, bootstrap CIs) | Significance of all pairwise comparisons | 4 |

**Total estimated effort**: ~93 hours (+ ~20 GPU-hours compute)

### Dependencies

- Milestone 0 (specs)
- Milestone 1 (pipeline, metrics, CUSUM, Random baselines)
- Flight Recorder codebase (existing — import and wrap)

### Success Criteria

1. All 5 environments produce hacking in ≥ 60% of runs.
2. Methods rank differently across environments (no single method dominates all — if one does, environments are too easy).
3. AUROC spread across methods is ≥ 0.2 (meaningful discrimination between good and bad detectors).
4. Bootstrap 95% CIs on AUROC have width < 0.05 (sufficient precision at N=50).
5. At least one environment where Flight Recorder outperforms CUSUM, and at least one where CUSUM is competitive (validates that the benchmark isn't biased toward one method).
6. All results reproducible from a single `make tier1` command.

### Tests

```python
def test_all_envs_hack_rate():
    """Every Tier 1 environment hacks ≥ 60% of runs."""
    for env_name in TIER1_ENVS:
        hack_rate = count_hacking_runs(env_name) / 50
        assert hack_rate >= 0.6, f"{env_name} only hacks {hack_rate*100}%"

def test_method_discrimination():
    """Not all methods have same AUROC (benchmark is discriminative)."""
    aurocs = {m: mean_auroc(m, tier=1) for m in ALL_METHODS}
    assert max(aurocs.values()) - min(aurocs.values()) > 0.2

def test_no_single_winner():
    """No method wins on all 5 environments."""
    winners = [best_method(env) for env in TIER1_ENVS]
    assert len(set(winners)) >= 2

def test_ci_width():
    """All bootstrap CIs < 0.05 width."""
    for m in ALL_METHODS:
        for env in TIER1_ENVS:
            ci = bootstrap_ci(m, env)
            assert ci[1] - ci[0] < 0.05

def test_reproducibility():
    """Running evaluation twice gives identical results."""
    r1 = run_full_evaluation(seed=0)
    r2 = run_full_evaluation(seed=0)
    assert r1 == r2
```

### Risk

- **High**: Some environments may not hack reliably with PPO defaults. Mitigation: tune reward scale / horizon / environment parameters to induce hacking. If an environment cannot be made to hack reliably, replace it (have 2 backup environment designs ready).
- **Medium**: Flight Recorder integration may have API mismatches. Mitigation: write adapter first, test on Milestone 1 data before scaling.
- **Low**: Compute time for 250 training runs may exceed estimate. Mitigation: parallelize across university cluster; Tier 1 environments should be small enough for CPU training.

### Decision Gate (End of Week 6)

> "Do the Tier 1 results tell a story? Can I write a coherent findings paragraph?"

If the results are noisy mush with no pattern → environment difficulty calibration is wrong. Spend 2 extra days adjusting difficulty knobs before proceeding.

---

## Milestone 4 — Scale + Ablations

**Duration**: 12 days (Week 7–8–early 9)

**Deliverable**: Tier 2 environments (4–7), all ablation studies, and the empirical difficulty hierarchy validated.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Implement 4 Tier 2 environments (higher-dim, subtler hacking) | 4 new env files | 32 |
| Generate 50 seeds × 4 Tier 2 envs | 200 trajectory files | 20 (GPU) |
| Onset labeling + validation | Labels checked | 6 |
| Full evaluation on Tier 2 | Results table | 8 (compute) |
| **Ablation: Onset definition sensitivity** ($k$, $\delta$, $\alpha$ sweep) | Table + stability plot | 8 |
| **Ablation: Seed count sensitivity** (N=10,20,30,50) | CI width vs N plot | 4 |
| **Ablation: Clean-run ratio** (50/50 to 90/10) | Ranking stability table | 4 |
| **Ablation: Difficulty knob sweep** per environment | AUROC vs difficulty plot | 8 |
| **Ablation: Algorithm sensitivity** (PPO vs SAC, 10 seeds each) | Cross-algorithm stability table | 12 (compute + analysis) |
| **Ablation: Access level degradation** (FR at L2 vs L1-restricted) | Access-level AUROC comparison | 4 |
| Validate difficulty hierarchy (Tier 1 easier than Tier 2) | Best-method AUROC: Tier 1 > Tier 2 | 2 |
| Validate hacking-type hierarchy (conjecture from Section 3.3) | Ordering plot | 4 |

**Total estimated effort**: ~112 hours (+ ~40 GPU-hours)

### Dependencies

- Milestone 3 (Tier 1 complete, all baselines working)
- Milestone 2 (theory predicts the hierarchy — validate empirically)

### Success Criteria

1. ≥ 4 Tier 2 environments functional and producing hacking.
2. Best-method AUROC on Tier 2 is strictly lower than on Tier 1 (difficulty ordering validated).
3. Onset definition ablation: Spearman rank correlation of method rankings across $k$ values ≥ 0.85.
4. Seed ablation: N=50 gives CI width < 0.05; plot shows diminishing returns beyond N=30.
5. Hacking-type hierarchy empirically matches theoretical prediction (Tampering < Sharp < Gradual) — or documented reason for deviation.
6. At least one ablation produces a surprise (something we learn we didn't expect). If all ablations confirm expectations, we're probably not testing the right things.

### Tests

```python
def test_tier2_harder_than_tier1():
    """Best AUROC on Tier 2 < Best AUROC on Tier 1."""
    best_t1 = max(mean_auroc(m, tier=1) for m in ALL_METHODS)
    best_t2 = max(mean_auroc(m, tier=2) for m in ALL_METHODS)
    assert best_t2 < best_t1

def test_onset_definition_stability():
    """Method rankings stable across k values."""
    rankings = {}
    for k in [100, 500, 1000, 5000]:
        rankings[k] = rank_methods(k=k)
    for k1, k2 in combinations(rankings.keys(), 2):
        rho, _ = spearmanr(rankings[k1], rankings[k2])
        assert rho > 0.85, f"Unstable between k={k1} and k={k2}: rho={rho}"

def test_algorithm_agnosticism():
    """PPO and SAC produce similar onset distributions."""
    ppo_onsets = get_onsets(algo='ppo')
    sac_onsets = get_onsets(algo='sac')
    # KS test should NOT reject (distributions should be similar)
    _, p = ks_2samp(ppo_onsets, sac_onsets)
    assert p > 0.05, "Onset distribution depends on training algorithm"

def test_hierarchy_ordering():
    """Empirical difficulty matches theoretical prediction."""
    type_aurocs = {ht: best_auroc_for_type(ht) for ht in HACKING_TYPES}
    # Tampering should be easiest (highest AUROC)
    assert type_aurocs['tampering'] > type_aurocs['sharp_proxy']
    assert type_aurocs['sharp_proxy'] > type_aurocs['gradual_proxy']
```

### Risk

- **High**: Tier 2 environments may require significant GPU time for training. Mitigation: start training jobs at beginning of week, run analysis while waiting. Budget 3 days of just "waiting for compute."
- **Medium**: Difficulty hierarchy may not cleanly validate. Mitigation: if ordering is partially violated, report honestly as "partially validated conjecture" — this is actually interesting.
- **Low**: Ablations may reveal the benchmark is fragile (rankings change with parameters). Mitigation: this would be a real finding — report it, fix the definition, re-run.

### Decision Gate (End of early Week 9)

> "Do I have enough empirical results to write a compelling Section 6 (Experiments)?"

Minimum requirement: 9 environments (5 Tier 1 + 4 Tier 2), 6 baselines, 4 ablations with clear conclusions. If yes → write. If no → identify the gap and fill it in ≤ 3 days.

---

## Milestone 5 — Paper Draft

**Duration**: 10 days (Week 9–10–early 11)

**Deliverable**: Complete paper draft, all sections, all figures, all tables, submittable quality.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Section 1: Introduction | 3 pages, framing the gap + contributions | 8 |
| Section 4: Benchmark Design | 4 pages, environment descriptions + design principles | 8 |
| Section 5: Metrics | 2 pages (mostly done from implementation docs) | 4 |
| Section 6: Experiments + Results | 5 pages, main table + findings + analysis | 12 |
| Section 7: Analysis & Discussion | 2 pages | 6 |
| Section 8: Related Work | 2 pages, positioning vs SafetyGym, Pan et al., CPD literature | 8 |
| Section 9: Conclusion | 1 page | 2 |
| Appendix A: Full proofs | 3–5 pages | 6 (formatting, already done) |
| Appendix B: Environment specs | Standardized tables | 4 |
| Appendix C: Extended results + ablations | All supplementary tables/figures | 6 |
| Figures: main results plot, hierarchy plot, onset visualization, pipeline diagram | 4–6 figures | 10 |
| Tables: main results, per-environment, ablation summaries | 5–8 tables | 6 |
| Abstract | 250 words, written last | 2 |
| Internal read-through + consistency check | Fix notation, cross-references, claims vs evidence | 6 |

**Total estimated effort**: ~88 hours

### Dependencies

- Milestone 2 (Sections 2–3 already drafted)
- Milestone 3 + 4 (all experimental results available)

### Success Criteria

1. Every claim in the paper is backed by either a theorem or an experiment with confidence intervals.
2. No section contains a TODO or placeholder.
3. The paper passes the "methods section test": a competent researcher could reproduce the benchmark from the paper alone.
4. All figures are publication-quality (vector graphics, readable at printed size, colorblind-safe).
5. The paper reads coherently start-to-finish without the reader needing to jump around.
6. Word count / length is appropriate for TMLR (no artificial padding, no critical compression).

### Tests

```
# Paper quality checklist (manual)
[ ] Every theorem has a proof (or explicit "proof in appendix" pointer)
[ ] Every experimental claim has CI or p-value
[ ] No "we believe" / "it seems" without evidence
[ ] Related work is fair to competitors (no strawmanning)
[ ] Limitations section is honest (not buried)
[ ] Figures have axis labels, legends, captions
[ ] All notation defined on first use
[ ] Cross-references resolve correctly
[ ] Bibliography is complete (no [?] artifacts)
[ ] Abstract matches the actual paper content
```

### Risk

- **Medium**: Writing may reveal gaps in the experimental evidence (a claim you want to make but don't have data for). Mitigation: budget 2 days for "fill-gap experiments" prompted by writing.
- **Low**: The paper may be too long. Mitigation: TMLR has no page limit; but if > 30 pages, consider what to move to appendix.

### Decision Gate (End of early Week 11)

> "Would I submit this draft as-is if the deadline were tomorrow?"

If no → identify the single weakest section and fix it. Do not rewrite the whole paper.

---

## Milestone 6 — Polish + Submit

**Duration**: 7 days (Week 11–12)

**Deliverable**: Submitted paper + released codebase + arXiv preprint.

### Work Items

| Item | Output | Hours |
|---|---|---|
| Get 1–2 external reads (peers, if available) | Feedback notes | 2 (your time) + wait |
| Address feedback (content) | Revised sections | 10 |
| Address feedback (clarity/typos) | Clean copy | 4 |
| Prepare TMLR submission (formatting, author info, cover letter) | Submission-ready PDF | 3 |
| Write TMLR cover letter (positioning statement from blueprint) | 1-page letter | 2 |
| Code release: clean API, README, examples, `pip install` test | Public GitHub repo | 8 |
| Reproduce all main results from clean install | Pass/fail | 4 |
| Post to arXiv (simultaneous with TMLR submission) | arXiv preprint | 2 |
| Create benchmark website / leaderboard (minimal — can be a GitHub page) | Live URL | 4 |

**Total estimated effort**: ~39 hours

### Dependencies

- Milestone 5 (complete draft)

### Success Criteria

1. TMLR submission compiles without errors.
2. Cover letter articulates why this is a formalization paper (not just a benchmark).
3. `pip install rhob && python -m rhob.evaluate --method random --env gridworld` works on a fresh machine.
4. README includes: install instructions, 5-minute quickstart, full reproduction instructions.
5. ArXiv preprint is live.
6. At least 1 external reader did not find any fatal flaw.

### Tests

```bash
# Fresh environment reproduction test
conda create -n rhob_test python=3.10 -y
conda activate rhob_test
pip install rhob
python -m rhob.reproduce --tier 1 --method cusum
# Should output: "Reproduction successful. AUROC = X.XX (paper: X.XX, within CI)"

# Submission checklist
pdflatex main.tex  # No errors
bibtex main        # No warnings
grep -c "TODO" main.tex  # 0
grep -c "\\\\cite{}" main.tex  # 0
```

### Risk

- **Low**: TMLR formatting may have edge cases. Mitigation: use their official template from day 1 (set up in Milestone 5).
- **Medium**: External feedback may request substantial changes. Mitigation: only incorporate changes that fix correctness issues or major clarity problems. Do not rewrite based on taste preferences.

---

## Summary Dashboard

| Milestone | Weeks | Hours | Cumulative | Key Deliverable |
|---|---|---|---|---|
| **M0: Foundation** | 1 | 32 | 32 | Locked specs + API contracts |
| **M1: Vertical Slice** | 2–3 | 47 | 79 | End-to-end pipeline working |
| **M2: Theory** | 3–4 | 70 | 149 | Theorem 1 proved + Sections 2–3 written |
| **M3: Tier 1** | 5–6 | 93 | 242 | 5 envs × 6 methods × 50 seeds |
| **M4: Scale + Ablations** | 7–9 | 112 | 354 | 9 envs + full ablation battery |
| **M5: Paper Draft** | 9–11 | 88 | 442 | Complete submittable draft |
| **M6: Polish + Submit** | 11–12 | 39 | 481 | Submitted to TMLR |

**Total: ~481 working hours over 12 weeks ≈ 40 hours/week**

---

## Critical Path (What Blocks Everything)

```
M0: Onset Definition ─────┬──→ M1: One Environment ──→ M3: All Tier 1 ──→ M4: Tier 2
                          │                                      │              │
                          └──→ M2: Theory ─────────────────────────────────────┘
                                                                 │              │
                                                                 ▼              ▼
                                                            M5: Write ──→ M6: Submit
```

**Parallelism**: M2 (Theory) can run in parallel with M1 and partially with M3. Start GPU training jobs for M3 environments while proving theorems.

---

## If You Only Have 8 Weeks

Cut scope:

| Keep | Drop |
|---|---|
| Tier 1 (5 environments) | Tier 2 |
| Theorem 1 | Theorem 2 |
| 4 baselines (Random, CUSUM, Flight Recorder, Ensemble) | KL Monitor, Gradient Norm |
| 3 ablations (onset sensitivity, seed count, hierarchy) | Algorithm sensitivity, access degradation, clean ratio |
| 30 seeds per environment | 50 seeds |

This compresses to ~320 hours / 8 weeks / 40 hrs per week. Still submittable. The paper is weaker (no Tier 2, no Theorem 2) but the core contributions (formalization + Theorem 1 + Tier 1 benchmark + first comparison) remain intact.

---

## Weekly Cadence for the Solo Researcher

| Day | Focus |
|---|---|
| **Monday** | Hardest intellectual task (proofs, algorithm design, tricky implementations) |
| **Tuesday** | Continue Monday's work; aim to complete by EOD |
| **Wednesday** | Engineering (implementations, debugging, pipeline work) |
| **Thursday** | Run experiments; write while waiting for results |
| **Friday** | Analysis of results; planning next week; writing |
| **Weekend** | Writing only (low-energy but high-value) OR rest |

**Non-negotiable rule**: Never start a new milestone until the previous one's decision gate passes. Accumulating unfinished milestones is the #1 solo-researcher failure mode.


