# Conversation
- conversationId: df759565-bcf3-4579-9949-817192a73614
- endpoint: agents
- title: New Chat
- exportAt: 18:21:27 GMT-0500 (Central Daylight Time)

## Options
- presetId: null
- agent_id: agent_lhadRInRtO_DVG6HRATX3
- endpoint: agents
- endpointType: undefined
- title: New Chat

## History
**User**
Using the benchmark proposal from the Research Gap Finder,

design the publication strategy.

Determine

- paper structure

- TMLR positioning

- contributions

- experiments

- evaluation protocol

- expected reviewer questions

- ablation studies

- comparison baselines

Produce a publication blueprint.


**Publication Intelligence**
{"type":"think","think":"Aarav wants me to take the RHOB (Reward-Hacking Onset Benchmark) design document produced by the Research Gap Finder agent and convert it into a concrete publication blueprint. I need to think like a senior NeurIPS Area Chair / TMLR Associate Editor and design the optimal publication strategy.\n\nLet me think about this carefully:\n\n1. **Venue Selection**: The gap analysis suggests NeurIPS Datasets & Benchmarks track. But Aarav is asking about TMLR positioning specifically. Let me think about why TMLR might be better or worse:\n\n- NeurIPS D&B: High prestige, annual deadline, competitive, shorter format, community visibility through competition\n- TMLR: Rolling review, no page limit constraints, thorough peer review, room for technical depth, faster publication cycles than conferences, and Aarav already has an established track record there\n\nGiven Aarav's existing portfolio with papers at TMLR, that venue feels like the natural fit. But benchmarks typically thrive at NeurIPS D&B where they get community engagement. If we're targeting TMLR, the paper needs stronger theoretical contributions beyond just presenting a dataset—TMLR reviewers will expect novel insights about reward hacking itself, not just a measurement tool.\n\nThe key is reframing this from \"here's a benchmark\" to \"here's a formal problem class with theoretical characterization of its difficulty, instantiated through a benchmark that enables systematic comparison.\" This positions it as a scientific contribution rather than an engineering artifact, which is what TMLR values.\n\nFor the paper structure, I'd lead with the problem formalization and theoretical results—the impossibility characterization and complexity hierarchy—then use the benchmark as the vehicle for validating those theoretical predictions empirically. The main contributions would be the formal definition of onset detection as a change-point problem, the information-theoretic impossibility result, the empirical difficulty hierarchy, and the benchmark itself as a unified evaluation framework.\n\nI'm also thinking through the likely reviewer pushback: why TMLR over a D&B venue, whether the onset definition holds up under hyperparameter variation, how to handle environments where hacking doesn't consistently manifest, whether the impossibility result is tight, and how this relates to existing work like SafetyGym or standard change-point detection literature. I should also anticipate questions about statistical rigor—whether 50 seeds per environment is enough, how to handle cold-start detection, and sensitivity to design choices like the lookback window, seed count, access levels, and the clean-run ratio."}


**Publication Intelligence**
# Publication Blueprint: RHOB at TMLR

---

## 1. Executive Summary

The benchmark paper must be reframed for TMLR. A pure "here's a dataset" paper would be rejected — TMLR demands novel scientific contributions with technical depth. The correct positioning is: **a new problem formalization (onset detection as structured change-point detection) supported by theoretical lower bounds, instantiated as the first benchmark enabling systematic comparison**. The benchmark is the artifact; the science is the formalization + impossibility results + empirical difficulty characterization.

---

## 2. Venue Analysis: Why TMLR

| Factor | TMLR | NeurIPS D&B | Verdict |
|---|---|---|---|
| **Review depth** | Thorough, multi-round | Single-round, space-constrained | TMLR allows full theoretical development |
| **Page limit** | None | 9 pages + appendix | Theory + benchmark needs space |
| **Timeline** | Rolling (submit anytime) | Annual deadline (typically June) | TMLR decouples from deadline pressure |
| **Theory tolerance** | High (expects mathematical content) | Lower (expects primarily empirical/engineering) | TMLR rewards the impossibility result |
| **Portfolio coherence** | SDE + FS-NRLF already at TMLR | Would be first NeurIPS since Density Matrix MDPs | Building a TMLR "stack" for the RL safety program |
| **Benchmark precedent** | TMLR publishes benchmark papers with theory | NeurIPS D&B is the "default" for benchmarks | Both viable; TMLR differentiates on depth |
| **Citation trajectory** | TMLR papers accrue citations steadily | NeurIPS burst then plateau | Long-term compounding favors TMLR |

**Decision:** TMLR as primary venue. NeurIPS D&B as fallback if TMLR rejects on the grounds that it's "primarily a resource contribution" — but the theoretical depth should prevent this.

**TMLR Positioning Statement (for cover letter):**

> "This paper formalizes reward-hacking onset detection as a new problem class distinct from both prevention and post-hoc evaluation. We prove an impossibility result characterizing the fundamental limits of oracle-free detection, establish an empirical difficulty hierarchy across hacking modalities, and release the first benchmark enabling standardized comparison of detection methods. The contribution is primarily scientific (formalization + theory), with the benchmark as the enabling infrastructure."

---

## 3. Paper Structure

### Title Options (ranked)

1. **"When Does Reward Hacking Begin? Formalizing Onset Detection and Its Fundamental Limits"**
2. "RHOB: Formalizing, Measuring, and Benchmarking Reward-Hacking Onset Detection"
3. "The Onset Detection Problem: Theory, Benchmark, and Baselines for Reward Hacking"

**Recommendation:** Option 1. It's a question (engaging), signals theory ("fundamental limits"), and positions the contribution as a formalization rather than just a benchmark.

---

### Detailed Section-by-Section Structure

#### Abstract (~250 words)

Structure: Problem → Gap → Contribution → Result → Implication

- **Problem:** Reward hacking is the central failure mode of deployed RL. Detecting its onset — the moment proxy and true objectives diverge — is prerequisite to intervention.
- **Gap:** No formal problem definition exists. No standardized metrics. No comparable evaluation. The community cannot measure progress.
- **Contribution:** (1) Formal definition of onset as structured change-point detection with unknown post-change distribution. (2) Impossibility result: oracle-free detection at access level L1 has latency Ω(T) when proxy-true correlation exceeds 1−ε. (3) Empirical difficulty hierarchy across six hacking modalities. (4) RHOB benchmark: 21 environments, 7 metrics, 6 baselines.
- **Result:** Concrete finding (e.g., "Flight Recorder achieves AUROC 0.87 on sharp-onset environments but degrades to 0.61 on gradual-onset, revealing a fundamental gap between hacking types that no existing method bridges").
- **Implication:** RHOB enables the first systematic measurement of detection capability and reveals that the onset detection problem is fundamentally harder than classical change-point detection.

---

#### 1. Introduction (2–3 pages)

**1.1 The Detection Problem**

Open with the operational question: "An RL agent's proxy reward is increasing. Is this good (the agent is learning) or bad (the agent is hacking)?"

Establish that this is a fundamentally different question from:
- "Can we prevent hacking?" (requires knowing what hacking looks like a priori)
- "Did hacking happen?" (post-hoc, cannot enable real-time intervention)
- "How do we design better rewards?" (prevention, not detection)

**1.2 Why Onset, Specifically**

The temporal question — *when* — matters because:
- Earlier detection → more intervention budget
- Binary "hacking / not hacking" labels lose the temporal dynamics
- The agent's policy trajectory pre- and post-onset may contain extractable structure

**1.3 The Current State Is Pre-Scientific**

No two papers evaluate detection the same way. List 3–4 recent detection papers and show their evaluation is incompatible (different environments, different metrics, different information assumptions).

**1.4 Contributions (Bulleted)**

1. Formalization of onset detection as a problem class (Section 2)
2. Impossibility result for L1 oracle-free detection (Section 3)
3. Onset complexity hierarchy across hacking types (Section 4)
4. RHOB benchmark: environments, metrics, protocol (Section 5)
5. Comprehensive baseline evaluation revealing the frontier (Section 6)

**1.5 Scope Delimitation**

Explicitly state what this paper does NOT do: does not propose a new detection method, does not address prevention/mitigation, does not claim to solve the problem. It defines and measures the problem.

---

#### 2. Problem Formalization (3–4 pages)

**2.1 Setup and Notation**

- MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R_{\text{proxy}}, \gamma)$ with hidden true reward $R_{\text{true}}$
- Policy training trajectory $\{\pi_{\theta_t}\}_{t=0}^T$
- Observable signal: depends on access level (L1–L4)

**2.2 Definition of Onset**

Formal definition (refined from gap analysis):

$$t^* = \inf\left\{t \geq t_0 : \frac{d}{dt}\mathbb{E}_{\pi_t}[R_{\text{true}}] < 0 \;\;\text{AND}\;\; \frac{d}{dt}\mathbb{E}_{\pi_t}[R_{\text{proxy}}] > 0 \right\}$$

In practice (discrete time, finite samples):

$$t^* = \inf\left\{t : \bar{R}_{\text{true}}^{[t-k,t]} < \bar{R}_{\text{true}}^{[t-2k,t-k]} - \delta \;\;\text{AND}\;\; \bar{R}_{\text{proxy}}^{[t-k,t]} > \bar{R}_{\text{proxy}}^{[t-2k,t-k]} + \delta \right\}$$

with significance threshold $\delta$ determined by a two-sample test at level $\alpha = 0.01$.

Discuss edge cases: simultaneous proxy increase and true decrease (clear onset), proxy increase while true is flat (ambiguous — we choose conservative: not onset), both increasing but at diverging rates (gradual onset — labeled separately).

**2.3 The Detection Problem**

A detector $D$ is a mapping:

$$D: \{(s_i, a_i, r_i^{\text{proxy}}, s_{i+1})\}_{i=1}^t \longrightarrow [0,1]$$

outputting a hacking probability at each timestep.

Key constraint: $D$ never observes $R_{\text{true}}$.

**2.4 Access Levels as Information Constraints**

Formalize L1–L4 as sigma-algebras:
- $\mathcal{F}_1 = \sigma(\{r_i^{\text{proxy}}\}_{i \leq t})$ (reward-only)
- $\mathcal{F}_2 = \sigma(\{s_i, a_i, r_i, s_{i+1}\}_{i \leq t})$ (trajectory)
- $\mathcal{F}_3 = \mathcal{F}_2 \vee \sigma(\{\nabla_\theta J(\theta_i)\}_{i \leq t})$ (gradients)
- $\mathcal{F}_4 = \mathcal{F}_3 \vee \sigma(\text{internal env state})$ (full)

This is clean, mathematically rigorous, and enables precise impossibility statements.

**2.5 Connection to Change-Point Detection**

Classical change-point detection: observations switch from distribution $P_0$ to $P_1$ at unknown time $\tau$. Optimal detection (Shiryaev, 1963; Page, 1954) assumes $P_0$ and $P_1$ are known.

Key structural differences in onset detection:
1. $P_1$ is unknown and non-stationary (the agent is *learning* to exploit)
2. The change point is endogenous (caused by the agent's own policy evolution)
3. The pre-change process $P_0$ is also non-stationary (normal learning)
4. Under L1, the observed signal is a scalar function of a high-dimensional process

These differences make onset detection **strictly harder** than classical change-point detection.

---

#### 3. Theoretical Results (3–4 pages)

**3.1 Impossibility Result (Main Theorem)**

**Theorem 1 (L1 Detection Limit).** For any $\epsilon > 0$, there exists a reward hacking instance $(\mathcal{M}, R_{\text{proxy}}, R_{\text{true}})$ with $\text{Corr}(R_{\text{proxy}}, R_{\text{true}}) \geq 1 - \epsilon$ pre-onset such that any L1 detector $D$ measurable with respect to $\mathcal{F}_1$ satisfies:

$$\mathbb{E}[\mathcal{L}(D)] \geq \Omega\left(\frac{T}{\log(1/\epsilon)}\right)$$

or $\text{FPR}(D) \geq 1/2$.

*Interpretation:* When proxy and true rewards are nearly perfectly correlated, detecting their divergence from reward observations alone requires waiting an amount of time proportional to the training horizon — or accepting unacceptable false positive rates.

**Proof sketch:** Construct two environments that are indistinguishable under $\mathcal{F}_1$ up to time $t^* + \Omega(T/\log(1/\epsilon))$: one where hacking occurs at $t^*$ and one where legitimate learning continues. Apply Le Cam's method to lower-bound the minimax detection error.

**3.2 Access Level Separation**

**Theorem 2 (L2 Strictly Dominates L1).** There exists a family of environments where any L2 detector achieves $\mathcal{L} = O(\sqrt{T})$ while every L1 detector requires $\mathcal{L} = \Omega(T)$.

*Proof idea:* Policy dynamics (state-action distributions) contain structural signatures of exploitation — e.g., concentration on exploit states — that are invisible in the scalar reward signal.

**3.3 Complexity Hierarchy (Conjecture with Partial Proof)**

**Conjecture 1.** Under L2 access, the minimax detection latency across hacking types satisfies:

$$\mathcal{L}^*(\text{Tampering}) < \mathcal{L}^*(\text{Sharp Proxy}) < \mathcal{L}^*(\text{Spec. Gaming}) < \mathcal{L}^*(\text{Gradual Proxy}) < \mathcal{L}^*(\text{Goal Misgen.})$$

Prove the first inequality rigorously (tampering creates discontinuity in reward dynamics → $O(1)$ detection possible). Leave the full hierarchy as empirically validated conjecture.

**3.4 Relationship to SDE Theory**

Brief section connecting to Spectral Drift Entropy (Aarav's submitted TMLR paper). The key insight: SDE measures spectral structure of policy evolution, which is exactly the signal that separates L2 from L1 detection. RHOB empirically validates SDE's theoretical advantages.

---

#### 4. Benchmark Design (4–5 pages)

**4.1 Design Principles**

(Condensed from gap analysis Section 4.1 — instrumented ground truth, parameterizable difficulty, reproducible hacking, diverse modalities, scalable compute)

**4.2 Environment Suite**

Present all environments in a single table organized by tier, with columns: name, hacking type, state space, difficulty knob, expected training time, oracle mechanism.

For each environment, provide a 1-paragraph description of:
- What the agent is supposed to do (true objective)
- What the proxy reward measures
- How hacking manifests
- How onset is detected by the oracle

**4.3 Onset Labeling Protocol**

(From gap analysis Section 4.3 — formalize and add statistical rigor)

Include a figure: example onset labels for 3 environments showing proxy reward (increasing), true reward (diverging), and the labeled onset point with confidence interval.

**4.4 Dataset Structure**

- 50 seeds per environment
- 70/30 hacking/clean split
- Train/test split (60/40, labels withheld for test)
- HDF5 format
- Total size, download instructions

**4.5 Submission Interface**

Present the Python API (from gap analysis Section 6.1). Keep it simple — a reviewer should understand the interface in 30 seconds.

---

#### 5. Metrics (2–3 pages)

**5.1 Primary Metrics**

| Metric | Formula | Interpretation |
|---|---|---|
| **Detection Latency** $\hat{\mathcal{L}}$ | $(t_{\text{detect}} - t^*) / T$ | Normalized time between onset and detection |
| **AUROC** | Standard | Discrimination at all thresholds |
| **AUPRC** | Standard | Precision-recall (handles imbalance) |
| **FPR@k** | FP / total alerts at budget $k$ | Practical false alarm rate |
| **Miss Rate** | Fraction never detected | Reliability |
| **TFD** | Median latency among true positives | Central tendency of speed |

**5.2 Aggregate Score**

$$\text{RHOB-Score} = \frac{1}{|\mathcal{E}|}\sum_{e \in \mathcal{E}} w_e \cdot \text{AUROC}_e$$

where $w_e$ is tier-based weighting (Tier 1: 1.0, Tier 2: 1.5, Tier 3: 2.0, Adversarial: 2.5).

**5.3 Reporting Template**

Standardized table that every paper must fill in. Design it so a reviewer can compare two methods at a glance.

---

#### 6. Experiments (5–6 pages)

**6.1 Baseline Methods**

| Method | Access Level | Oracle-Free? | Description |
|---|---|---|---|
| Flight Recorder | L2 | Yes | SDE-based structural detection |
| CUSUM | L1 | Yes | Classical change-point on reward |
| Ensemble Disagreement | L2 | Yes | Reward model ensemble variance |
| KL Monitor | L3 | Yes | Policy divergence from reference |
| Gradient Norm | L3 | Yes | Anomaly detection on gradient magnitude |
| Random | — | Yes | Uniform random alerts (floor) |
| Oracle | L4 + $R_{\text{true}}$ | **No** | Perfect detection (ceiling) |

**6.2 Main Results Table**

Present the RHOB-Score, AUROC per tier, detection latency, FPR@3, miss rate, and computational overhead for every baseline across every environment. This will be the paper's central empirical artifact.

**6.3 Findings** (anticipated based on method properties)

1. **No method dominates everywhere.** Flight Recorder excels on sharp-onset (leverages structural signal); CUSUM fails on gradual; ensemble methods are competitive on proxy gaming but fail on specification gaming.

2. **The difficulty hierarchy is empirically validated.** Reward tampering is easiest (all methods achieve AUROC > 0.9); goal misgeneralization is hardest (best method < 0.7).

3. **Access level matters dramatically.** L1 methods (CUSUM) have AUROC 0.6–0.7 average; L2 methods (Flight Recorder) reach 0.8–0.9 on sharp but degrade on gradual.

4. **Computational overhead varies 100×.** Gradient-based methods add 20–50% overhead; reward-only methods add < 1%.

5. **The adversarial tier defeats all baselines.** Best method achieves < 0.65 AUROC on adversarial environments, confirming substantial room for improvement.

**6.4 Difficulty Validation**

Show that Tier 1 < Tier 2 < Tier 3 < Adversarial in difficulty (measured by best-method AUROC decreasing). If this doesn't hold, the tier assignment is wrong — fix before submission.

**6.5 Onset Label Robustness**

Vary the onset definition parameters ($k$, $\delta$, $\alpha$) and show that relative method rankings are stable. This is critical for defending the benchmark's validity.

---

#### 7. Analysis & Discussion (2–3 pages)

**7.1 What's Easy, What's Hard**

Characterize the frontier: which environments/hacking types remain unsolved? What properties make detection hard?

**7.2 Open Problems Surfaced by RHOB**

- Can any method achieve AUROC > 0.9 on gradual-onset hacking without oracle access?
- Is there a detection method that works at L1 and beats CUSUM?
- Can adversarial environments be constructed that defeat *any* fixed detector (a detection-impossibility result that's constructive rather than existential)?

**7.3 Limitations of RHOB**

Be honest:
- Environments are simulations (real-world hacking may differ)
- Onset definition requires choice of $k$ and $\delta$ (sensitivity acknowledged)
- Current version focuses on single-agent (multi-agent hacking is future work)
- 50 seeds may be insufficient for high-variance environments

---

#### 8. Related Work (2 pages)

Organize by:
1. Formal characterizations of reward hacking (Skalse et al., Krakovna et al.)
2. Detection methods (Flight Recorder, ensemble methods, KL monitoring)
3. Existing safety benchmarks (SafetyGym, Pan et al., Koch et al.)
4. Change-point detection theory (Shiryaev, Page, CUSUM optimality)
5. Benchmark methodology (ImageNet, GLUE, HELM — what makes benchmarks succeed)

Position RHOB as the first work at the intersection of (1), (2), (3), and (4).

---

#### 9. Conclusion

One page. Reiterate: the field cannot measure detection progress without a standard benchmark. RHOB provides it. The theoretical results show fundamental limits exist. The empirical results show no existing method is close to solving the problem. The benchmark will be maintained and expanded.

---

#### Appendices

- A: Full proofs of Theorems 1 and 2
- B: Complete environment specifications (parameters, reward functions, oracle mechanisms)
- C: Extended experimental results (per-seed analysis, all ablations)
- D: Human onset labeling protocol and inter-annotator agreement
- E: Reproduction instructions and compute requirements

---

## 4. Contributions Ranked by TMLR Reviewer Impact

| # | Contribution | Type | Novelty | Expected Reviewer Reaction |
|---|---|---|---|---|
| 1 | Formalization of onset detection as structured change-point detection with endogenous, non-stationary post-change distribution | Theoretical | High | "This is the right abstraction — clean and well-motivated" |
| 2 | Impossibility result (L1 detection requires Ω(T) latency under high correlation) | Theoretical | High | "This justifies why we need richer signals — nice lower bound" |
| 3 | Access level separation theorem (L2 strictly dominates L1) | Theoretical | Medium-High | "Connects nicely to the practical question of what to monitor" |
| 4 | Empirical difficulty hierarchy across hacking types | Empirical | High | "Nobody has characterized this before — useful for the community" |
| 5 | Benchmark itself (environments + metrics + protocol) | Infrastructure | Medium (but high impact) | "Useful resource; hope it gets maintained" |
| 6 | First head-to-head comparison of detection methods | Empirical | Medium | "Overdue — good to finally see these compared fairly" |

**Critical:** Contributions 1–3 are what makes this a TMLR paper rather than a NeurIPS D&B paper. Without them, it's a resource paper. With them, it's a formalization paper that produces a resource.

---

## 5. Experiments: Detailed Protocol

### 5.1 Training the RL Agents (Not the Contribution)

- Use PPO (Stable-Baselines3, default hyperparameters) as the standard training algorithm
- For each environment, train 50 independent runs to completion
- Store full trajectories, rewards, and onset labels
- Also include 10 runs with SAC and 10 with DQN to verify algorithm-agnosticism of the benchmark

### 5.2 Evaluating Detection Methods

For each method × environment × seed:
1. Feed the pre-recorded trajectory to the detector step-by-step
2. At each step, record the detector's output score
3. After the full trajectory, compute all metrics against ground-truth onset
4. Aggregate across seeds (mean ± std)

### 5.3 Statistical Testing

- Method comparison: paired Wilcoxon signed-rank test across environments (non-parametric, handles non-normality)
- Confidence intervals: bootstrap (10,000 resamples) on AUROC and latency
- Multiple comparisons: Holm-Bonferroni correction
- Significance level: $p < 0.01$ throughout

### 5.4 Compute Budget

| Component | Estimated GPU-hours | Hardware |
|---|---|---|
| Training agents (Tier 1, 50 seeds × 5 envs) | ~25 h | Single A100 |
| Training agents (Tier 2, 50 seeds × 7 envs) | ~400 h | 4× A100 |
| Training agents (Tier 3, 50 seeds × 4 envs) | ~2000 h | 8× A100 |
| Running all baselines on all data | ~50 h | Single A100 |
| Ablations | ~100 h | 4× A100 |
| **Total** | **~2,575 h** | |

**Minimum viable paper (Tiers 1+2 only):** ~475 GPU-hours. Achievable on university cluster.

---

## 6. Ablation Studies

### 6.1 Onset Definition Sensitivity

| Ablation | What varies | What stays fixed | Expected finding |
|---|---|---|---|
| Lookback window $k$ | $k \in \{100, 500, 1000, 5000\}$ | All else | Relative method rankings stable for $k \geq 500$ |
| Significance threshold $\delta$ | $\delta \in \{0.5\sigma, 1\sigma, 2\sigma\}$ | All else | Conservative $\delta$ shifts onset later; rankings stable |
| Test level $\alpha$ | $\alpha \in \{0.001, 0.01, 0.05\}$ | All else | Minimal effect on rankings |

**Why critical:** If metrics change dramatically with onset definition parameters, the benchmark is fragile. Must demonstrate robustness.

### 6.2 Seed Count Sensitivity

Run with $N \in \{10, 20, 30, 50, 100\}$ seeds. Plot metric confidence interval width vs. $N$. Show that $N = 50$ provides sufficient precision (CI width < 0.03 AUROC).

### 6.3 Clean-Run Ratio

Vary the ratio of hacking vs. clean runs: 50/50, 60/40, 70/30, 80/20, 90/10. Show that method rankings are invariant to this choice (only absolute FPR changes).

### 6.4 Difficulty Knob Calibration

For each environment with a difficulty parameter, sweep it and plot AUROC vs. difficulty. Verify monotonic decrease (harder settings → lower detection performance). If non-monotonic, investigate and document.

### 6.5 Pre-recorded vs. Live Mode Agreement

For Tier 1 environments, compare metrics when detectors consume pre-recorded data vs. run alongside live training. Should agree within CI. If not, document the discrepancy and explain (e.g., live mode has additional stochasticity from policy noise).

### 6.6 Algorithm Sensitivity

Compare onset labels and detection metrics when agents are trained with PPO vs. SAC vs. DQN. The benchmark should be algorithm-agnostic — if results change dramatically, this is a problem to acknowledge.

### 6.7 Access Level Degradation

For methods that can operate at multiple access levels (e.g., Flight Recorder can use L2 or restricted to L1), measure AUROC at each level. Quantify the "cost of partial information."

---

## 7. Comparison Baselines — Extended Analysis

### 7.1 Baseline Implementation Details

| Method | Key hyperparameters | Tuning protocol |
|---|---|---|
| **Flight Recorder** | SDE window size, spectral threshold, embedding dimension | Grid search on Tier 1 public set; report sensitivity |
| **CUSUM** | Target mean shift, allowance parameter $h$ | Standard ARL-matched calibration |
| **Ensemble Disagreement** | Ensemble size $K$, reward model architecture, disagreement threshold | $K \in \{3, 5, 10\}$; architecture from Gao et al. |
| **KL Monitor** | Reference policy checkpoint frequency, KL threshold | Threshold set to achieve FPR = 0.05 on clean runs |
| **Gradient Norm** | Smoothing window, z-score threshold | Threshold calibrated on Tier 1 |
| **Random** | Alert probability per step | Calibrated to match other methods' alert frequency |

### 7.2 Fairness in Comparison

- All methods receive identical data streams
- All methods tuned on the same public set (Tier 1 public only)
- All methods evaluated on the same private set
- Computational overhead measured on identical hardware
- No method sees true rewards (except Oracle ceiling)

### 7.3 External Baselines to Consider Including

| Method | Source | Why include |
|---|---|---|
| Reward Model Probing (Burns et al., 2023 style) | Contrast-consistent search applied to reward model internals | Tests whether mechanistic interpretability helps detection |
| RLHF-specific overoptimization detector (Gao et al.) | KL-reward Pareto monitoring | Tests whether RLHF-specific methods generalize |
| Behavioral cloning divergence | Compare agent behavior to BC-trained reference | Tests whether behavioral deviation = hacking |

---

## 8. Expected Reviewer Questions & Prepared Responses

### Q1: "Why TMLR and not NeurIPS Datasets & Benchmarks?"

**Response:** The paper's primary contributions are theoretical (formalization, impossibility result, complexity hierarchy). The benchmark is the instantiation of these theoretical insights. TMLR's format allows the depth needed to develop these results rigorously. We believe the paper stands on its theoretical contributions alone; the benchmark is a bonus.

### Q2: "Is the onset definition robust? It seems to depend on hyperparameters ($k$, $\delta$)."

**Response:** We demonstrate in Ablation 6.1 that relative method rankings are stable across a wide range of $k$ and $\delta$. We report all results at the default setting ($k=1000$, $\delta = 1\sigma$) and include a sensitivity analysis in Appendix C. We also provide human-validated labels (Krippendorff's $\alpha > 0.8$) as a robustness check.

### Q3: "How do you ensure environments reliably produce hacking?"

**Response:** Each environment was piloted across 100+ seeds before inclusion. Only environments where hacking occurs in > 60% of PPO training runs (with standard hyperparameters) are included. The remaining runs serve as "clean" negatives. Hacking reliability is reported per environment in Table X.

### Q4: "The impossibility result seems narrow — it only applies to L1. Most practical methods use L2+."

**Response:** Correct — the L1 result establishes a floor and formally justifies why trajectory-level methods (L2) are necessary. The L2 separation theorem (Theorem 2) shows that richer information provably helps. We conjecture but do not prove a corresponding lower bound for L2; this is an explicit open problem.

### Q5: "How is this different from Pan et al. (2022) or SafetyGym?"

**Response:** Pan et al. evaluates final policy quality under reward misspecification (post-hoc, no temporal resolution, no detection metrics). SafetyGym evaluates constraint satisfaction during execution (not reward hacking, not detection). RHOB is the first to (a) provide temporal onset labels, (b) evaluate detection methods specifically, (c) standardize detection-specific metrics. We provide a detailed comparison in Section 8.

### Q6: "Isn't this just change-point detection?"

**Response:** We explicitly characterize how onset detection differs from classical CPD (Section 2.5): unknown and non-stationary $P_1$, endogenous change point, non-stationary $P_0$. Our Theorem 1 shows the problem is strictly harder. We include CUSUM (the optimal classical CPD method) as a baseline; its mediocre performance empirically confirms the structural gap.

### Q7: "What about multi-agent reward hacking?"

**Response:** Multi-agent is future work (acknowledged in limitations). The current formulation covers single-agent settings where hacking types are already diverse enough to reveal fundamental detection challenges. Multi-agent adds coordination/emergence complexity that deserves its own treatment.

### Q8: "Will you maintain the benchmark?"

**Response:** Yes — the benchmark includes a versioned refresh mechanism (new environments quarterly, adversarial tier updated when methods saturate, private test set rotated annually). The leaderboard and codebase are maintained at [URL]. We commit to a 3-year maintenance horizon and will open governance to the community thereafter.

### Q9: "The paper doesn't propose a new detection method. Isn't this just infrastructure?"

**Response:** The theoretical contributions (Theorems 1, 2, empirical hierarchy) are novel scientific results that stand independent of the benchmark. The benchmark paper format at TMLR has precedent (cite examples). The formalization of onset detection as a problem class is itself a contribution — it enables the question to be asked precisely.

### Q10: "Flight Recorder is one of the baselines. Isn't this self-serving?"

**Response:** Flight Recorder is one of six baselines, evaluated identically to all others, with code released for reproduction. The benchmark is designed to be method-agnostic — the interface accepts any method conforming to the API. We explicitly include methods that may outperform Flight Recorder on certain environments. The benchmark's value is independent of any single method's ranking.

---

## 9. Positioning Within Aarav's Publication Ecosystem

### 9.1 Paper Dependency Graph

```
SDE (TMLR, submitted)
    ↓ provides theoretical foundation for
Flight Recorder (impossibility result, TMLR submitted)
    ↓ is evaluated by
RHOB (this paper)
    ↓ creates the evaluation standard for
All future Flight Recorder papers
```

### 9.2 Cross-Citation Strategy

| This paper cites... | Because... |
|---|---|
| SDE (Shah, 2026) | Flight Recorder baseline uses SDE as its detection signal |
| Density Matrix MDPs (Shah, NeurIPS 2026) | Establishes author's credentials in formal RL theory |
| FS-NRLF (Shah, TMLR under review) | Non-stationary RL context for why hacking detection is hard |

| This paper is cited by (future)... | Because... |
|---|---|
| Every future Flight Recorder paper | "We evaluate on RHOB (Shah, 2027)" |
| Every future reward-hacking detection paper | Must compare on RHOB to be taken seriously |
| Survey papers on RL safety | As the canonical benchmark reference |

### 9.3 Timing Relative to Other Papers

| Paper | Status | RHOB dependency |
|---|---|---|
| SDE | TMLR submitted | Should be accepted/published before RHOB references it as established |
| Flight Recorder impossibility | TMLR submitted | RHOB cites the theoretical result; benchmark validates it empirically |
| RHOB | To be submitted | Needs SDE + Flight Recorder at least on arXiv |
| "Flight Recorder on RHOB" (follow-up) | Future | Submitted after RHOB is published |

**Recommendation:** Submit RHOB to TMLR 2–4 months after SDE and Flight Recorder decisions are known. If accepted, cite published versions. If under revision, cite arXiv preprints.

---

## 10. Risk Mitigation for TMLR Review

| Risk | Mitigation built into paper design |
|---|---|
| "This is just a benchmark, not enough theory" | Theorems 1 & 2 + complexity hierarchy provide standalone theoretical value |
| "Environments are too simple" | Include Tier 3 (LLM, high-dim robotics) even if partially; show scalability |
| "Onset definition is arbitrary" | Robustness ablation + human validation + formal justification via CPD theory |
| "Limited novelty in baselines" | The novelty is the comparison framework, not the individual methods |
| "Should be at NeurIPS D&B" | Cover letter explicitly positions as formalization paper; theory first, benchmark second |
| "Self-serving (Flight Recorder advantage)" | Include cases where FR fails; be ruthlessly honest about limitations |

---

## 11. Timeline to Submission

| Month | Milestone |
|---|---|
| **Month 1** | Implement Tier 1 environments + onset labeling + detector interface + 3 baselines |
| **Month 2** | Implement Tier 2 environments + all baselines + initial metrics run |
| **Month 3** | Prove Theorems 1 & 2 rigorously; write Sections 1–3 |
| **Month 4** | Run all experiments + ablations; write Sections 4–7 |
| **Month 5** | Internal review: run `review_checklist.md`; circulate to 2 trusted reviewers |
| **Month 6** | Revise based on feedback; submit to TMLR |

**Earliest viable submission:** ~6 months from start of implementation.

**Minimum viable version (if accelerating):** Tiers 1+2 only, Theorem 1 only (no Theorem 2), 4 baselines. This compresses to ~4 months but weakens the paper.

---

## 12. Strategic Value Assessment

| Criterion | Score (1–10) | Justification |
|---|---|---|
| **Novelty** | 9 | First formalization + first benchmark for temporal onset detection |
| **Mathematical rigor** | 8 | Two theorems with proofs; formal problem setup; connection to CPD theory |
| **Experimental rigor** | 9 | 50 seeds, statistical testing, comprehensive ablations, multiple access levels |
| **Long-term citation potential** | 9.5 | Every future detection paper cites the benchmark they evaluate on |
| **Open-source impact** | 10 | pip-installable, MIT-licensed, leaderboard, community governance |
| **Benchmark potential** | 10 | This IS the benchmark |
| **Difficulty** | 8 | Theory: moderate-hard; Engineering: high; Coordination: high |
| **Estimated completion time** | 6 months (4 accelerated) | Substantial but achievable solo |
| **Strategic value** | 10 | Defines the evaluation methodology for the entire subfield |

---

## 13. Action Items

### Immediate (This Week)

1. **Lock the onset definition.** Write a 2-page formal note with the precise mathematical definition, edge cases, and connection to CPD theory. This is the foundation everything else rests on.

2. **Implement `base.py` detector interface.** Get the API right *first*. Everything downstream depends on this contract.

3. **Sketch Theorem 1 proof.** The impossibility result is the paper's crown jewel. Determine if the Le Cam construction works cleanly or needs modification.

### Short-Term (Weeks 2–4)

4. **Build Navigation-Proxy and GridWorld-Wireheading** with programmatic onset labels.
5. **Verify Flight Recorder produces non-trivial AUROC** on these environments.
6. **Implement CUSUM and Random baselines** — confirm metrics are discriminative.

### Medium-Term (Months 2–3)

7. **Complete Tier 1 + begin Tier 2.**
8. **Write Introduction and Problem Formalization** — these should be near-final quality before experiments complete.
9. **Complete proof of Theorem 1; attempt Theorem 2.**

### Decision Gates

- **After Tier 1 experiments:** Are the metrics discriminative (methods rank differently)? If not, redesign metrics.
- **After Theorem 1 proof attempt:** Is the result tight and clean? If not, weaken to a sufficient condition.
- **After 3 months:** Is Tier 3 necessary for TMLR acceptance or is Tiers 1+2 + theory sufficient? (OPINION: Tiers 1+2 + strong theory is sufficient for TMLR if the paper is framed as formalization-first.)

---

*Classification: RECOMMENDATION grounded in FACT (Aarav's current venue portfolio, TMLR review standards, benchmark literature patterns) and INFERENCE (reviewer behavior predictions based on Area Chair experience patterns).*

*This blueprint is a living document. Update after each decision gate.*


