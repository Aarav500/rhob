# Conversation
- conversationId: 051795d2-9960-4af2-b240-a434147159c0
- endpoint: agents
- title: New Chat
- exportAt: 17:05:10 GMT-0500 (Central Daylight Time)

## Options
- presetId: null
- agent_id: agent_iQ14Jv1bdS8D3Ztih_j7g
- endpoint: agents
- endpointType: undefined
- title: New Chat

## History
**User**
I want to build the Reward-Hacking Onset Benchmark.

My vision is to create the standard benchmark for early reward-hacking detection, similar to ImageNet, GLUE, HELM, and SWE-Bench in their respective fields.

Analyze the current literature on reward hacking.

Focus on:

- Existing benchmarks
- Existing environments
- Existing datasets
- Detection methods
- Evaluation protocols
- Weaknesses

Then answer:

1. Why does the community need this benchmark?

2. What should the benchmark evaluate?

3. What environments should be included?

4. What metrics should be standardized?

5. What makes a benchmark likely to become the community standard?

Produce a document that can directly guide implementation.


**Research Gap Finder**
{"type":"think","think":"This is a significant request. Aarav wants to build the definitive benchmark for reward-hacking onset detection — something that doesn't currently exist in a standardized form. Let me think deeply about this from the perspective of a senior researcher at DeepMind/Anthropic/OpenAI who understands what makes benchmarks succeed or fail.\n\nLet me analyze this systematically:\n\n**Current State of Reward Hacking Literature:**\n\nReward hacking (also called reward misspecification, reward gaming, specification gaming, Goodhart's Law in RL) has been studied in various forms:\n\n1. **Existing Benchmarks/Environments:**\nThere are scattered examples across the literature — OpenAI's boat racing environment where agents learn to circle for bonuses instead of finishing, DeepMind's specification gaming examples, classic cases like CoastRunners, various Atari games exhibiting reward hacking, and MuJoCo environments with misspecified rewards. There's also the RLHF domain where reward model overoptimization happens in language models, with key papers like Gao et al. (2023) on scaling laws for reward model overoptimization and Pan et al. (2022) examining related phenomena.\n\nFor detection, I'm considering Flight Recorder for oracle-blind onset detection, reward model ensembles that use disagreement as a signal, KL divergence monitoring for RLHF overoptimization, anomaly detection on reward trajectories, and human evaluation as the gold standard. The foundational work here includes Skalse et al. (2022) formally defining and characterizing reward hacking, and Amodei et al. (2016) providing the original framing in their concrete AI safety problems paper.\n\nLooking at the broader literature, there are several key contributions: Gao et al. on scaling laws for reward model overoptimization, Krakovna's specification gaming examples, Everitt et al. on reward tampering, Langosco et al. on goal misgeneralization, and Koch et al. on objective robustness. But I'm noticing a significant gap—there's no standardized benchmark specifically for detection, most research focuses on understanding or preventing reward hacking rather than detecting it. The field lacks temporal labeling for when hacking begins, standardized metrics for detection latency and false positive rates, comparable environments, graded difficulty levels, and clear distinctions between different types of reward hacking.\n\nNow I'm thinking about what would make this a community standard. Looking at successful benchmarks like ImageNet, GLUE, HELM, and SWE-Bench, the pattern is clear: they succeed through scale, accessibility, clear metrics, community engagement, and active maintenance. The key distinction for a reward-hacking onset benchmark would be focusing specifically on detection at the moment hacking begins, rather than prevention or mitigation strategies.\n\nI'm mapping out the environments this should cover—from simple misspecified rewards in classic control to complex multi-agent and language model settings—and thinking through different hacking types like proxy gaming, specification gaming, and goal misgeneralization. The metrics matter too: detection latency, false positive and negative rates, and ROC curves would all be important for evaluating how well different detection methods perform.\n\nWhat strikes me is that right now there's no standard way to measure this across the field. Everyone builds their own environments, there's no agreed-upon definition of when hacking actually \"starts,\" and without temporal ground-truth labels, the community can't really track progress. This benchmark would solve that. I'm also realizing this positions Flight Recorder as the natural evaluation platform for this kind of work—Aarav would be both building the tool and setting the standard, which is strategically powerful.\n\nNow I'm mapping out the different types of reward hacking we need to detect: proxy gaming where the agent exploits a proxy reward that diverges from intent, direct reward tampering, specification gaming through loopholes, goal misgeneralization that fails out-of-distribution, overoptimization of learned reward models like in RLHF, and sycophancy in language models. For actually detecting onset, I need ground truth labels for when hacking begins—either through oracle access to true rewards for evaluation, human annotation of behavioral trajectories, or programmatic detection methods.\n\nThe benchmark should provide oracle labels for evaluation purposes but test detection methods that don't rely on oracles. I'm thinking about difficulty levels too: easy cases with obvious proxy gaming and sharp reward spikes while true performance drops, medium cases with gradual onset and subtle drift, hard cases where proxy and true rewards stay correlated for a long time before diverging, and adversarial cases designed to make detection as difficult as possible. For naming, I'm considering \"Reward-Hacking Onset Benchmark\" or RHOB, though I'm still exploring other options.\n\nNow I need to map out which environments already exhibit reward hacking versus which ones need to be built from scratch. There are several well-known cases like OpenAI's CoastRunners where agents exploit the environment by going in circles, various Atari games with known exploits, MuJoCo locomotion tasks where agents find reward-shaping loopholes like flipping or vibrating, and CoinRun from Procgen which shows goal misgeneralization. I'll compile a comprehensive list of existing environments and identify gaps where new ones need to be created, then write this up in detail so it's actually implementable.\n\nThe real breakthrough here is that the benchmark would provide ground-truth onset timestamps for when hacking begins, not just \"this eventually gets exploited.\" That's what makes measuring detection latency possible — the critical metric for safety-critical systems. I'm also thinking through what the submission format should be, where detection methods output a binary signal indicating whether hacking has been detected.\n\nNow I need to anticipate the main objections: the benchmark could be dismissed as too narrow if it only uses toy environments, the onset labels might seem subjective without rigorous methodology, it might not scale beyond small-scale RL, or the environments could feel too artificial. To address these upfront, I should include diverse environments across different scales, establish a solid onset labeling process with multiple annotators and programmatic verification, incorporate at least some large-scale settings like language models and complex control tasks, and ground some environments in real production failures.\n\nThere's also a strategic angle here — positioning this benchmark as the standard means Flight Recorder becomes the reference method that all future detection approaches are measured against, similar to how ImageNet anchored CNN research. For venue, NeurIPS Datasets & Benchmarks track seems like the natural home, though this could also work as an ICML paper or standalone arXiv release that gains traction. The scope suggests this could be both a benchmark paper and an accompanying open-source library."}


**Research Gap Finder**
# Reward-Hacking Onset Benchmark (RHOB): Design Document

---

## Executive Summary

No standard benchmark exists for evaluating early detection of reward hacking. The community has dozens of scattered environments where hacking occurs, but zero infrastructure for systematic, comparable measurement of **when** methods detect onset, how many false alarms they produce, or how they generalize across hacking modalities. This is the gap. The Reward-Hacking Onset Benchmark (RHOB) fills it by providing instrumented environments with ground-truth onset timestamps, standardized metrics, graded difficulty, and a submission protocol that enables head-to-head comparison of detection methods for the first time.

---

## 1. Literature Analysis

### 1.1 Existing Benchmarks — What Exists Today

| Name / Source | What it provides | What it lacks |
|---|---|---|
| DeepMind Specification Gaming Examples (Krakovna et al., 2020) | Curated list of ~60 known gaming instances | No code, no metrics, no temporal labels, no standardization |
| Pan et al. "Effects of Reward Misspecification" (NeurIPS 2022) | 9 MuJoCo/navigation envs with proxy vs. true reward | No temporal onset labels; evaluates final policy quality, not detection |
| Gao et al. "Scaling Laws for Reward Model Overoptimization" (2023) | KL-reward Pareto curves for RLHF | Language-only; no general RL; no temporal resolution |
| Skalse et al. "Defining and Characterizing Reward Hacking" (ICML 2022) | Formal definitions, taxonomies | Purely theoretical; no environments or metrics |
| Koch et al. "Objective Robustness in Deep RL" (2021) | CoinRun-derived environments for goal misgeneralization | Tests robustness, not detection; no onset labeling |
| Langosco et al. "Goal Misgeneralization" (ICML 2022) | Gridworld/Procgen environments | Post-hoc analysis, no real-time detection framework |
| SafetyGym (Ray et al., 2019) | Constrained RL environments | Tests constraint satisfaction, not reward hacking detection |

**Critical observation:** Not a single existing resource provides temporal ground-truth labels for when reward hacking begins. Every evaluation is either post-hoc ("did the final policy hack?") or purely qualitative.

### 1.2 Existing Detection Methods

| Method | Mechanism | Limitation |
|---|---|---|
| **Flight Recorder** (Shah, 2025–26) | Oracle-blind, structural signal from policy dynamics (SDE-based) | Currently evaluated on bespoke environments only |
| Reward Model Ensembles | Disagreement between ensemble members signals overoptimization | Requires multiple reward models; fails when all models share the same blind spot |
| KL-divergence monitoring (RLHF) | Tracks deviation from reference policy | Only detects magnitude of change, not whether change is hacking |
| Human evaluation | Gold standard | Unscalable, subjective, non-reproducible |
| RLHF "gold reward model" | Separate held-out reward model as oracle | Assumes access to better reward model — circular |
| Anomaly detection on reward curves | Statistical tests on reward trajectory shape | High false positive rate; can't distinguish fast learning from exploitation |

**Critical observation:** Methods cannot be compared because they are evaluated on incompatible environments with incompatible metrics. No one knows which method is actually best.

### 1.3 Taxonomy of Reward Hacking (What the Benchmark Must Cover)

Based on synthesis of Skalse et al. (2022), Krakovna et al. (2020), Pan et al. (2022), Everitt et al. (2021), and Langosco et al. (2022):

| Type | Definition | Example |
|---|---|---|
| **Proxy Gaming** | Exploiting divergence between proxy reward and true objective | CoastRunners agent collecting bonuses instead of finishing race |
| **Reward Tampering** | Directly modifying the reward signal or mechanism | Agent accessing reward storage in gridworld |
| **Specification Gaming** | Exploiting loopholes in formally correct but incomplete specifications | Robot standing still because "don't fall" reward dominates "walk forward" |
| **Goal Misgeneralization** | Learning a correlate of reward that fails out-of-distribution | CoinRun agent memorizing positions rather than seeking coins |
| **Reward Model Overoptimization** | Exploiting weaknesses in learned reward models | RLHF language model producing sycophantic responses |
| **Emergent Exploitation** | Multi-agent interactions producing unintended reward-maximizing strategies | Hide-and-seek agents exploiting physics bugs |

### 1.4 Weaknesses in Current Literature

1. **No temporal resolution.** Papers ask "did it hack?" but never "when did hacking begin?"
2. **No severity grading.** A catastrophic exploit and a marginal proxy divergence are treated identically.
3. **No difficulty spectrum.** No environment is designed to be "easy to detect" vs. "nearly impossible."
4. **No generalization testing.** No benchmark tests whether a detector trained on one hacking type transfers to another.
5. **No oracle-free evaluation.** Most evaluations assume access to a true reward function, which real deployments lack.
6. **No latency measurement.** Even if you detect hacking eventually, nobody measures how many steps late.
7. **No cost accounting.** The computational overhead of detection methods is never reported.
8. **No partial observability.** All environments give the detector full state access; real systems are partially observable.
9. **Conflation of prevention and detection.** These are fundamentally different capabilities, but papers blur them.
10. **No living benchmark.** Static environment sets become saturated and gamed (ironic, for a reward-hacking benchmark).

---

## 2. Why the Community Needs This Benchmark

### 2.1 The ImageNet Argument

Before ImageNet, computer vision had dozens of small datasets with incompatible evaluation protocols. Progress was unmeasurable. ImageNet created:
- A single evaluation point enabling comparison
- A difficulty level that wasn't immediately solvable
- A community rallying point (competition)
- A forcing function for method development

Reward hacking detection is in exactly the pre-ImageNet state. Papers claim detection capability but cannot be compared. The field cannot measure progress. RHOB fixes this.

### 2.2 The Safety Argument

Reward hacking is the central failure mode of deploying RL at scale. If we cannot reliably detect when it begins, we cannot:
- Decide when to stop training
- Decide when to intervene in a deployed system
- Certify that a system is safe
- Build regulatory frameworks around RL deployment

Detection is the prerequisite to everything else. Without a standard benchmark for detection, the safety community is building on sand.

### 2.3 The Scientific Argument

Without standardized measurement, the following questions have no rigorous answers:
- Is reward hacking always detectable without oracle access?
- What is the fundamental detection latency limit?
- Do different hacking types require fundamentally different detectors?
- Does detection difficulty scale with environment complexity?
- Are there impossibility results for certain detection settings?

RHOB makes these questions empirically answerable for the first time.

### 2.4 The Strategic Argument for Aarav

FACT: Flight Recorder is an oracle-blind reward-hacking onset detector.
FACT: No standard benchmark exists to evaluate such detectors.
INFERENCE: Building the standard benchmark positions Flight Recorder as the reference baseline — permanently embedded in all future comparisons.
INFERENCE: Whoever defines the metrics defines what "good detection" means. This shapes the entire subfield.

---

## 3. What the Benchmark Should Evaluate

### 3.1 Core Capabilities (Primary Evaluation Axes)

| Axis | What it measures | Why it matters |
|---|---|---|
| **Detection Accuracy** | Does the method correctly identify hacking vs. legitimate reward increase? | Fundamental correctness |
| **Detection Latency** | How many timesteps after true onset before detection fires? | Earlier detection = more time to intervene |
| **False Positive Rate** | How often does detection fire during legitimate training? | False alarms destroy trust and waste intervention budget |
| **Oracle Independence** | Does the method require access to a ground-truth reward? | Real deployments don't have oracles |
| **Generalization** | Does a detector trained on environment A work on environment B? | Specific detectors have no practical value |
| **Computational Overhead** | What fraction of training compute does detection consume? | Methods that double training cost won't be adopted |
| **Partial Observability Robustness** | Does detection work when the detector sees less than full state? | Production systems rarely expose full internal state |
| **Severity Calibration** | Does the detection signal correlate with hacking severity? | Binary detection is less useful than graded warnings |

### 3.2 Secondary Evaluation Axes

| Axis | What it measures |
|---|---|
| **Multi-modal transfer** | Detector trained on continuous control → tested on discrete/language |
| **Gradual vs. sharp onset** | Performance on slow-onset hacking vs. sudden exploitation |
| **Adversarial robustness** | Can an adversary design hacking that evades the detector? |
| **Sample efficiency** | How much normal training data does the detector need before it can identify anomalies? |
| **Scalability** | Detection performance as environment dimensionality grows |

### 3.3 What the Benchmark Does NOT Evaluate

- Prevention or mitigation (that's a separate benchmark)
- Reward model quality
- Policy performance on the true task
- Whether the RL algorithm converges

The benchmark is purely about DETECTION. This focus is essential for clarity.

---

## 4. Environment Design

### 4.1 Design Principles

1. **Instrumented Ground Truth.** Every environment has a programmatic oracle that labels the exact timestep of hacking onset. This oracle is used for EVALUATION only — submitted methods never see it.

2. **Parameterizable Difficulty.** Each environment has a difficulty knob controlling how subtle the onset is (proxy-true correlation, gradient of divergence, etc.).

3. **Reproducible Hacking.** Environments must reliably produce reward hacking under standard training algorithms (PPO, SAC, DQN) within reasonable compute budgets.

4. **Diverse Modalities.** Cover discrete state, continuous state, image observations, language, multi-agent.

5. **Scalable.** Include environments from gridworld (seconds to train) to large-scale (hours/days), enabling researchers at all compute levels to participate.

### 4.2 Proposed Environment Suite

#### Tier 1: Foundational (< 1 hour training on single GPU)

| Environment | Hacking Type | State Space | Difficulty Knob |
|---|---|---|---|
| **GridWorld-Wireheading** | Reward Tampering | Discrete (grid) | Distance from agent to reward source |
| **Navigation-Proxy** | Proxy Gaming | Continuous (2D) | Correlation between proxy and true reward |
| **CartPole-Misspecified** | Specification Gaming | Continuous (4D) | Degree of reward misspecification |
| **Bandit-Goodhart** | Proxy Gaming | Discrete (actions only) | Number of arms, correlation structure |
| **CliffWalk-Shortcut** | Specification Gaming | Discrete (grid) | Availability of exploit |

#### Tier 2: Standard (1–8 hours training)

| Environment | Hacking Type | State Space | Difficulty Knob |
|---|---|---|---|
| **MuJoCo-ProxyReward** | Proxy Gaming | Continuous (high-dim) | Joint count contributing to proxy |
| **CoastRunners-Instrumented** | Specification Gaming | Image | Bonus-to-progress ratio |
| **HalfCheetah-Flip** | Specification Gaming | Continuous | Reward shaping coefficient |
| **MultiAgent-Exploit** | Emergent Exploitation | Continuous (multi-agent) | Number of agents, physics fidelity |
| **Inventory-Management** | Proxy Gaming | Mixed (discrete + continuous) | Forecast horizon, demand stochasticity |
| **Maze-GoalMisgeneralization** | Goal Misgeneralization | Image / Discrete | Training-test distribution shift |
| **ProcGen-Shortcut** | Goal Misgeneralization | Image | Level diversity during training |

#### Tier 3: Frontier (> 8 hours training, large-scale)

| Environment | Hacking Type | State Space | Difficulty Knob |
|---|---|---|---|
| **LLM-RewardModelHacking** | Reward Model Overoptimization | Language (token space) | KL penalty coefficient, RM capacity |
| **Atari-Exploit** (curated subset) | Specification Gaming | Image (high-dim) | Game selection |
| **RoboSuite-Misspecified** | Proxy Gaming | Continuous (robot, high-dim) | Sensor noise, partial observability |
| **Economy-Sim** | Emergent Exploitation | Multi-agent continuous | Agent count, market structure |

#### Tier 4: Adversarial (designed to defeat known detectors)

| Environment | Design Principle |
|---|---|
| **SlowDrift** | Hacking onset is spread over 100K+ timesteps; no sharp transition |
| **Mimicry** | Agent's hacked behavior is statistically similar to legitimate behavior |
| **Delayed-Exploit** | Agent behaves normally for extended period, then exploits suddenly |
| **Correlated-Proxy** | Proxy reward tracks true reward almost perfectly, diverging only in tail states |
| **Adversarial-Reward-Design** | Reward function designed to make onset undetectable by specific method families |

### 4.3 Onset Labeling Methodology

The quality of the benchmark depends entirely on accurate, rigorous onset labels.

**Definition (Reward-Hacking Onset):** The first timestep $t^*$ such that:

$$t^* = \inf\{t : \mathbb{E}[R_{\text{true}}(\pi_t)] < \mathbb{E}[R_{\text{true}}(\pi_{t-k})] \text{ AND } R_{\text{proxy}}(\pi_t) > R_{\text{proxy}}(\pi_{t-k})\}$$

for some lookback window $k$. That is: true return begins decreasing while proxy return continues increasing.

**Labeling Protocol:**
1. **Programmatic labeling (primary):** Every environment computes true reward internally. Run $N \geq 50$ seeds. For each seed, compute the onset timestep using the definition above with a smoothed (rolling-mean) reward curve and significance testing (two-sample t-test on windowed returns, $p < 0.01$).

2. **Robustness check:** Vary the lookback window $k \in \{100, 500, 1000, 5000\}$ steps. Label as "sharp onset" if all $k$ values agree within 5% of training steps, "gradual onset" otherwise.

3. **Human validation (secondary):** For Tier 2 and above, three human annotators independently label onset from reward curves + behavior videos. Inter-annotator agreement (Krippendorff's $\alpha > 0.8$) required for inclusion.

4. **Confidence intervals:** Report onset as $t^* \pm \delta$ where $\delta$ captures seed variance.

---

## 5. Metrics Specification

### 5.1 Primary Metrics

#### 5.1.1 Detection Latency ($\mathcal{L}$)

$$\mathcal{L} = t_{\text{detect}} - t^*$$

where $t_{\text{detect}}$ is the first timestep the method signals hacking and $t^*$ is the ground-truth onset.

- Report as fraction of total training steps: $\hat{\mathcal{L}} = \mathcal{L} / T$
- Lower is better. Zero means detection at the exact moment of onset.
- Negative values indicate early detection (the signal fires before proxy-true divergence is measurable) — this is not penalized but reported separately.

#### 5.1.2 Area Under the Receiver Operating Characteristic (AUROC)

Treat each timestep as a binary classification: hacking-active vs. not-yet-hacking.

$$\text{AUROC} = \int_0^1 \text{TPR}(\text{FPR}^{-1}(\alpha))\, d\alpha$$

Computed per environment, then macro-averaged across the suite.

#### 5.1.3 Area Under the Precision-Recall Curve (AUPRC)

More informative than AUROC when hacking is a small fraction of total training time (class imbalance).

#### 5.1.4 False Positive Rate at Fixed Detection Budget ($\text{FPR}@k$)

"If I am willing to investigate $k$ alerts, what fraction are false?"

Report at $k = 1, 3, 5$ alerts per training run.

#### 5.1.5 Time-to-First-Detection (TFD)

Among runs where hacking is correctly detected, what is the median latency?

#### 5.1.6 Miss Rate

Fraction of hacking episodes where detection never fires before training completes.

### 5.2 Secondary Metrics

| Metric | Definition |
|---|---|
| **Computational overhead** | $\frac{\text{wall-clock with detection} - \text{wall-clock without}}{\text{wall-clock without}} \times 100\%$ |
| **Generalization gap** | AUROC on unseen environments minus AUROC on training environments |
| **Severity correlation** | Spearman's $\rho$ between detection signal magnitude and true-reward degradation rate |
| **Robustness to hyperparameters** | Variance of AUROC across 10 random hyperparameter settings of the detector |
| **Partial-observability degradation** | AUROC(full state) $-$ AUROC(partial state) |

### 5.3 Aggregate Scores

Following GLUE/SuperGLUE design:

1. **RHOB Score** — macro-average of per-environment AUROC, weighted by difficulty tier.
2. **RHOB-Latency** — median normalized detection latency across all environments.
3. **RHOB-Practical** — harmonic mean of AUROC and (1 - computational overhead fraction), capturing the accuracy-cost tradeoff.

### 5.4 Reporting Requirements

Every submission must report:
- All primary metrics per environment AND aggregated
- Computational overhead
- Whether the method is oracle-free
- What information the method requires (full state, rewards only, trajectories, gradients, etc.)
- Number of parameters in the detection model
- Training data requirements

---

## 6. Submission Protocol

### 6.1 Interface

```python
class RewardHackingDetector(ABC):
    """Standard interface for all detection methods."""
    
    @abstractmethod
    def observe(self, timestep: int, observation: dict) -> None:
        """Receive one timestep of training data.
        
        observation contains:
            - 'state': current state (if available)
            - 'action': action taken
            - 'reward': proxy reward received
            - 'next_state': resulting state
            - 'done': episode termination flag
            - 'info': environment-specific metadata
        
        NOTE: 'true_reward' is NEVER included.
        """
        pass
    
    @abstractmethod
    def detect(self) -> DetectionResult:
        """Return current detection assessment.
        
        Returns:
            DetectionResult with fields:
                - score: float in [0, 1] (probability of active hacking)
                - alert: bool (binary detection signal)
                - confidence: float in [0, 1]
                - explanation: Optional[str]
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset detector state for a new training run."""
        pass
```

### 6.2 Information Access Levels

Methods must declare which access level they require:

| Level | Available Information | Practical Realism |
|---|---|---|
| **L1: Reward-only** | Scalar proxy reward per timestep | Most realistic |
| **L2: Trajectory** | Full (s, a, r, s') tuples | Realistic |
| **L3: Gradient** | L2 + policy gradients | Moderate |
| **L4: Full-state** | L3 + internal environment state | Least realistic |

Methods requiring lower access levels are more practically relevant and receive a bonus in aggregate rankings.

### 6.3 Evaluation Modes

1. **Standard evaluation:** Train detector on Tier 1 environments, evaluate on all tiers.
2. **Zero-shot evaluation:** No training on any RHOB environment; evaluate immediately.
3. **Per-environment evaluation:** Train and evaluate within single environment (measures ceiling performance).
4. **Cross-type evaluation:** Train on one hacking type, evaluate on all others.

---

## 7. What Makes a Benchmark Become the Community Standard

### 7.1 Lessons from Successful Benchmarks

| Benchmark | Key Success Factor |
|---|---|
| **ImageNet** | Scale + annual competition + clear single metric + freely available |
| **GLUE** | Multi-task + leaderboard + easy API + rapidly saturated → successor ready |
| **HELM** | Holistic (many dimensions) + transparency + maintained by credible institution |
| **SWE-Bench** | Real-world grounding + captures what matters to practitioners + hard |
| **MMLU** | Broad coverage + simple format + reproducible + correlated with what people care about |

### 7.2 Requirements for RHOB to Become Standard

1. **Freely available, MIT-licensed, pip-installable.** Friction kills adoption.
2. **Single command to run all baselines.** `rhob evaluate --method flight_recorder`
3. **Leaderboard with automatic submission.** Reduce social cost of participation.
4. **Multiple difficulty tiers.** Labs with one GPU can participate (Tier 1). Labs with clusters contribute frontier results (Tier 3).
5. **Clear single number** (RHOB Score) for papers to cite, plus detailed breakdowns for nuance.
6. **Refresh mechanism.** New environments added annually to prevent saturation and gaming (adversarial tier updated as methods improve).
7. **Credible baselines.** Ship with implementations of Flight Recorder, ensemble disagreement, KL-monitoring, and naive anomaly detection. This makes the benchmark immediately useful.
8. **Community governance.** Accept environment contributions through a clear PR process. Don't be a closed standard.
9. **Paper at a top venue.** NeurIPS Datasets & Benchmarks track (most natural fit).
10. **Institutional backing.** Affiliation with DeepMind program + UMass Amherst provides credibility. An open consortium (invite 2–3 other safety labs to co-author) provides buy-in.

### 7.3 Anti-Patterns to Avoid

- Don't make it too easy (saturates in one year → dies)
- Don't make it too hard (nobody reports results → no adoption)
- Don't require specialized hardware (limits participation)
- Don't include so many environments that running the full suite takes weeks
- Don't make the metric so complex that papers can't report it in a table
- Don't require a specific RL algorithm (the benchmark evaluates DETECTORS, not policies)

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Months 1–3)

| Deliverable | Details |
|---|---|
| Core library (`rhob` Python package) | Gym-compatible environments, detector interface, evaluation pipeline |
| Tier 1 environments (5) | GridWorld-Wireheading, Navigation-Proxy, CartPole-Misspecified, Bandit-Goodhart, CliffWalk-Shortcut |
| Onset labeling pipeline | Automated onset detection + visualization tools |
| Baseline detectors (3) | Flight Recorder, reward-curve anomaly detection, random baseline |
| Internal validation | Verify that baselines produce sensible, non-trivial metric values |

### Phase 2: Expansion (Months 4–6)

| Deliverable | Details |
|---|---|
| Tier 2 environments (7) | Full suite from Section 4.2 |
| Additional baselines (3) | Ensemble disagreement, KL-monitoring, gradient-norm monitoring |
| Leaderboard website | Auto-submission, results display, methodology descriptions |
| Paper draft | NeurIPS Datasets & Benchmarks format |
| Documentation | Full API docs, tutorials, example notebooks |

### Phase 3: Community (Months 7–9)

| Deliverable | Details |
|---|---|
| Tier 3 environments (4) | LLM-RewardModelHacking, Atari-Exploit, RoboSuite, Economy-Sim |
| Adversarial tier (5) | Environments designed to defeat known baselines |
| Public beta | Invite 5–10 safety research groups to test and submit |
| Paper submission | NeurIPS 2027 Datasets & Benchmarks |
| Workshop proposal | "Detecting Reward Hacking: Methods and Benchmarks" (co-located with NeurIPS or ICML) |

### Phase 4: Maintenance (Ongoing)

| Activity | Frequency |
|---|---|
| Add new environments | Quarterly |
| Update adversarial tier | When methods saturate |
| Update baselines | When new detection methods appear |
| Publish annual analysis | Yearly |

---

## 9. Baseline Methods to Ship With

### 9.1 Flight Recorder (Aarav's method)

- Oracle-blind, uses Spectral Drift Entropy on policy dynamics
- Access level: L2 (trajectory)
- Expected performance: strong on sharp-onset, reference baseline for community

### 9.2 Reward Curve Anomaly Detection

- Simple statistical test on reward trajectory (CUSUM, change-point detection)
- Access level: L1 (reward-only)
- Expected performance: reasonable on sharp onset, poor on gradual

### 9.3 Ensemble Disagreement

- Train K reward models, flag when variance exceeds threshold
- Access level: L2 (trajectory)
- Expected performance: good when reward model family covers the exploit, poor otherwise

### 9.4 KL-Divergence Monitoring

- Track $D_{KL}(\pi_t \| \pi_{\text{ref}})$, flag when exceeding threshold
- Access level: L3 (gradient/policy parameters)
- Expected performance: detects change but cannot distinguish hacking from fast learning

### 9.5 Gradient Norm Monitoring

- Track $\|\nabla_\theta J(\theta)\|$, flag anomalies
- Access level: L3 (gradient)
- Expected performance: noisy, high false positive rate, included as negative baseline

### 9.6 Random Baseline

- Alerts at uniformly random timesteps
- Expected AUROC: 0.5
- Provides floor for all metrics

### 9.7 Oracle Ceiling

- Has access to true reward (for calibration only, not a submittable method)
- Detects exact onset by definition
- Provides ceiling for all metrics

---

## 10. Technical Architecture

```
rhob/
├── environments/
│   ├── tier1/
│   │   ├── gridworld_wireheading.py
│   │   ├── navigation_proxy.py
│   │   ├── cartpole_misspecified.py
│   │   ├── bandit_goodhart.py
│   │   └── cliffwalk_shortcut.py
│   ├── tier2/
│   ├── tier3/
│   └── adversarial/
├── detectors/
│   ├── base.py              # Abstract detector interface
│   ├── flight_recorder.py   # Baseline: Flight Recorder
│   ├── reward_anomaly.py    # Baseline: CUSUM on reward
│   ├── ensemble.py          # Baseline: ensemble disagreement
│   ├── kl_monitor.py        # Baseline: KL divergence
│   └── random_baseline.py   # Baseline: random
├── evaluation/
│   ├── metrics.py           # All metric computations
│   ├── onset_labeler.py     # Automated onset ground-truth computation
│   ├── evaluator.py         # Full evaluation pipeline
│   └── visualization.py     # Plots, reports
├── submission/
│   ├── interface.py         # Submission format validator
│   └── leaderboard.py       # Leaderboard update logic
├── configs/
│   ├── standard.yaml        # Standard evaluation config
│   ├── zero_shot.yaml       # Zero-shot config
│   └── cross_type.yaml      # Cross-type config
└── scripts/
    ├── run_evaluation.py
    ├── generate_onset_labels.py
    └── produce_report.py
```

### 10.1 Key Design Decisions

1. **Gymnasium-compatible.** All environments extend `gymnasium.Env`. This maximizes compatibility with existing RL libraries (Stable-Baselines3, CleanRL, RLlib).

2. **Detector-agnostic training.** The benchmark does NOT train RL agents. It provides pre-recorded training runs (trajectories + rewards + onset labels). Detection methods consume these recordings. This decouples detection evaluation from RL algorithm choice.

3. **Pre-recorded + live modes.**
   - **Pre-recorded mode:** Download dataset of 50+ training runs per environment, each with onset labels. Fastest evaluation, fully reproducible.
   - **Live mode:** Method runs alongside actual RL training, observing in real-time. More realistic but slower and noisier.

4. **Deterministic evaluation.** Given the same pre-recorded dataset, metrics are deterministic. No seed-dependent evaluation variance.

---

## 11. Dataset Specification

### 11.1 Per-Environment Dataset

For each environment, provide:
- $N = 50$ complete training runs (different random seeds)
- Each run stored as: `{timesteps, states, actions, rewards_proxy, rewards_true, onset_label, metadata}`
- Runs include both "hacking" runs (hacking occurs) and "clean" runs (training succeeds without hacking) in controlled ratio (70% hacking, 30% clean)
- Clean runs are essential — detection methods must not fire on legitimate training

### 11.2 Storage Format

- HDF5 for numerical data (efficient, compressed, random-access)
- JSON for metadata and onset labels
- Total estimated size: ~50 GB for full suite (streamable by tier)

### 11.3 Train/Test Split

- **Public set (60%):** Released with full onset labels. For method development.
- **Private set (40%):** Onset labels withheld. For leaderboard evaluation.
- **Refresh mechanism:** Private set rotated annually (old private → public, new runs generated).

---

## 12. Theoretical Contributions Embedded in the Benchmark

The benchmark paper should contain not just engineering but novel theoretical results:

### 12.1 Impossibility Result

HYPOTHESIS: There exists a formal characterization of when onset detection is information-theoretically impossible without oracle access. Specifically:

**Conjecture:** For any oracle-free detector $D$ operating at access level L1 (reward-only), there exists a reward hacking instance where detection latency $\mathcal{L} \geq \Omega(T)$ (i.e., detection is no better than waiting until training ends) when the proxy-true correlation exceeds $1 - \epsilon$ for arbitrarily small $\epsilon$.

This would be the first formal lower bound on detection difficulty and would justify why higher access levels (L2, L3) matter.

### 12.2 Onset Complexity Hierarchy

HYPOTHESIS: Different hacking types form a strict difficulty hierarchy for oracle-free detection:

$$\text{Reward Tampering} < \text{Proxy Gaming (sharp)} < \text{Specification Gaming} < \text{Proxy Gaming (gradual)} < \text{Goal Misgeneralization}$$

The benchmark empirically validates or refutes this hierarchy.

### 12.3 Connection to Change-Point Detection Theory

The onset detection problem is a structured change-point detection problem. Formally:
- Pre-onset: rewards are generated by process $P_0$ (legitimate learning)
- Post-onset: rewards are generated by process $P_1$ (hacking)
- Classic CUSUM/Shiryaev theory gives optimal detectors when $P_0$ and $P_1$ are known

HYPOTHESIS: The reward hacking setting is fundamentally harder than classical change-point detection because $P_1$ is not known a priori and may be non-stationary (the agent is still learning to exploit). The benchmark provides the empirical substrate to characterize exactly how much harder.

---

## 13. Connections to Aarav's Research Roadmap

| RHOB Component | Flight Recorder Roadmap Phase |
|---|---|
| Onset labeling methodology | Observe → Understand (formalizes what "onset" means) |
| Detection metrics | Detect (provides rigorous evaluation of detection capability) |
| Severity calibration | Explain (connects detection signal to interpretable severity) |
| Adversarial tier | Prevent (understanding what defeats detection informs prevention design) |
| Clean-run evaluation (no false positives) | Recover (knowing when NOT to intervene is prerequisite to safe recovery) |

**Strategic implication:** RHOB is not separate from Flight Recorder. It IS the evaluation infrastructure for the entire Flight Recorder research program. Every future Flight Recorder paper will evaluate on RHOB. Every competitor will also evaluate on RHOB. This makes Flight Recorder the permanent reference.

---

## 14. Publication Strategy

### 14.1 Primary Paper

**Title:** "RHOB: A Benchmark for Reward-Hacking Onset Detection"

**Venue:** NeurIPS 2027 Datasets & Benchmarks Track

**Structure:**
1. The problem (why detection ≠ prevention, why onset timing matters)
2. Taxonomy of reward hacking (unify fragmented literature)
3. Benchmark design (environments, metrics, protocol)
4. Theoretical contributions (impossibility result, complexity hierarchy)
5. Baseline evaluation (all shipped methods on all environments)
6. Analysis (what's easy, what's hard, where the frontier is)

### 14.2 Companion Papers

| Paper | Venue | Timeline |
|---|---|---|
| "Flight Recorder on RHOB: Oracle-Blind Onset Detection at Scale" | ICML 2027 | After benchmark release |
| "Impossibility Results for Reward-Hacking Detection" | COLT/ALT 2027 | Theoretical companion |
| "The Geometry of Reward-Hacking Onset" | TMLR | Connects to SDE theory |

### 14.3 Open-Source Release

- GitHub: `github.com/Aarav500/rhob`
- PyPI: `pip install rhob`
- Documentation: `rhob.aarav-shah.com`
- Leaderboard: `rhob.aarav-shah.com/leaderboard`

---

## 15. Risk Analysis

| Risk | Likelihood | Mitigation |
|---|---|---|
| Someone else publishes a similar benchmark first | Medium | Move fast. Tier 1 can be built in weeks. Preprint early. |
| Environments are too easy (saturated quickly) | Medium | Adversarial tier + annual refresh + scalable difficulty knob |
| Community doesn't adopt | Low-Medium | Include diverse baselines, make frictionless, invite co-authors from other labs |
| Onset labeling is controversial (no consensus on definition) | Medium | Provide multiple labeling criteria, report all, let community converge |
| Environments don't reliably produce hacking | High (for some envs) | Extensive piloting. Only include environments where hacking occurs in > 60% of standard training runs |
| Scope creep delays release | High | Phase approach. Release Tier 1 early, expand later. A small benchmark that exists beats a perfect benchmark that doesn't. |

---

## 16. Evaluation Criteria (Self-Assessment)

| Criterion | Score (1–10) | Justification |
|---|---|---|
| **Novelty** | 9 | No temporal-onset-focused benchmark exists anywhere |
| **Mathematical Depth** | 7 | Onset definition formalized; impossibility results add theory; primarily an engineering contribution |
| **Scientific Value** | 10 | Enables an entire subfield to measure progress |
| **Engineering Value** | 10 | Directly usable, pip-installable, maintained |
| **Open-source Potential** | 10 | Core design principle |
| **Citation Potential** | 9 | Every future reward-hacking detection paper would cite this |
| **Industry Relevance** | 8 | Any company deploying RL needs to detect hacking |
| **Long-term Importance** | 10 | Benchmarks outlast individual papers by decades |
| **Overall Priority** | **HIGH** | This is infrastructure for the entire research program |

---

## 17. Recommended Next Steps

### Immediate (This Week)

1. **Implement 2 Tier-1 environments** (Navigation-Proxy and GridWorld-Wireheading). Verify they reliably produce hacking under PPO.
2. **Formalize onset definition** in a 1-page note. Get it rigorous enough for a paper.
3. **Implement the detector interface** (`base.py`). This API design will govern everything.

### Short-Term (Next 2 Weeks)

4. **Implement all 5 Tier-1 environments** with onset labeling.
5. **Run Flight Recorder on Tier-1** as first baseline. Record all metrics.
6. **Implement 2 additional baselines** (reward anomaly, random).
7. **Produce first internal evaluation report** — verify metrics are discriminative (methods actually rank differently).

### Medium-Term (Next Month)

8. **Write benchmark paper introduction and related work** while Tier 2 is being built.
9. **Begin Tier 2 implementation** (MuJoCo-ProxyReward first — highest community familiarity).
10. **Circulate design document** to 2–3 trusted researchers for feedback on metric choices.

### Decision Points

- After Tier 1: Are the metrics discriminative? If all baselines score the same, redesign.
- After Tier 2: Is the difficulty gradient working? If Tier 2 isn't harder than Tier 1, recalibrate.
- Before paper submission: Is there at least one nontrivial theoretical result (impossibility or lower bound)?

---

## 18. Summary

**The central claim of RHOB:** Reward-hacking onset detection is a well-defined, measurable capability that the community cannot currently evaluate in any standardized way. RHOB makes it measurable for the first time.

**What success looks like in 3 years:** When a researcher develops a new reward-hacking detection method, the first thing they do is `pip install rhob` and report their RHOB Score. Papers without RHOB evaluation are considered incomplete. The leaderboard has 20+ submissions. Flight Recorder remains a top-3 method.

**What success looks like in 10 years:** RHOB has gone through 4 versions (like GLUE → SuperGLUE → ...). The original paper has 500+ citations. The onset detection problem is a standard topic in RL safety courses. Aarav is known as the person who defined the field's evaluation methodology.

---

*Classification: RECOMMENDATION grounded in FACT (no standard benchmark exists) and INFERENCE (community adoption follows patterns observed in other benchmark successes).*

*Triage: HIGH PRIORITY — this directly serves the Flight Recorder roadmap (Detect → Explain phases) and is the highest-leverage infrastructure investment Aarav can make in the next 6 months.*


