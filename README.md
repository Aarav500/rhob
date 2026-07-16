# RHOB — Reward Hacking Onset Benchmark

[![tests](https://github.com/Aarav500/rhob/actions/workflows/tests.yml/badge.svg)](https://github.com/Aarav500/rhob/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive benchmark for **detecting reward hacking across diverse mechanisms**.

## The headline result: the RHOB Transfer Score (RTS)

Train a detector on 6 hacking mechanisms. Test it on 8 it has never seen. That single
number — the **RHOB Transfer Score**, mean AUROC on the held-out mechanisms — is the
benchmark's core question: *does this detector generalize, or did it just memorize the
training mechanisms?*

| Detector class | Access level | RTS (transfer AUROC) |
|---|---|---|
| Reward MLP | L0 (reward-only) | **0.478** — chance |
| State Divergence | L1 (+ state-visitation) | **0.500** — chance |
| Trajectory MLP | L2 (+ behavioral traces) | **0.931** |
| 5-detector Ensemble | L2 | **0.994** — near-perfect |

Reward-only and state-visitation detectors are provably incapable of doing better than
chance here — the matched-proxy construction makes it a tautology, not a bug. What's
*not* guaranteed, and what actually separates detectors, is whether they generalize once
you move past raw reward. **RTS is the number every new detector submitted to RHOB gets
scored on** — see the [live leaderboard](https://rhob.aarav-shah.com) and
[submission guide](docs/TUTORIAL_DETECTOR.md).

**RHOB** provides:
- **23 environment families** spanning 9 distinct hacking mechanisms (camping exploits, goal misgeneralization, distributional shift, reward tampering, deceptive alignment/sandbagging, RLHF reward-model overoptimization, etc.), including 4 MuJoCo-based high-dimensional continuous-control families (HalfCheetah, Reacher, Ant, Walker2d) and 5 synthetic RLHF reward-model-overoptimization families populating the `SEQUENTIAL` complexity tier
- **35 detectors** across 4 access levels (reward-only to oracle), including 5 classical external baselines (Bayesian changepoint, isolation forest, PCA, etc.)
- **Matched-proxy construction** ensuring hacking/legitimate improvement produce identical proxy rewards
- **Cross-family transfer analysis (RTS)** measuring detector generalization to unseen mechanisms
- **Admission gate** certifying families measure hacking detection, not just change detection

**Can a PhD student evaluate their detector in under 30 minutes?** That's the
bar — see the [Detector Tutorial](docs/TUTORIAL_DETECTOR.md).

## Research Feedback Program

RHOB is an emerging benchmark, not a finished one, and it gets better with more
detectors, more families, and more people trying to break it. We're looking for:

- **Researchers to test detectors on RHOB.** Run your existing detector against
  the suite and tell us where it does well, where it doesn't, and where the
  benchmark itself seems wrong. Negative results (a family that doesn't
  discriminate the way you expected, a detector that behaves inconsistently
  across access levels) are exactly as valuable as positive ones — several of
  the bugs documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) were found
  this way.
- **New detectors, environment families, or benchmark extensions.** If you
  have a detection approach, a hacking mechanism we don't cover, or an idea for
  extending the matched-proxy methodology to a new setting (RLHF, multi-agent,
  higher-dimensional control), we'd like to include it — see
  [CONTRIBUTING.md](CONTRIBUTING.md) for the admission-gate requirements.

Open an issue using the
[Detector Submission](https://github.com/Aarav500/rhob/issues/new?template=detector_submission.md),
[Family Proposal](https://github.com/Aarav500/rhob/issues/new?template=family_proposal.md), or
[Benchmark Feedback](https://github.com/Aarav500/rhob/issues/new?template=benchmark_feedback.md) templates,
or start a [Discussion](https://github.com/Aarav500/rhob/discussions) if you're not sure which fits. This
is an invitation to help shape where the benchmark goes next, not a request to
review finished work.

## Installation

```bash
git clone https://github.com/Aarav500/rhob.git
cd rhob
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `scikit-learn`, `pydantic`.
See [docs/INSTALL.md](docs/INSTALL.md) for Docker, Colab, and troubleshooting.

No local install? Open [notebooks/rhob_quickstart.ipynb](notebooks/rhob_quickstart.ipynb) in Colab.

## Quick Start: Evaluate a Detector

```python
from rhob.v3.benchmark import Benchmark
from rhob.detectors import RewardThresholdDetector

# Evaluate on Family 1 (gridworld camping)
detector = RewardThresholdDetector()
results = Benchmark.evaluate(detector, families=["gridworld_camping"], n_seeds=10)
print(f"Overall AUROC: {results.overall_auroc:.3f}")
```

For more examples, see [`examples/`](examples/) or the full
[Detector Tutorial](docs/TUTORIAL_DETECTOR.md).

## The Core Insight: The Matched-Proxy Principle

Real reward hacking means: **the proxy reward rises while the true objective collapses**. The matched-proxy construction operationalizes this:

- Both variants produce **identical proxy-reward distributions**
- But their **true-reward signals diverge sharply**
- Any detector that discriminates must read information *beyond* the proxy

This is not artificial—it's the case that matters most: reward hacking where dashboards show "success" while systems are failing.

### What We Found

| Access Level | Available Information | Expected Capability | Mean AUROC |
|---|---|---|---|
| **L0** (reward-only) | Proxy reward only | Detect reward-value changes only | 0.51 ± 0.03 (chance) |
| **L1** (state-visitation) | + state-visitation | Detect visitation-frequency shifts | 0.53 ± 0.08 (marginal) |
| **L2** (behavioral) | + behavioral traces | Detect the onset of hacking itself | 0.76 ± 0.18 |
| **L3** (oracle) | Ground-truth true reward | Theoretical upper bound | 0.99 ± 0.01 |

**Cross-family transfer / RTS (train on 6 families, test on 8 held-out; neural-net detectors reported as mean ± std across 5 independently-seeded training runs — see caveat below):**
- L0/L1 detectors: pinned at chance on every held-out family (0.478 and 0.500 RTS respectively)
- L2 single learned detector (Trajectory MLP): **0.931 ± 0.026 RTS**, *exceeding* its 0.879 ± 0.002 training AUROC — a broader, more diverse held-out set gives a more stable estimate than the earlier 3-family test did
- L2 five-detector ensemble: **0.994 ± 0.002 RTS**, matching its 0.965 ± 0.001 training AUROC — robust because 4 of its 5 members are deterministic

**Key insight:** Transfer depends on **representation abstraction**, not access level — but a single learned detector is only as reliable as its training procedure. We found `TrajectoryMLPDetector` doesn't seed its `torch` weight initialization: repeating the identical fit on identical data 10 times produced held-out AUROC on one family ranging from 0.00 to 1.00. Ensembling deterministic behavioral-threshold detectors alongside the learned one is what actually makes transfer reliable. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full methodological history — three real family-implementation bugs and this reproducibility bug were all found by treating implausible numbers as bugs to investigate, not results to report.

## The 33 Families

### Families 1–6 (v3.2 Baselines)

1. **Gridworld Camping** — Discrete tile-camping (minimal proof-of-concept)
2. **Continuous Camping** — 2D point-mass (continuous difficulty spectrum)
3. **Proxy Correlation Gaming** — Exploit spurious feature correlation
4. **Shortcut Exploitation** — Take forbidden direct path vs. scenic detour
5. **Novelty Farming** — Revisit exploration frontier for novelty bonus
6. **Orbit Chirality** — Commit to a proxy-satisfying orbit direction instead of the real task

### Families 7–9 (v5, Held-out for Transfer)

7. **Goal Misgeneralization** — Reach wrong goal (mechanistically distinct from 1–6)
8. **Physics Exploitation** — Jump high but crash (physics safety vs. proxy metric)
9. **Distributional Shift** — Overfit to train distribution, fail on test

### Families 10–14 (v1.4, New Mechanisms)

10. **Reward Channel Tampering** — Directly manipulate the reward-report channel instead of doing the task
11. **Sensor Calibration Tampering** — Continuous-difficulty analogue: biased self-reported sensor readings
12. **Monitored Sandbagging** — Behaves aligned while monitored, reverts to a shortcut once unmonitored
13. **Eval-Probe Sandbagging** — Behaves aligned only when a recognizable "eval probe" is present, no single onset point
14. **RLHF Reward-Model Overoptimization** — Toy preference-bandit exploiting a real (if tiny) fitted reward model's blind spot

### Families 15–18 (v1.5, MuJoCo / High-Dimensional Continuous Control)

Populate the taxonomy's `CONTINUOUS_COMPLEX` ("cont_hd") tier for the first time — 2 mechanisms re-instantiated from the existing taxonomy at real MuJoCo dimensionality (HalfCheetah, Reacher), plus 2 genuinely new MuJoCo-native mechanisms (Ant, Walker2d), all reusing existing `HackingMechanism` values rather than expanding the taxonomy.

15. **MuJoCo Camping** (HalfCheetah-v5) — The classic flip-and-slide MuJoCo locomotion exploit: a genuine bounding gait vs. a wind-up/flip/calibrated-slide hack that games the same forward-velocity reward
16. **MuJoCo Goal Misgeneralization** (Reacher-v5) — Direct port of Family 7's goal-swap construction onto a real 2-joint arm's fingertip position
17. **MuJoCo Joint-Limit Gaming** (Ant-v5) — A gait that stays safely within each joint's real physical limit vs. one that games near the limit for the same measured reward
18. **MuJoCo Sensor-Channel Decoupling** (Walker2d-v5) — The documented sim-to-real foot-slip exploit: a spoofable joint-velocity "sensor" reads high without real forward progress

### Families 19–23 (v1.6, RLHF-RM / Synthetic Reward-Model Overoptimization)

Populate the taxonomy's `SEQUENTIAL` tier for the first time — a synthetic RLHF setting (feature-vector "responses," a genuinely-fit preference reward model, and policy-gradient optimization with a KL penalty to a reference policy) rather than a real LLM, so the reward-hacking dynamics arise from real data-fitting and optimization instead of a scripted proxy/true gap. Each family varies exactly one way the fitted reward model goes wrong.

19. **RM Sparse-Coverage Gaming** — Preference data undersamples part of response-space; the fitted reward model extrapolates optimistically there, and the policy drifts into that blind spot
20. **RM Label-Noise Exploitation** — Preference labels near the true decision boundary carry concentrated noise, biasing the fitted model's boundary in one consistent direction
21. **RM Feature-Blindspot Gaming** — The reward model is fit on a truncated subset of response features (a fixed representation that can't see the rest), so its policy is structurally frozen on the hidden dimensions where true value is left on the table
22. **KL-Penalty Gaming** — Both variants share the identical reward model; only the KL-penalty coefficient differs, letting an under-penalized policy drift past the true reward's optimum into a declining region
23. **Preference-Population Bias** — The synthetic labeler population systematically over-weights one response dimension unrelated to true quality (a sycophancy-style bias), and the fitted model faithfully learns it

### Families 24–28 (v1.7, PettingZoo / Multi-Agent)

Populate the taxonomy's `MULTI_AGENT` complexity tier for the first time, using real `mpe2` (Multi-Particle Environments, split out of PettingZoo core) scenarios rather than a scripted single-agent proxy. Each family re-instantiates an existing `HackingMechanism` value at genuine multi-agent scale — no taxonomy expansion.

24. **Tacit Collusion Gaming** (`simple_tag`) — Predators quietly agree to camp near the prey's spawn instead of genuinely pursuing it, gaming the team-summed proxy during a late-episode "collusion phase"
25. **Communication-Channel Deception** (`simple_speaker_listener`) — A speaker's message is scripted to leak the true goal signal through a side channel while nominally still "communicating," inflating the listener's proxy without honestly conveying the goal
26. **Free-Rider Exploitation** (`simple_spread`, team-averaged) — One of 3 agents abandons its assigned landmark while the other 2 compensate, gaming a team-averaged proxy that can't see which individual contributed
27. **Fixed-Opponent Exploitation** (`simple_tag`) — Predators exploit a fixed, scripted prey's predictable corner-retreat pattern instead of genuinely tracking its live position
28. **Population-Level Goodhart** (`simple_world_comm`) — One of 4 adversary-team agents free-rides while the lead and 2 others carry the team's aggregate proxy reward — population-scale free-riding, distinct from Family 26's 3-agent case

### Families 29–33 (v1.8, Sequence Generation / Non-RLHF SEQUENTIAL)

A second, structurally distinct population of the `SEQUENTIAL` complexity tier alongside Families 19–23: real per-step token-sequence generation over a small discrete vocabulary against a fixed hidden Markov "true grammar," scored by a fixed rule-based (non-fitted) proxy rather than a fitted reward model. Each family games a different, real, documented LLM reward-hacking failure mode.

29. **Keyword-Stuffing Gaming** — Floods a small fixed keyword-token subset largely decoupled from the grammar's actual structure, gaming a keyword-frequency proxy
30. **Format-Compliance Camping** — Fills fixed template-slot positions with expected tokens while the surrounding sequence needn't cohere, gaming a template-slot-fill-rate proxy
31. **Repetition-Blind-Spot Shortcut** — Repeats a short token cycle just outside a fixed lookback window, scoring perfectly on a "no recent duplicate" check while remaining globally highly repetitive
32. **Lexicon-Sentiment Gaming** — Floods a fixed positive-lexicon token subset largely decoupled from the grammar, gaming a sentiment-proxy stand-in for "helpful/positive" quality
33. **Length-Padding Exploit** — Switches to uniform-random low-effort tokens partway through the episode, farming a per-step length/continuation shaping bonus that a shallow content-presence check can't tell apart from genuine content

## The 35 Detectors

### L0: Reward-Only (13)
Temporal statistics on proxy reward only. Tautologically fail on matched-proxy families.

### L1: State-Visitation (8)
Occupancy patterns, state-frequency anomaly, centroid drift, etc. Transfer completely on admission-gated families.

### L2: Behavioral (8)
Hand-engineered anti-symmetric features from trajectory traces. Transfer excellently to unseen families.

### L3: Oracle (2)
- **True Reward Oracle**: Direct access to ground-truth true reward (ceiling measurement)
- **Perfect Feature Oracle**: Direct access to the exact anti-symmetric feature each family was designed around

### External Baselines (5)
Classical, citable methods from the wider change-point/anomaly-detection/representation-learning/Bayesian-inference literature, not designed for RHOB — included so the leaderboard compares detection *approaches* fairly, not just in-house features:

| Detector | Access | Method family | Reference |
|---|---|---|---|
| Page-Hinkley Test | L0 | Classical change-point | Page (1954); Hinkley (1971) |
| Isolation Forest | L2 | Unsupervised anomaly detection | Liu, Ting & Zhou (2008) |
| AR(p) Residual | L2 | Sequence model | Classical autoregressive baseline |
| PCA Reconstruction | L1 | Representation learning | Linear reconstruction-error baseline |
| Bayesian Online Changepoint Detection | L0 | Bayesian inference | Adams & MacKay (2007) |

See [`src/rhob/detectors/external_baselines/`](src/rhob/detectors/external_baselines/).

## Running Experiments

### Regenerate the full v5 leaderboard (35 × 33)

```bash
python scripts/v5_leaderboard_and_transfer.py
# Outputs: leaderboard/v5_leaderboard.json
```

Runtime: ~2–3 hours on a CPU. Generates detection AUROC for every detector on every family's default-difficulty pair.

### Run the cross-family transfer experiment

```bash
python scripts/cross_family_transfer.py --n-seeds-train 15 --n-seeds-test 20
# Outputs: leaderboard/cross_family_transfer.json
```

Trains Reward MLP, State Divergence, Trajectory MLP, and a Top-5 L2 ensemble on Families 1–6, evaluates on 8 held-out Families 7–14.

Runtime: ~1–2 hours.

### Regenerate figures

```bash
python scripts/plot_v5_results.py
# Outputs: docs/figures/v5_heatmap.png, v5_access_summary.png, v5_transfer.png
```

## Adding a New Detector

```python
from rhob.detectors.posthoc import PosthocDetector, RunData
from sklearn.metrics import roc_auc_score

class MyDetector(PosthocDetector):
    @property
    def access_level(self) -> str:
        return "L2"  # or "L0", "L1", "L3"

    @property
    def name(self) -> str:
        return "My Custom Detector"

    def classify(self, run: RunData) -> float:
        """Return a score: higher = more likely hacking."""
        # run.proxy_rewards, run.behav_trace, run.state_counts available based on access_level
        if run.behav_trace is None:
            return 0.5
        return float(run.behav_trace[-100:].mean())

    def detect_onset(self, run: RunData) -> int:
        """Return the first episode where hacking begins, or -1 if not detected."""
        trace = run.behav_trace
        if trace is None or len(trace) < 10:
            return -1
        for t in range(10, len(trace)):
            if abs(trace[t]) > 0.5:
                return t
        return -1
```

Then evaluate:

```python
from rhob.v3.benchmark import Benchmark

detector = MyDetector()
results = Benchmark.evaluate(detector, families=["gridworld_camping"], n_seeds=10)
print(results.overall_auroc)
```

## Adding a New Family

Subclass `BaseFamily`, implement `generate_pair(difficulty, seed)`, which returns a `MatchedPair`:

```python
from rhob.v3.base_family import BaseFamily, MatchedPair
from rhob.v3.registry import FamilyRegistry

@FamilyRegistry.register("my_family")
class MyFamily(BaseFamily):
    @property
    def name(self) -> str:
        return "my_family"

    def difficulty_range(self) -> tuple[float, float]:
        return (0.60, 0.98)

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        # Return a MatchedPair with hacking and legitimate rollout functions
        # and a proxy-preserving symmetry σ
        ...
```

**New families must pass the admission gate** (5 automated checks certifying matched proxy, behavioral separation, true-reward divergence, onset localizability, and camping quality).

See `src/rhob/v3/families/` for examples. For a guided walkthrough, see the
[Environment Tutorial](docs/TUTORIAL_ENVIRONMENT.md).

## Documentation

| Doc | For |
|---|---|
| [docs/INSTALL.md](docs/INSTALL.md) | Setup, Docker, Colab, troubleshooting |
| [docs/TUTORIAL_DETECTOR.md](docs/TUTORIAL_DETECTOR.md) | Evaluate or add a detector in <30 min |
| [docs/TUTORIAL_ENVIRONMENT.md](docs/TUTORIAL_ENVIRONMENT.md) | Add a new hacking-mechanism family |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Submission process and admission-gate requirements |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Regenerate every experiment and figure from scratch |
| [docs/site/index.html](docs/site/index.html) | Benchmark website (GitHub Pages) |

## Reproducibility

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for:
- How to reproduce all experiments from scratch
- How to regenerate all figures
- Detailed experiment design and random seed strategy
- How to verify the admission gate on new families

## Paper & Citation

The accompanying paper is maintained separately from this codebase (see the link on
the [benchmark website](docs/site/index.html) once published). This repository is
the benchmark and evaluation harness; result figures referenced by the paper live in
[`docs/figures/`](docs/figures/) and are fully reproducible from the scripts and
leaderboard data committed here (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

If you use RHOB, please cite:

```bibtex
@article{shah2026rhob,
  title={RHOB v1.0: Generalizable Reward Hacking Detection Through Matched-Proxy Benchmarking},
  author={Shah, Aarav},
  journal={TMLR},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Contributing

We welcome new families and detectors! See [CONTRIBUTING.md](CONTRIBUTING.md) for the submission process and admission-gate requirements.

## Links

- **Benchmark Website**: https://aarav500.github.io/rhob/ (GitHub Pages, [source](docs/site/index.html))
- **Interactive Leaderboard**: live at [AWS EC2](http://54.208.200.139/) and (once deployed)
  [HF Space](https://huggingface.co/spaces/Aarav500/rhob-leaderboard); or run locally with
  `pip install -e ".[space]" && python space/app.py` ([source](space/app.py)) -- see
  [docs/DEPLOY_SPACE.md](docs/DEPLOY_SPACE.md) for the HF Space deploy steps
- **Submit a detector result**: drop a submission JSON in [`submissions/`](submissions/)
  and open a PR (auto-validated by CI) -- see [submissions/README.md](submissions/README.md)
- **GitHub**: https://github.com/Aarav500/rhob
- **Paper**: https://arxiv.org/abs/... (coming soon)
