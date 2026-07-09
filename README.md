# RHOB — Reward Hacking Onset Benchmark

[![tests](https://github.com/Aarav500/rhob/actions/workflows/tests.yml/badge.svg)](https://github.com/Aarav500/rhob/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive benchmark for **detecting reward hacking across diverse mechanisms**.

**RHOB** provides:
- **14 environment families** spanning 9 distinct hacking mechanisms (camping exploits, goal misgeneralization, distributional shift, reward tampering, deceptive alignment/sandbagging, RLHF reward-model overoptimization, etc.)
- **35 detectors** across 4 access levels (reward-only to oracle), including 5 classical external baselines (Bayesian changepoint, isolation forest, PCA, etc.)
- **Matched-proxy construction** ensuring hacking/legitimate improvement produce identical proxy rewards
- **Cross-family transfer analysis** measuring detector generalization to unseen mechanisms
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

**Cross-family transfer (train on 6 families, test on 3 held-out; neural-net detectors reported as mean ± std across 3 independently-seeded training runs — see caveat below):**
- L0/L1 detectors: pinned at chance on every held-out family
- L2 single learned detector (Trajectory MLP): **0.77 ± 0.14 average transfer AUROC**, below its 0.93 training AUROC, with high seed-to-seed variance on one held-out family
- L2 five-detector ensemble: **1.00 ± 0.00 transfer AUROC**, matching its 0.97 training AUROC — robust because 4 of its 5 members are deterministic

**Key insight:** Transfer depends on **representation abstraction**, not access level — but a single learned detector is only as reliable as its training procedure. We found `TrajectoryMLPDetector` doesn't seed its `torch` weight initialization: repeating the identical fit on identical data 10 times produced held-out AUROC on one family ranging from 0.00 to 1.00. Ensembling deterministic behavioral-threshold detectors alongside the learned one is what actually makes transfer reliable. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full methodological history — three real family-implementation bugs and this reproducibility bug were all found by treating implausible numbers as bugs to investigate, not results to report.

## The 14 Families

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

### Regenerate the full v5 leaderboard (30 × 9)

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

Trains Reward MLP, State Divergence, Trajectory MLP, and a Top-5 L2 ensemble on Families 1–6, evaluates on held-out Families 7–9.

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
