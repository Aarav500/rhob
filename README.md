# RHOB — Reward Hacking Onset Benchmark

[![tests](https://github.com/Aarav500/rhob/actions/workflows/tests.yml/badge.svg)](https://github.com/Aarav500/rhob/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive benchmark for **detecting reward hacking across diverse mechanisms**.

**RHOB** provides:
- **9 environment families** spanning 9 distinct hacking mechanisms (camping exploits, goal misgeneralization, distributional shift, etc.)
- **35 detectors** across 4 access levels (reward-only to oracle), including 5 classical external baselines (Bayesian changepoint, isolation forest, PCA, etc.)
- **Matched-proxy construction** ensuring hacking/legitimate improvement produce identical proxy rewards
- **Cross-family transfer analysis** measuring detector generalization to unseen mechanisms
- **Admission gate** certifying families measure hacking detection, not just change detection

**Can a PhD student evaluate their detector in under 30 minutes?** That's the
bar — see the [Detector Tutorial](docs/TUTORIAL_DETECTOR.md).

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

| Access Level | Typical AUROC | Key Finding |
|---|---|---|
| **L0** (reward-only) | ~0.50 (chance) | Matched proxy = information barrier |
| **L1** (state-visitation) | ~0.53 | Raw histograms fail due to representational incompatibility across families |
| **L2** (behavioral) | ~0.69 (in-dist), 0.95 (transfer) | Compact features transfer excellently to unseen mechanisms |
| **L3** (oracle) | ~0.87 | Ground-truth provides ceiling for measurement |

**Cross-family transfer (train on 6 families, test on 3 held-out):**
- L0/L1 detectors: pinned at chance on every held-out family
- L2 detectors: **0.95 average transfer AUROC**, exceeding in-distribution performance
- L2 ensemble: **perfect 1.00 transfer** on all three held-out families

**Key insight:** Transfer depends on **representation abstraction**, not access level. Raw state histograms are dimensioned per-family and structurally incompatible across mechanisms.

## The 9 Families

### Families 1–6 (v3.2 Baselines)

1. **Gridworld Camping** — Discrete tile-camping (minimal proof-of-concept)
2. **Continuous Camping** — 2D point-mass (continuous difficulty spectrum)
3. **Proxy Correlation Gaming** — Exploit spurious feature correlation
4. **Shortcut Exploitation** — Take forbidden direct path vs. scenic detour
5. **Novelty Farming** — Revisit exploration frontier for novelty bonus
6. **Orbit Chirality** — Angular pattern (deceptive-alignment analogue)

### Families 7–9 (v5 New, Held-out for Transfer)

7. **Goal Misgeneralization** — Reach wrong goal (mechanistically distinct from 1–6)
8. **Physics Exploitation** — Jump high but crash (physics safety vs. proxy metric)
9. **Distributional Shift** — Overfit to train distribution, fail on test

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
# Outputs: paper/figures/v5_heatmap.png, v5_access_summary.png, v5_transfer.png
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

- **Paper (v1.0)**: `paper/main.tex` (PDF available in releases)
- **Supplementary material**: `supplementary_material/`

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
- **Interactive Leaderboard**: https://huggingface.co/spaces/Aarav500/rhob-leaderboard (coming soon)
- **GitHub**: https://github.com/Aarav500/rhob
- **Paper**: https://arxiv.org/abs/... (coming soon)
