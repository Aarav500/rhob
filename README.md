# RHOB — Reward-Hacking Onset Benchmark

RHOB is a standardized benchmark for **early detection of reward hacking**. It
provides instrumented environments with ground-truth *onset* timestamps,
standardized metrics, and a simple detector interface, so that detection methods
can be compared head-to-head for the first time.

> **Status: Milestone 1 (Vertical Slice).** This release ships one complete
> end-to-end pipeline: the `GridWorld-Wireheading` environment, the `Random` and
> `CUSUM` baselines, the full metrics suite, HDF5 storage, and the evaluation
> runner. Additional environments, baselines (Flight Recorder, ensembles, …),
> tiers, and the leaderboard arrive in later milestones.

---

## What RHOB measures

Given a training run whose **proxy** reward is rising, is the agent *learning*
(good) or *hacking* (bad)? RHOB frames this as **onset detection**: find the
first training step at which the true objective begins to degrade while the
proxy keeps improving. Detectors never see the true reward — they must infer the
onset from what a real monitor could observe.

Each environment is a **data generator**: it records complete training runs
(proxy/true return curves plus behavioural features) and attaches an oracle
onset label. Detectors are evaluated on these pre-recorded trajectories, which
makes evaluation fast, deterministic, and fully reproducible.

---

## Installation

```bash
git clone <repo-url> rhob
cd rhob
pip install -e ".[dev]"      # core + test/lint tooling
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `scikit-learn`,
`h5py`, `pydantic`, `click`. (The `environments` extra adds `gymnasium` and
`stable-baselines3` for PPO-based generation in future continuous-control
tiers; the Tier 1 tabular environment does not need them.)

---

## Quickstart

### 1. Generate the benchmark dataset

```bash
python scripts/generate_gridworld_data.py --output data/gridworld_wireheading.h5
```

This trains tabular Q-learning agents on `GridWorld-Wireheading` (7 hacking + 3
clean runs by default), labels the onsets, and writes an HDF5 dataset.

### 2. Evaluate detectors (the "10-line evaluation")

```python
import rhob

trajectories = rhob.load_dataset("data/gridworld_wireheading.h5")

reports = rhob.compare(
    [rhob.RandomDetector(), rhob.CUSUMDetector()],
    trajectories,
)
print(rhob.results_table(reports))
```

```
| Method | Access | RHOB-Score | 99% CI          | Mean AUROC | Miss | TFD   | FPR@k |
|--------|--------|------------|-----------------|------------|------|-------|-------|
| CUSUM  | L1     | 0.994      | [0.989, 0.997]  | 0.994      | 0.00 | 0.002 | 0.300 |
| Random | L1     | 0.494      | [0.484, 0.498]  | 0.494      | 0.00 | —     | 0.767 |
```

### 3. Write your own detector

```python
import rhob

class MyDetector(rhob.Detector):
    name = "my_detector"
    id = "user/my_detector"
    access_level = rhob.AccessLevel.L1   # reward-only

    def reset(self):
        self.history = []

    def step(self, obs):
        self.history.append(obs.reward)
        # ... return a hacking probability in [0, 1] ...
        return 0.0

report = rhob.evaluate(MyDetector(), trajectories)
print(report.rhob_score)
```

Subclass `rhob.Detector`, declare an `access_level`, implement `reset()` and
`step()`. The runner enforces the detector contract (bounded, deterministic,
access-compliant scores) before scoring.

---

## The environment: GridWorld-Wireheading

A Tier 1 **reward-tampering** environment. The agent should reach and hold a
goal cell, but an adjacent *wirehead* cell pays large proxy reward for nothing.
Early in training the agent legitimately learns the goal (proxy and true rise
together); part-way through, the tampering opportunity activates and the agent
switches to farming it — the proxy keeps rising while the true return collapses.
That divergence is the **onset**.

See [`docs/data_schema.md`](docs/data_schema.md) for the onset definition, the
data format, and the generation design (including why tabular Q-learning is used
for this discrete environment), and [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md)
for the full record of engineering and research decisions (alternatives
considered, rejection rationale, and implications).

---

## Metrics

| Metric | Meaning |
|---|---|
| **AUROC / AUPRC** | Per-step discrimination of post-onset vs. pre-onset/clean steps |
| **Detection latency** | Normalized `(t_detect − t*) / T` |
| **Miss rate** | Fraction of hacking runs never detected |
| **TFD** | Median latency among detected runs |
| **FPR@k** | False-positive rate among the *k* most confident alerts |
| **RHOB-Score** | Tier-weighted mean AUROC (primary ranking metric) |

All aggregates are reported with bootstrap confidence intervals.

---

## Access levels

Detectors declare how much information they consume; the access filter
*structurally* prevents seeing anything above the declared level.

| Level | Sees |
|---|---|
| **L1** | Proxy reward only |
| **L2** | + policy/behavioural features |
| **L3** | + gradients / KL (future) |
| **L4** | + internal state (future) |

---

## Specification suite

The benchmark is defined by a set of frozen specification documents (the
foundation for all future development):

| Document | Purpose |
|---|---|
| [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md) | Vision, scope, definitions, assumptions, success criteria |
| [ENVIRONMENT_SPEC.md](ENVIRONMENT_SPEC.md) | Environment interface (add environments without touching core) |
| [DETECTOR_API.md](DETECTOR_API.md) | Standardized detector interface (statistical + deep) |
| [METRICS_SPEC.md](METRICS_SPEC.md) | Official metrics, when to report, visualization |
| [DATASET_SPEC.md](DATASET_SPEC.md) | HDF5 layout + external-dataset compatibility |
| [CONFIG_SPEC.md](CONFIG_SPEC.md) | Configuration system and YAML schemas |
| [LEADERBOARD_SPEC.md](LEADERBOARD_SPEC.md) | Submission format, protocol, reproducibility |
| [REPOSITORY_ARCHITECTURE.md](REPOSITORY_ARCHITECTURE.md) | Modules, dependencies, extension points |
| [docs/difficulty_spectrum.md](docs/difficulty_spectrum.md) | Tier 2 continuous pairs (02–04): the tunable difficulty spectrum |
| [results/detector_evaluation/](results/detector_evaluation/) | The 6-detector × 4-pair results table: L0 barrier, L2 sufficiency |
| [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) | Every significant decision and its rationale |
| [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) | Critical review of the implementation |
| [REFACTOR_PLAN.md](REFACTOR_PLAN.md) · [ROADMAP_NEXT.md](ROADMAP_NEXT.md) | Prioritized refactors and next milestones |

Stability tags used throughout: **[Stable]** frozen for 1.x · **[Provisional]**
may change once · **[Planned]** specified but not yet implemented.

---

## Development

```bash
pytest                       # run the test suite (76 tests)
pytest --cov=rhob            # with coverage (≈96%)
ruff check src tests         # lint
ruff format src tests        # format
```

---

## License

MIT — see [LICENSE](LICENSE).
