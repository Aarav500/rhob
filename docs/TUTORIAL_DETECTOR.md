# Detector Tutorial: Evaluate Yours in Under 30 Minutes

This walks through the fastest path from "I have a detection idea" to "I have
an AUROC number on the RHOB leaderboard." Total time: ~20-30 minutes,
including a coffee break while the evaluation runs.

## 0. Install (2 min)

```bash
pip install -e ".[dev]"
```

See [INSTALL.md](INSTALL.md) if this fails.

## 1. Understand the interface (5 min)

Every post-hoc detector implements `rhob.detectors.posthoc.PosthocDetector`:

```python
from rhob.detectors.posthoc import PosthocDetector, RunData

class MyDetector(PosthocDetector):
    @property
    def access_level(self) -> str:
        return "L2"  # "L0" reward-only, "L1" +state-visitation, "L2" +behavioral trace

    @property
    def name(self) -> str:
        return "My Detector"

    def classify(self, run: RunData) -> float:
        """Higher score = more likely hacking. Must be in [0, 1] or monotone-comparable."""
        ...

    def detect_onset(self, run: RunData) -> int:
        """First episode where hacking begins, or -1 if never detected."""
        ...
```

`RunData` gives you exactly the fields your declared access level is allowed
to see:

| Field | Available at | Meaning |
|---|---|---|
| `proxy_rewards` | L0+ | Per-episode proxy reward (what the agent optimizes) |
| `state_counts` | L1+ | Per-episode state-visitation histogram |
| `behav_trace` | L2+ | Anti-symmetric behavioral feature (mean at 0 pre-onset, flips sign post-onset per variant) |
| `true_rewards` | oracle-only | **Never read this in a real detector** — it's the ground truth used to grade you |

If your detector needs training data (most L1/L2 detectors do), add an
optional `fit(runs_a, runs_b)` method — `runs_a` is the hacking variant,
`runs_b` is legitimate.

**Don't know where to start?** Copy the smallest existing detector at your
access level and modify it:
- L0: [`src/rhob/detectors/l0_reward_threshold.py`](../src/rhob/detectors/l0_reward_threshold.py)
- L1: [`src/rhob/detectors/l1_state_divergence.py`](../src/rhob/detectors/l1_state_divergence.py)
- L2: [`src/rhob/detectors/l2_behavioral_threshold.py`](../src/rhob/detectors/l2_behavioral_threshold.py)

## 2. Evaluate on one family (5 min)

```python
from rhob.v3.benchmark import Benchmark
from my_detector import MyDetector

detector = MyDetector()
results = Benchmark.evaluate(detector, families=["gridworld_camping"], n_seeds=10)
print(f"AUROC: {results.overall_auroc:.3f}")
```

`Benchmark.evaluate` runs 5-fold stratified cross-validation automatically —
you don't need to split data yourself unless your detector needs a
`fit()` step, in which case pass labeled runs from the training fold.

## 3. Evaluate across all 14 families (10-15 min, mostly runtime)

```python
from rhob.v3.registry import FamilyRegistry

all_families = FamilyRegistry.list_names()  # all 9
results = Benchmark.evaluate(detector, families=all_families, n_seeds=15)
for fam, auroc in results.per_family_auroc.items():
    print(f"{fam:30s} {auroc:.3f}")
```

## 4. Sanity-check against the admission gate expectations

Your detector should score:
- **~0.50** on families where your access level is information-starved (e.g.
  an L0 detector on a family with a genuinely matched proxy — this is
  correct, not a bug)
- **Meaningfully above 0.50** wherever your access level has the information
  needed to discriminate

If an L2+ detector scores at chance everywhere, something is probably wrong
with feature extraction, not the family (see
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the debugging playbook we
used on real bugs of exactly this shape).

## 5. Test cross-family transfer (optional, ~2 min setup)

Train on Families 1-6, evaluate frozen on held-out Families 7-9 — the
strongest signal of whether your detector learned something general or just
memorized one mechanism:

```bash
python scripts/cross_family_transfer.py --n-seeds-train 15 --n-seeds-test 20
```

## 6. Submit

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the submission workflow (tests
required, code style, PR checklist).
