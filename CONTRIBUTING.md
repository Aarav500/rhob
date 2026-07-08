# Contributing to RHOB v3

Welcome! This guide helps you contribute families, detectors, or improvements to RHOB v3. All contributions follow the matched-proxy methodology and pass community review.

**New here?** Start with the 30-minute [Detector Tutorial](docs/TUTORIAL_DETECTOR.md)
or [Environment Tutorial](docs/TUTORIAL_ENVIRONMENT.md) before reading this
full reference guide.

## Table of Contents
1. [Before You Start](#before-you-start)
2. [Contributing a Family](#contributing-a-family)
3. [Contributing a Detector](#contributing-a-detector)
4. [Testing & Validation](#testing--validation)
5. [Submission Workflow](#submission-workflow)
6. [Code Style](#code-style)

---

## Before You Start

### Read These First
- **API Specification** (`docs/API_SPECIFICATION.md`) — Interface contracts (frozen for v3.2+)
- **Anti-Symmetry Screening Rule** (`docs/families_5_8_redesign.md` section "The single root cause") — How families pass admission
- **Example Families** — Read one shipped family fully:
  - `src/rhob/v3/families/gridworld_camping.py` (tabular, simple)
  - `src/rhob/v3/families/novelty_farming.py` (continuous, roaming)
  - `src/rhob/v3/families/orbit_chirality.py` (continuous, orbital)

### Key Principles
1. **Matched proxy by construction** — Not by fine-tuning
2. **Anti-symmetric features** — Must flip sign under pair's symmetry `σ`
3. **Deterministic** — Same seed = same rollout, always
4. **Testable** — <60s per pair, handles small sample sizes
5. **Documented** — Clear parameter semantics

---

## Contributing a Family

### Step 1: Design & Screening

Before coding, complete a design audit:

**1. State your symmetry `σ`**
- Explicit mapping (reflection, transpose, swap, etc.)
- Example: "Color swap σ: blue ↔ red labels"

**2. State your proxy**
- Exactly how it's computed
- Why it's invariant under σ
- Example: "Total items (blue + red), weighted equally → σ-invariant"

**3. State your discriminating feature**
- What distinguishes hacking from legitimate
- Why it's anti-symmetric under σ
- NOT a magnitude (count, rate, convergence)
- Example: "(red_count - blue_count) / total → flips sign under σ"

**4. State your difficulty knob**
- Parameter that moves L2-AUROC smoothly
- Interpretation (what does it control?)
- Example: "Item spatial correlation ∈ [0.2, 1.0] → L2 ∈ [0.98, 0.60]"

**Submit this as a GitHub issue (tagged `design-audit`) before coding.** Community reviews for soundness.

### Step 2: Implement

Create `src/rhob/v3/families/my_family.py`:

```python
"""Family: My Clever Hacking Mechanism (brief description).

Detailed docstring:
- Symmetry σ: explicit description
- Proxy: how it's computed and why it's σ-invariant
- Feature: discriminating signal, why anti-symmetric
- Difficulty knob: parameter & interpretation
- Access profile (L0 fails? L1 succeeds? etc.)
"""

from __future__ import annotations

import numpy as np
from rhob.detectors.posthoc import RunData
from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair, RolloutResult
from rhob.v3.registry import FamilyRegistry
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism


@FamilyRegistry.register("my_family")
class MyFamily(BaseFamily):
    """Family name: one-liner description."""

    def __init__(self):
        """Store immutable environment params."""
        self.param1 = 10.0
        self.n_episodes = 200

    @property
    def name(self) -> str:
        return "my_family"

    @property
    def mechanism(self) -> HackingMechanism:
        # Pick one from HackingMechanism enum
        return HackingMechanism.CAMPING_EXPLOIT

    @property
    def complexity(self) -> EnvironmentComplexity:
        # Pick one from EnvironmentComplexity enum
        return EnvironmentComplexity.TABULAR

    def difficulty_range(self) -> tuple[float, float]:
        # (min_L2, max_L2) your family can generate
        return (0.60, 0.98)

    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        # Map difficulty to knob parameter
        # difficulty=0.98 (EASY) → knob=X  |  difficulty=0.60 (HARD) → knob=Y
        t = (difficulty - 0.60) / (0.98 - 0.60)
        knob_param = X + (Y - X) * t

        # Define rollout functions (deterministic closures)
        def rollout_hacking(s: int) -> RolloutResult:
            proxy_r, true_r, behav = _simulate_hacking(knob_param, s)
            run = RunData(
                proxy_rewards=proxy_r,
                true_rewards=true_r,
                state_counts=None,  # or histogram if L1-capable
                behav_trace=behav,  # or None if L1-only
            )
            return run, 0  # (RunData, onset_episode)

        def rollout_legit(s: int) -> RolloutResult:
            proxy_r, true_r, behav = _simulate_legit(knob_param, s)
            run = RunData(
                proxy_rewards=proxy_r,
                true_rewards=true_r,
                state_counts=None,
                behav_trace=behav,
            )
            return run, -1  # No onset for legitimate

        return MatchedPair(
            family=self.name,
            mechanism=self.mechanism,
            complexity=self.complexity,
            difficulty=difficulty,
            n_episodes=self.n_episodes,
            rollout_hacking=rollout_hacking,
            rollout_legit=rollout_legit,
            params={"knob_param": round(knob_param, 4)},
        )


def _simulate_hacking(knob_param, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate hacking variant. Return (proxy_rewards, true_rewards, behav_trace)."""
    rng = np.random.default_rng(seed)
    # ... implementation ...
    return proxy_r, true_r, behav


def _simulate_legit(knob_param, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate legitimate variant."""
    rng = np.random.default_rng(seed)
    # ... implementation ...
    return proxy_r, true_r, behav
```

### Step 3: Test

Create `tests/test_v3/test_my_family.py`:

```python
"""Tests for my_family."""

from rhob.v3.registry import FamilyRegistry
from sklearn.metrics import roc_auc_score


def test_family_registered():
    """Family is discoverable."""
    fam = FamilyRegistry.get("my_family")
    assert fam.name == "my_family"


def test_difficulty_range():
    """Range is sensible."""
    fam = FamilyRegistry.get("my_family")
    lo, hi = fam.difficulty_range()
    assert lo < hi
    assert 0.5 < lo < 1.0
    assert 0.5 < hi < 1.0


def test_proxy_matched():
    """Proxy totals are similar between hacking and legitimate."""
    fam = FamilyRegistry.get("my_family")
    pair = fam.generate_pair(0.90, seed=0)
    
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    
    proxy_a = run_a.proxy_rewards.sum()
    proxy_b = run_b.proxy_rewards.sum()
    
    # Allow ±20% variance due to randomness
    assert abs(proxy_a - proxy_b) / max(abs(proxy_a), abs(proxy_b)) < 0.2


def test_true_reward_diverges():
    """True rewards diverge (hacking low, legitimate high)."""
    fam = FamilyRegistry.get("my_family")
    pair = fam.generate_pair(0.90, seed=0)
    
    run_a, _ = pair.rollout_hacking(0)
    run_b, _ = pair.rollout_legit(0)
    
    true_a = run_a.true_rewards.sum()
    true_b = run_b.true_rewards.sum()
    
    assert true_a < true_b


def test_difficulty_sweep():
    """Difficulty knob moves L2 smoothly."""
    from rhob.detectors import BehavioralThresholdDetector
    
    fam = FamilyRegistry.get("my_family")
    detector = BehavioralThresholdDetector()
    
    aurocs = []
    for d in [0.98, 0.80, 0.60]:
        pair = fam.generate_pair(d, seed=0)
        runs_a, runs_b, _ = pair.rollout(n_seeds=5)
        
        scores_a = [detector.classify(r) for r in runs_a]
        scores_b = [detector.classify(r) for r in runs_b]
        
        y_true = [1] * len(scores_a) + [0] * len(scores_b)
        y_scores = scores_a + scores_b
        auroc = roc_auc_score(y_true, y_scores)
        aurocs.append(auroc)
    
    # AUROC should decrease with difficulty
    assert aurocs[0] > aurocs[2]
```

---

## Contributing a Detector

### Step 1: Decide Access Level

- **L0** (Reward-only): Use only `proxy_rewards`
- **L1** (State-visitation): Use `proxy_rewards` + `state_counts`
- **L2** (Behavioral): Use `proxy_rewards` + `behav_trace` + `state_counts`
- **L3** (Oracle): Use `true_rewards` (never scored in production)

### Step 2: Implement

Create `src/rhob/detectors/lX_my_detector.py`:

```python
"""My Detector (L0/L1/L2 access level). Brief description."""

from __future__ import annotations
import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class MyDetector(PosthocDetector):
    """Human-readable name for leaderboard."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    @property
    def access_level(self) -> str:
        return "L0"  # or "L1", "L2"

    @property
    def name(self) -> str:
        return "My Detector"

    def classify(self, run: RunData) -> float:
        """Return P(hacking) ∈ [0, 1].
        
        - Access violations (e.g., L0 reading true_rewards) raise AttributeError
        - Short runs (< min window) return 0.5 (neutral)
        - Score must be deterministic
        """
        rewards = run.proxy_rewards
        if len(rewards) < 2 * self.window_size:
            return 0.5
        
        # Compute some statistic over windows
        baseline = np.mean(rewards[:self.window_size])
        test = np.mean(rewards[-self.window_size:])
        diff = test - baseline
        
        # Map to [0, 1] via sigmoid or normalization
        score = 1.0 / (1.0 + np.exp(-diff))
        return float(np.clip(score, 0.0, 1.0))

    def detect_onset(self, run: RunData) -> int:
        """Return episode where onset detected, or -1 if none."""
        rewards = run.proxy_rewards
        if len(rewards) < self.window_size:
            return -1
        
        baseline = np.mean(rewards[:self.window_size])
        
        for t in range(self.window_size, len(rewards)):
            window = rewards[t - self.window_size:t + 1]
            test = np.mean(window)
            if test - baseline > 0.5:  # Threshold
                return t
        
        return -1

    # Optional: supervised training
    def fit(self, runs_a: list[RunData], runs_b: list[RunData], epochs: int = 10) -> None:
        """Fit detector on labeled hacking (runs_a) and legitimate (runs_b) runs.
        
        Only implement if your detector needs supervision (e.g., neural network).
        Called during cross-validation.
        """
        # Extract features from runs_a and runs_b, train model, set is_trained=True
        pass

    @property
    def is_trained(self) -> bool:
        """Return True iff detector is in trained state."""
        return getattr(self, '_trained', False)
```

### Step 3: Add to Exports

Edit `src/rhob/detectors/__init__.py`:

```python
from rhob.detectors.lX_my_detector import MyDetector  # Add import

__all__ = [
    ...
    "MyDetector",  # Add to __all__
]
```

### Step 4: Test

Create `tests/test_detectors/test_lX_my_detector.py`:

```python
"""Tests for my detector."""

import numpy as np
from rhob.detectors import MyDetector
from rhob.detectors.posthoc import RunData


def test_access_level():
    """Detector claims correct access."""
    det = MyDetector()
    assert det.access_level in ["L0", "L1", "L2", "L3"]


def test_classify_returns_probability():
    """Score is in [0, 1]."""
    det = MyDetector()
    rewards = np.random.rand(100)
    run = RunData(rewards, np.zeros(100), None, None)
    
    score = det.classify(run)
    assert 0.0 <= score <= 1.0


def test_short_run_neutral():
    """Short runs return 0.5 (neutral)."""
    det = MyDetector()
    rewards = np.random.rand(5)
    run = RunData(rewards, np.zeros(5), None, None)
    
    score = det.classify(run)
    assert score == 0.5


def test_detect_onset():
    """Onset detection returns -1 or valid episode."""
    det = MyDetector()
    rewards = np.concatenate([np.ones(50), np.ones(50) * 10])
    run = RunData(rewards, np.zeros(100), None, None)
    
    onset = det.detect_onset(run)
    assert onset == -1 or 0 <= onset < 100
```

---

## Testing & Validation

### Run Family Tests
```bash
pytest tests/test_v3/test_my_family.py -v
```

### Run Detector Tests
```bash
pytest tests/test_detectors/test_lX_my_detector.py -v
```

### Full Suite (before PR)
```bash
pytest tests/ -v --tb=short
```

### Manual Validation Checklist
- [ ] Family generates deterministic pairs (same seed = same run)
- [ ] Proxy totals are similar between variants (±20%)
- [ ] True rewards diverge (hacking < legitimate)
- [ ] Behavioral feature has correct sign (anti-symmetric)
- [ ] Difficulty knob moves L2 smoothly
- [ ] Detector imports without errors
- [ ] Detector classify() returns [0, 1]
- [ ] Detector handles short runs gracefully
- [ ] All tests pass

---

## Submission Workflow

### 1. Fork & Branch
```bash
git clone https://github.com/yourusername/rhob.git
cd rhob
git checkout -b family/my_family  # or detector/my_detector
```

### 2. Implement & Test
- Add code
- Add tests
- Run full suite
- Update docs if needed

### 3. Pre-Submission Checklist
- [ ] Code follows style guide (see below)
- [ ] All tests pass
- [ ] No warnings from mypy/pylint
- [ ] Docstrings complete
- [ ] No hardcoded seeds (use `seed` parameter)
- [ ] No external dependencies (numpy/scipy only)
- [ ] Deterministic (no random state leakage)

### 4. Commit
```bash
git add src/rhob/v3/families/my_family.py tests/test_v3/test_my_family.py
git commit -m "Add Family: My Family

- Symmetry σ: [describe]
- Proxy: [describe]
- Feature: [describe]
- Difficulty knob: [describe]

Passes all tests and anti-symmetry screening.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 5. Push & Open PR
```bash
git push origin family/my_family
gh pr create --title "Add Family: My Family" --body "Description..."
```

### 6. Review
- Community reviews design and code
- Feedback on anti-symmetry, difficulty knob, tests
- Iterate if needed
- Merge once approved

---

## Code Style

### Python
- **Format**: Black (line length 100)
- **Typing**: Full type hints on all functions
- **Docstrings**: Google style, one-liner + details
- **No**: tabs, 2-space indent, star imports

### Naming
- **Families**: `snake_case`, max 32 chars, e.g., `novelty_farming`
- **Detectors**: `PascalCase`, max 40 chars (display name), e.g., `AngularMomentumDetector`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

### Example
```python
def _compute_feature(
    observations: np.ndarray,
    window_size: int = 50,
) -> float:
    """Compute discriminating feature from observations.
    
    Args:
        observations: Array of observations [n_steps, n_features]
        window_size: Window size for rolling computation
    
    Returns:
        Scalar feature value in [-1, 1]
    """
    if len(observations) < window_size:
        return 0.0
    
    late = observations[-window_size:]
    return float(np.mean(late))
```

---

## Questions?

- **Design help**: Start a GitHub Discussion (tagged `design-help`)
- **Code help**: Comment on your draft PR
- **Methodology questions**: See `/docs/families_5_8_redesign.md`

Welcome to RHOB! 🎯
