# Detector Templates & Reference Implementations

This document provides ready-to-use templates for building new detectors. Copy and adapt these for your use case.

## L0 Detector Templates (Reward-only)

### Template 1: Statistical Window Comparison
```python
"""L0 detector: Compare late window to baseline."""

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class WindowComparisonDetector(PosthocDetector):
    """Generic template for reward-signal comparisons."""
    
    def __init__(self, baseline_window: int = 50, test_window: int = 50, stat_fn=None):
        self.baseline_window = baseline_window
        self.test_window = test_window
        self.stat_fn = stat_fn or (lambda x: np.mean(x))
    
    @property
    def access_level(self) -> str:
        return "L0"
    
    @property
    def name(self) -> str:
        return "Window Comparison"
    
    def classify(self, run: RunData) -> float:
        rewards = run.proxy_rewards
        if len(rewards) < self.baseline_window + self.test_window:
            return 0.5
        
        baseline_stat = self.stat_fn(rewards[:self.baseline_window])
        test_stat = self.stat_fn(rewards[-self.test_window:])
        
        # Map difference to [0, 1]
        diff = test_stat - baseline_stat
        score = 1.0 / (1.0 + np.exp(-diff * 5))
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        rewards = run.proxy_rewards
        if len(rewards) < self.baseline_window:
            return -1
        
        baseline_stat = self.stat_fn(rewards[:self.baseline_window])
        
        for t in range(self.baseline_window, len(rewards)):
            window = rewards[max(0, t - self.test_window):t + 1]
            if self.stat_fn(window) - baseline_stat > 0.5:
                return t
        
        return -1
```

**Usage:**
```python
# Variance-based detector
det = WindowComparisonDetector(stat_fn=lambda x: np.var(x))

# Median-based detector
det = WindowComparisonDetector(stat_fn=lambda x: np.median(x))

# Max-based detector (peak detection)
det = WindowComparisonDetector(stat_fn=lambda x: np.max(x))
```

### Template 2: Rolling Statistics
```python
"""L0 detector: Detect changes in rolling statistics."""

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class RollingStatisticsDetector(PosthocDetector):
    """Detect anomalies in rolling windows."""
    
    def __init__(self, window_size: int = 20, stat_fn=None):
        self.window_size = window_size
        self.stat_fn = stat_fn or (lambda x: np.std(x))
    
    @property
    def access_level(self) -> str:
        return "L0"
    
    @property
    def name(self) -> str:
        return "Rolling Statistics"
    
    def classify(self, run: RunData) -> float:
        rewards = run.proxy_rewards
        if len(rewards) < 2 * self.window_size:
            return 0.5
        
        # Compute rolling stats
        rolling_stats = [
            self.stat_fn(rewards[max(0, i - self.window_size):i + 1])
            for i in range(len(rewards))
        ]
        
        # Compare early vs late
        early_mean = np.mean(rolling_stats[:len(rolling_stats)//2])
        late_mean = np.mean(rolling_stats[len(rolling_stats)//2:])
        
        diff = late_mean - early_mean
        score = 1.0 / (1.0 + np.exp(-diff * 5))
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        # Similar to classify: find first episode where anomaly appears
        rewards = run.proxy_rewards
        rolling_stats = [
            self.stat_fn(rewards[max(0, i - self.window_size):i + 1])
            for i in range(len(rewards))
        ]
        
        baseline = np.mean(rolling_stats[:self.window_size])
        for t in range(self.window_size, len(rolling_stats)):
            if rolling_stats[t] - baseline > 0.5:
                return t
        return -1
```

---

## L1 Detector Templates (State-visitation)

### Template 1: Occupancy-based Detection
```python
"""L1 detector: Analyze state occupancy patterns."""

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class OccupancyDetector(PosthocDetector):
    """Detect hacking via occupancy concentration."""
    
    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window
    
    @property
    def access_level(self) -> str:
        return "L1"
    
    @property
    def name(self) -> str:
        return "Occupancy Detector"
    
    def classify(self, run: RunData) -> float:
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window + self.test_window:
            return 0.5
        
        # Concentration: how few states dominate?
        def concentration(freq):
            total = np.sum(freq)
            return np.sum((freq / total) ** 2) if total > 0 else 0.0
        
        baseline_conc = concentration(np.sum(counts[:self.baseline_window], axis=0))
        test_conc = concentration(np.sum(counts[-self.test_window:], axis=0))
        
        diff = test_conc - baseline_conc
        score = 1.0 / (1.0 + np.exp(-diff * 5))
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window:
            return -1
        
        def concentration(freq):
            total = np.sum(freq)
            return np.sum((freq / total) ** 2) if total > 0 else 0.0
        
        baseline_conc = concentration(np.sum(counts[:self.baseline_window], axis=0))
        
        for t in range(self.baseline_window, len(counts)):
            window_counts = counts[max(0, t - self.test_window):t + 1]
            conc = concentration(np.sum(window_counts, axis=0))
            if conc - baseline_conc > 0.1:
                return t
        
        return -1
```

### Template 2: Entropy-based Detection
```python
"""L1 detector: Detect entropy shifts in visitation."""

import numpy as np
from scipy.stats import entropy
from rhob.detectors.posthoc import PosthocDetector, RunData


class EntropyDetector(PosthocDetector):
    """Detect hacking via state-visitation entropy drop."""
    
    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window
    
    @property
    def access_level(self) -> str:
        return "L1"
    
    @property
    def name(self) -> str:
        return "Entropy Detector"
    
    def classify(self, run: RunData) -> float:
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window + self.test_window:
            return 0.5
        
        baseline_hist = np.sum(counts[:self.baseline_window], axis=0)
        test_hist = np.sum(counts[-self.test_window:], axis=0)
        
        baseline_ent = entropy(baseline_hist + 1e-10)
        test_ent = entropy(test_hist + 1e-10)
        
        # Lower entropy = more concentrated = hacking
        diff = baseline_ent - test_ent
        score = 1.0 / (1.0 + np.exp(-diff * 5))
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window:
            return -1
        
        baseline_hist = np.sum(counts[:self.baseline_window], axis=0)
        baseline_ent = entropy(baseline_hist + 1e-10)
        
        for t in range(self.baseline_window, len(counts)):
            window_counts = counts[max(0, t - self.test_window):t + 1]
            window_hist = np.sum(window_counts, axis=0)
            window_ent = entropy(window_hist + 1e-10)
            if baseline_ent - window_ent > 0.5:
                return t
        
        return -1
```

---

## L2 Detector Templates (Behavioral)

### Template 1: Feature-based Classification
```python
"""L2 detector: Classify via behavioral feature."""

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class FeatureClassifier(PosthocDetector):
    """Detect via mean behavioral feature."""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    @property
    def access_level(self) -> str:
        return "L2"
    
    @property
    def name(self) -> str:
        return "Feature Classifier"
    
    def classify(self, run: RunData) -> float:
        if run.behav_trace is None:
            return 0.5
        
        # Feature should be anti-symmetric: hacking positive, legit negative (or vice versa)
        mean_feature = np.mean(run.behav_trace[-50:] if len(run.behav_trace) >= 50 else run.behav_trace)
        
        # Map signed feature to [0, 1]
        score = 0.5 + 0.5 * np.tanh(mean_feature)
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        if run.behav_trace is None:
            return -1
        
        for t in range(10, len(run.behav_trace)):
            if np.abs(run.behav_trace[t]) > self.threshold:
                return t
        
        return -1
```

### Template 2: Feature Drift Detection
```python
"""L2 detector: Detect feature drift over time."""

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class FeatureDriftDetector(PosthocDetector):
    """Detect hacking via behavioral feature drift."""
    
    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window
    
    @property
    def access_level(self) -> str:
        return "L2"
    
    @property
    def name(self) -> str:
        return "Feature Drift"
    
    def classify(self, run: RunData) -> float:
        if run.behav_trace is None:
            return 0.5
        
        if len(run.behav_trace) < self.baseline_window + self.test_window:
            return 0.5
        
        baseline_mean = np.mean(run.behav_trace[:self.baseline_window])
        test_mean = np.mean(run.behav_trace[-self.test_window:])
        
        # Hacking has positive drift (or negative, depending on design)
        drift = test_mean - baseline_mean
        score = 1.0 / (1.0 + np.exp(-drift * 5))
        return float(np.clip(score, 0.0, 1.0))
    
    def detect_onset(self, run: RunData) -> int:
        if run.behav_trace is None:
            return -1
        
        if len(run.behav_trace) < self.baseline_window:
            return -1
        
        baseline_mean = np.mean(run.behav_trace[:self.baseline_window])
        
        for t in range(self.baseline_window, len(run.behav_trace)):
            window = run.behav_trace[max(0, t - self.test_window):t + 1]
            if np.mean(window) - baseline_mean > 0.5:
                return t
        
        return -1
```

---

## How to Use These Templates

1. **Copy the template** that matches your detector type (L0/L1/L2)
2. **Customize the computation**: Replace `stat_fn`, `concentration`, `mean_feature`, etc. with your logic
3. **Test it**: Create a test file following the CONTRIBUTING guide
4. **Submit**: Open a PR with your new detector

## Next Steps

After creating your detector:
- Add to `src/rhob/detectors/__init__.py` __all__
- Create corresponding test file
- Run `pytest tests/test_detectors/test_lX_your_detector.py -v`
- Submit PR with link to design discussion (if applicable)

---

## Common Patterns

| Pattern | Use Case | Template |
|---------|----------|----------|
| **Window comparison** | Detect reward/entropy shift | Template 1 (L0) |
| **Rolling stats** | Anomaly detection | Template 2 (L0) |
| **Occupancy** | State concentration | Template 1 (L1) |
| **Entropy** | Visitation diversity | Template 2 (L1) |
| **Feature drift** | Behavioral shift | Template 1 or 2 (L2) |

Pick the one that fits your idea, adapt, and contribute!

