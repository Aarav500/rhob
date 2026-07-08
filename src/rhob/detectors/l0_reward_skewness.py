"""Skewness: L0 detector for asymmetry in reward distribution.

Hacking often produces skewed reward distributions (e.g., frequent zeros + occasional peaks).
Legitimate trajectories have more symmetric distributions. Detects via distribution skewness.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from rhob.detectors.posthoc import PosthocDetector, RunData


class RewardSkewnessDetector(PosthocDetector):
    """Detect via skewness in reward distribution."""

    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward Skewness"

    def classify(self, run: RunData) -> float:
        """Score: difference in skewness between late and early windows."""
        rewards = run.proxy_rewards
        if len(rewards) < self.baseline_window + self.test_window:
            return 0.5

        baseline_skew = float(stats.skew(rewards[:self.baseline_window]))
        test_skew = float(stats.skew(rewards[-self.test_window:]))

        # Hacking has higher positive skew (occasional spikes)
        diff = test_skew - baseline_skew
        score = 1.0 / (1.0 + np.exp(-diff))  # Sigmoid
        return float(np.clip(score, 0.0, 1.0))

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as first episode where skewness jumps."""
        rewards = run.proxy_rewards
        if len(rewards) < self.baseline_window:
            return -1

        baseline_skew = float(stats.skew(rewards[:self.baseline_window]))

        for t in range(self.baseline_window, len(rewards)):
            window = rewards[max(0, t - self.test_window): t + 1]
            test_skew = float(stats.skew(window))
            if test_skew - baseline_skew > 0.5:  # Significant jump
                return t

        return -1
