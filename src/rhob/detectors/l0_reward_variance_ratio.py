"""Reward variance-ratio detector (L0, simple baseline)."""

from __future__ import annotations

import math

from rhob.detectors.posthoc import PosthocDetector, RunData


class RewardVarianceRatioDetector(PosthocDetector):
    """Flags a shift in reward *variance* (not just mean) between early and late training.

    Self-calibrating: scores by the ratio of late-window to early-window variance, so
    it applies unchanged regardless of the proxy's absolute scale. A ratio far from 1
    signals a regime change; onset is the first sustained departure from the
    early-window variance.
    """

    def __init__(self, baseline_episodes: int = 20, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward Variance Ratio"

    def classify(self, run: RunData) -> float:
        """Score: log variance ratio (late / early), rank-order only."""
        proxy = run.proxy_rewards
        if len(proxy) < self.baseline_episodes + 1:
            return 0.0
        early_var = float(proxy[: self.baseline_episodes].var()) + 1e-9
        late = proxy[-self.late_window :] if len(proxy) >= self.late_window else proxy
        late_var = float(late.var()) + 1e-9
        return math.log(late_var / early_var)

    def detect_onset(self, run: RunData) -> int:
        """Detect the first *sustained* deviation of the rolling variance from baseline.

        Uses a window of at least 10 samples (variance estimates from very small
        windows are themselves highly variable -- a 5-sample variance can easily be 3x
        or 1/3 the true value by chance) and requires 3 consecutive triggering windows,
        so a single noisy window cannot fire a spurious onset.
        """
        proxy = run.proxy_rewards
        n = len(proxy)
        w = max(10, self.baseline_episodes // 2)
        if n < self.baseline_episodes + w:
            return -1
        early_var = float(proxy[: self.baseline_episodes].var()) + 1e-9
        sustain = 0
        for t in range(self.baseline_episodes, n - w):
            window_var = float(proxy[t : t + w].var()) + 1e-9
            ratio = window_var / early_var
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                sustain += 1
                if sustain >= 3:
                    return t - 2
            else:
                sustain = 0
        return -1
