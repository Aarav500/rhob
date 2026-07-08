"""Reward kernel-density outlier detector (L0, simple baseline)."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

from rhob.detectors.posthoc import PosthocDetector, RunData


class RewardKDEDetector(PosthocDetector):
    """Flags late-training reward that looks unlikely under the early-training distribution.

    Fits a 1-D kernel density estimate on the early-window proxy rewards, then scores
    a run by the mean negative log-likelihood of its late-window rewards under that
    density: a large value means the late reward looks nothing like the early reward,
    i.e. an outlier relative to the run's own baseline (self-calibrating; no absolute
    reward-scale assumption).
    """

    def __init__(self, baseline_episodes: int = 20, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward KDE"

    def classify(self, run: RunData) -> float:
        """Score: mean negative log-density of late rewards under the early KDE."""
        proxy = run.proxy_rewards
        if len(proxy) < self.baseline_episodes + 2:
            return 0.0
        early = proxy[: self.baseline_episodes]
        late = proxy[-self.late_window :] if len(proxy) >= self.late_window else proxy
        if float(early.std()) < 1e-9:
            early = early + np.linspace(-1e-6, 1e-6, len(early))
        kde = gaussian_kde(early)
        density = np.clip(kde(late), 1e-300, None)
        return float(-np.mean(np.log(density)))

    def detect_onset(self, run: RunData) -> int:
        """Detect the first sustained rise in outlier score above the early baseline."""
        proxy = run.proxy_rewards
        n = len(proxy)
        w = max(3, self.baseline_episodes // 4)
        if n < self.baseline_episodes + 2 * w:
            return -1
        early = proxy[: self.baseline_episodes]
        if float(early.std()) < 1e-9:
            early = early + np.linspace(-1e-6, 1e-6, len(early))
        kde = gaussian_kde(early)

        baseline_scores = -np.log(np.clip(kde(early), 1e-300, None))
        threshold = float(baseline_scores.mean() + 3 * baseline_scores.std() + 1e-6)

        run_len = 0
        for t in range(self.baseline_episodes, n - w):
            window = proxy[t : t + w]
            score = float(-np.mean(np.log(np.clip(kde(window), 1e-300, None))))
            if score > threshold:
                run_len += 1
                if run_len >= 3:
                    return t - 2
            else:
                run_len = 0
        return -1
