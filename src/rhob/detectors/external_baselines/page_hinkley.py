"""Page-Hinkley test (Page, 1954; Hinkley, 1971) -- classical change-point detector (L0)."""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class PageHinkleyDetector(PosthocDetector):
    """Sequential change-point test on the proxy-reward stream.

    Tracks the cumulative deviation of the running mean from its historical
    minimum (or maximum); a sustained deviation beyond ``threshold`` flags a
    change point. This is the classical statistical-process-control test, not
    a method designed for RHOB -- included so the leaderboard compares
    detection approaches, not just in-house features.

    Score is the peak (normalized) Page-Hinkley statistic observed over the
    run; onset is the first index where the statistic crosses ``threshold``.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 3.0):
        self.delta = delta
        self.threshold = threshold

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Page-Hinkley Test"

    def classify(self, run: RunData) -> float:
        stat = self._ph_statistic(run.proxy_rewards)
        if stat is None or len(stat) == 0:
            return 0.5
        peak = float(np.max(stat))
        # squash to (0, 1); peak == threshold -> ~0.5
        return float(1.0 / (1.0 + np.exp(-(peak - self.threshold))))

    def detect_onset(self, run: RunData) -> int:
        stat = self._ph_statistic(run.proxy_rewards)
        if stat is None:
            return -1
        crossed = np.where(stat > self.threshold)[0]
        return int(crossed[0]) if len(crossed) else -1

    def _ph_statistic(self, rewards: np.ndarray) -> np.ndarray | None:
        if rewards is None or len(rewards) < 3:
            return None
        x = rewards.astype(np.float64)
        mean_t = np.cumsum(x) / (np.arange(len(x)) + 1)
        deviations = x - mean_t - self.delta
        cum_dev = np.cumsum(deviations)
        running_min = np.minimum.accumulate(cum_dev)
        return cum_dev - running_min
