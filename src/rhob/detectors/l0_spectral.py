"""Spectral (FFT-based) reward detector (L0, simple baseline)."""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class SpectralRewardDetector(PosthocDetector):
    """Flags a shift in the reward series' frequency content (spectral entropy).

    Computes the power spectrum of the (detrended) proxy-reward series in a trailing
    window and scores by its spectral entropy relative to an early-window baseline: a
    reward series that becomes more periodic or more erratic shows up as a spectral
    entropy shift, independent of the series' mean or absolute scale.
    """

    def __init__(self, baseline_episodes: int = 20, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Spectral Reward"

    def classify(self, run: RunData) -> float:
        """Score: |spectral entropy shift| between the early and late windows."""
        proxy = run.proxy_rewards
        if len(proxy) < self.baseline_episodes + 4:
            return 0.0
        early = proxy[: self.baseline_episodes]
        late = proxy[-self.late_window :] if len(proxy) >= self.late_window else proxy
        return abs(self._spectral_entropy(late) - self._spectral_entropy(early))

    def detect_onset(self, run: RunData) -> int:
        """Detect the first sustained departure of rolling spectral entropy from baseline."""
        proxy = run.proxy_rewards
        n = len(proxy)
        w = max(8, self.baseline_episodes // 2)
        if n < self.baseline_episodes + 2 * w:
            return -1
        early_entropy = self._spectral_entropy(proxy[: self.baseline_episodes])

        run_len = 0
        for t in range(self.baseline_episodes, n - w):
            window_entropy = self._spectral_entropy(proxy[t : t + w])
            if abs(window_entropy - early_entropy) > 0.5:
                run_len += 1
                if run_len >= 3:
                    return t - 2
            else:
                run_len = 0
        return -1

    @staticmethod
    def _spectral_entropy(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 4 or float(x.std()) < 1e-9:
            return 0.0
        x = x - x.mean()
        power = np.abs(np.fft.rfft(x)) ** 2
        power = power[1:]  # drop the DC component
        total = power.sum()
        if total <= 0:
            return 0.0
        p = power / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)) / np.log(len(p) + 1e-9)) if len(p) > 1 else 0.0
