"""CUSUM change-point detector (L0, simple baseline)."""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector, RunData


class RewardCUSUMDetector(PosthocDetector):
    """Cumulative sum (CUSUM) change-point detection on reward signal.

    Detects abrupt shifts in reward level. ``drift``/``threshold`` are in units
    of the run's own early-training reward standard deviation (self-
    calibrating), so the detector applies unchanged across pairs whose proxy
    magnitudes differ by orders of magnitude. Expected to fail to discriminate
    on admission pairs because both variants have identical reward transitions
    (matched proxy).
    """

    def __init__(self, drift_std: float = 0.5, threshold_std: float = 5.0, baseline_window: int = 50):
        self.drift_std = drift_std
        self.threshold_std = threshold_std
        self.baseline_window = baseline_window

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward CUSUM"

    def classify(self, run: RunData) -> float:
        """Score: maximum absolute standardized CUSUM statistic."""
        max_stat, _ = self._cusum_trace(run)
        return max_stat

    def detect_onset(self, run: RunData) -> int:
        """Detect first episode where the standardized CUSUM statistic exceeds threshold_std."""
        _, onset = self._cusum_trace(run)
        return onset

    def _cusum_trace(self, run: RunData) -> tuple[float, int]:
        proxy = run.proxy_rewards
        if len(proxy) < self.baseline_window:
            return 0.0, -1
        mean_est = float(proxy[: self.baseline_window].mean())
        std_est = float(proxy[: self.baseline_window].std()) + 1e-6
        drift = self.drift_std * std_est
        threshold = self.threshold_std * std_est

        s_pos = s_neg = max_stat = 0.0
        onset = -1
        for t in range(self.baseline_window, len(proxy)):
            r = proxy[t]
            s_pos = max(0.0, s_pos + (r - mean_est) - drift)
            s_neg = max(0.0, s_neg - (r - mean_est) - drift)
            stat = max(s_pos, s_neg)
            max_stat = max(max_stat, stat)
            if onset < 0 and stat > threshold:
                onset = t
        return max_stat / self.threshold_std, onset
