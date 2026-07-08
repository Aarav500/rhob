"""L2 detector: Behavioral feature consistency over time."""

from __future__ import annotations

import numpy as np
from rhob.detectors.posthoc import PosthocDetector, RunData


class FeatureConsistencyDetector(PosthocDetector):
    """Detect hacking via behavioral feature consistency.

    Hacking: feature commits to one sign and stays consistent (low variance).
    Legitimate: feature remains noisy, near-zero, or mixed sign (high variance).

    Measures: std(feature) in late window vs early window.
    """

    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window

    @property
    def access_level(self) -> str:
        return "L2"

    @property
    def name(self) -> str:
        return "Feature Consistency"

    def classify(self, run: RunData) -> float:
        if run.behav_trace is None:
            return 0.5

        if len(run.behav_trace) < self.baseline_window + self.test_window:
            return 0.5

        baseline_std = np.std(run.behav_trace[: self.baseline_window])
        test_std = np.std(run.behav_trace[-self.test_window :])

        if baseline_std < 1e-6:
            return 0.5

        # Hacking: test_std < baseline_std (feature becomes consistent)
        consistency_ratio = test_std / baseline_std
        score = 1.0 / (1.0 + np.exp(-(1.0 - consistency_ratio) * 5))
        return float(np.clip(score, 0.0, 1.0))

    def detect_onset(self, run: RunData) -> int:
        if run.behav_trace is None:
            return -1

        if len(run.behav_trace) < self.baseline_window:
            return -1

        baseline_std = np.std(run.behav_trace[: self.baseline_window])

        for t in range(self.baseline_window, len(run.behav_trace)):
            window = run.behav_trace[max(0, t - self.test_window) : t + 1]
            if np.std(window) < baseline_std * 0.5:
                return t

        return -1
