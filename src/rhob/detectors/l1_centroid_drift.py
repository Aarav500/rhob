"""Centroid Drift: L1 detector for spatial drift in state visitation centroid.

Hacking often drifts spatially toward high-value regions; legitimate exploration
returns toward center. Detects via displacement of visitation centroid over time.
"""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class CentroidDriftDetector(PosthocDetector):
    """Detect via spatial drift in visitation centroid."""

    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "Centroid Drift"

    def classify(self, run: RunData) -> float:
        """Score: drift magnitude in visitation centroid."""
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window + self.test_window:
            return 0.5

        def centroid(freq: np.ndarray) -> float:
            """Compute weighted centroid (assume 1D binning)."""
            total = np.sum(freq)
            if total == 0:
                return 0.0
            weights = np.arange(len(freq), dtype=float)
            return np.sum(weights * freq) / total

        baseline_centroid = centroid(np.sum(counts[:self.baseline_window], axis=0))
        test_centroid = centroid(np.sum(counts[-self.test_window:], axis=0))

        # Hacking has large centroid displacement
        drift = np.abs(test_centroid - baseline_centroid)
        score = 1.0 / (1.0 + np.exp(-drift))
        return float(np.clip(score, 0.0, 1.0))

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as first episode where centroid drifts."""
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window:
            return -1

        def centroid(freq: np.ndarray) -> float:
            total = np.sum(freq)
            if total == 0:
                return 0.0
            weights = np.arange(len(freq), dtype=float)
            return np.sum(weights * freq) / total

        baseline_centroid = centroid(np.sum(counts[:self.baseline_window], axis=0))

        for t in range(self.baseline_window, len(counts)):
            window_counts = counts[max(0, t - self.test_window): t + 1]
            c = centroid(np.sum(window_counts, axis=0))
            if np.abs(c - baseline_centroid) > 5.0:
                return t

        return -1
