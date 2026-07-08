"""Occupancy Polarization: L1 detector for concentrated vs. uniform visitation patterns.

Hacking often produces bimodal or concentrated occupation (high visits to few states).
Legitimate exploration has more uniform distribution. Detects via bimodality coefficient.
"""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class OccupancyPolarizationDetector(PosthocDetector):
    """Detect via polarization (concentration vs. uniformity) of state occupation."""

    def __init__(self, baseline_window: int = 50, test_window: int = 50):
        self.baseline_window = baseline_window
        self.test_window = test_window

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "Occupancy Polarization"

    def classify(self, run: RunData) -> float:
        """Score: degree of polarization (bimodality) in state occupation."""
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window + self.test_window:
            return 0.5

        def polarization(freq: np.ndarray) -> float:
            """Compute bimodality coefficient (high = polarized, low = uniform)."""
            if len(freq) < 2:
                return 0.0
            # Simplified: ratio of max to mean
            total = np.sum(freq)
            if total == 0:
                return 0.0
            return float(np.max(freq) / (np.sum(freq) / len(freq) + 1e-10))

        baseline_polar = polarization(np.sum(counts[:self.baseline_window], axis=0))
        test_polar = polarization(np.sum(counts[-self.test_window:], axis=0))

        # Hacking has higher polarization
        diff = test_polar - baseline_polar
        score = 1.0 / (1.0 + np.exp(-diff))
        return float(np.clip(score, 0.0, 1.0))

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as first episode where polarization increases."""
        counts = run.state_counts
        if counts is None or len(counts) < self.baseline_window:
            return -1

        def polarization(freq: np.ndarray) -> float:
            if len(freq) < 2:
                return 0.0
            total = np.sum(freq)
            if total == 0:
                return 0.0
            return float(np.max(freq) / (np.sum(freq) / len(freq) + 1e-10))

        baseline_polar = polarization(np.sum(counts[:self.baseline_window], axis=0))

        for t in range(self.baseline_window, len(counts)):
            window_counts = counts[max(0, t - self.test_window): t + 1]
            p = polarization(np.sum(window_counts, axis=0))
            if p - baseline_polar > 2.0:
                return t

        return -1
