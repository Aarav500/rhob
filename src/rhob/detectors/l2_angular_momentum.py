"""Angular Momentum Detector: L2 detector specific to Orbit Chirality (Family 8).

Detects chirality commitment via signed angular momentum in behavioral trace.
High positive/negative values indicate strong CW/CCW commitment (anti-symmetric).
"""

from __future__ import annotations

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class AngularMomentumDetector(PosthocDetector):
    """Detect chirality via angular momentum (Family 8 specific)."""

    def __init__(self):
        pass

    @property
    def access_level(self) -> str:
        return "L2"

    @property
    def name(self) -> str:
        return "Angular Momentum"

    def classify(self, run: RunData) -> float:
        """Score: absolute angular momentum (higher = more committed to chirality)."""
        if run.behav_trace is None:
            return 0.5

        # Mean absolute angular momentum in late window
        mean_am = np.abs(np.mean(run.behav_trace[-50:] if len(run.behav_trace) >= 50 else run.behav_trace))
        score = min(1.0, mean_am)
        return float(score)

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as first episode where angular momentum becomes significant."""
        if run.behav_trace is None:
            return -1

        for t in range(10, len(run.behav_trace)):
            if np.abs(run.behav_trace[t]) > 0.3:
                return t

        return -1
