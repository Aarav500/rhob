"""State-coverage-rate detector (L1, simple baseline)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class StateCoverageRateDetector(PosthocDetector):
    """Nearest-centroid classifier over the rate of new-state discovery.

    A camping strategy stops discovering new bins once it settles; a roaming
    (novelty-farming) strategy keeps discovering new bins throughout. The raw
    coverage-decay rate is a magnitude feature that does not say *which* variant a
    run is on its own, so this detector is fit on labeled runs (nearest centroid over
    the early/late coverage-rate pair), like state divergence and entropy trend.
    """

    def __init__(self, baseline_episodes: int = 50, late_window: int = 100):
        self.baseline_episodes = baseline_episodes
        self.late_window = late_window
        self.centroid_a: Optional[np.ndarray] = None
        self.centroid_b: Optional[np.ndarray] = None
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "State Coverage Rate"

    def fit(self, runs_a: list[RunData], runs_b: list[RunData]) -> None:
        feats_a = [f for f in (self._features(r) for r in runs_a) if f is not None]
        feats_b = [f for f in (self._features(r) for r in runs_b) if f is not None]
        if not feats_a or not feats_b:
            return
        self.centroid_a = np.mean(feats_a, axis=0)
        self.centroid_b = np.mean(feats_b, axis=0)
        self.is_trained = True

    def classify(self, run: RunData) -> float:
        if not self.is_trained:
            return 0.5
        feat = self._features(run)
        if feat is None:
            return 0.5
        d_a = float(np.linalg.norm(feat - self.centroid_a))
        d_b = float(np.linalg.norm(feat - self.centroid_b))
        total = d_a + d_b
        return 0.5 if total < 1e-12 else d_b / total

    def detect_onset(self, run: RunData) -> int:
        """Detect the first sustained drop in new-bin discovery rate below the early rate."""
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.baseline_episodes + 3:
            return -1
        rate = self._coverage_rate_trace(counts)
        baseline = float(np.mean(rate[: self.baseline_episodes]))
        run_len = 0
        for t in range(self.baseline_episodes, len(rate)):
            if abs(rate[t] - baseline) > 0.5 * max(baseline, 1e-6):
                run_len += 1
                if run_len >= 3:
                    return t - 2
            else:
                run_len = 0
        return -1

    def _features(self, run: RunData) -> Optional[np.ndarray]:
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.baseline_episodes:
            return None
        rate = self._coverage_rate_trace(counts)
        baseline = float(np.mean(rate[: self.baseline_episodes]))
        window = rate[-self.late_window :] if len(rate) >= self.late_window else rate
        late = float(np.mean(window))
        return np.array([baseline, late])

    @staticmethod
    def _coverage_rate_trace(counts: np.ndarray) -> np.ndarray:
        """Fraction of bins newly visited (first non-zero count) at each episode."""
        seen = np.zeros(counts.shape[1], dtype=bool)
        rate = np.zeros(counts.shape[0])
        for t in range(counts.shape[0]):
            visited = counts[t] > 0
            new = visited & ~seen
            rate[t] = new.sum() / counts.shape[1]
            seen |= visited
        return rate
