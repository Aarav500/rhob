"""Visitation-entropy-trend detector (L1, simple baseline)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData


class VisitationEntropyTrendDetector(PosthocDetector):
    """Nearest-centroid classifier over the *trend* in visitation entropy.

    Concentrating on a single state (camping) lowers visitation entropy relative to
    early, unfocused exploration; spreading out (roaming/novelty-farming) raises it.
    The raw entropy trend (a magnitude) does not by itself say *which* variant a run
    is, so -- like state divergence -- this detector must be fit on labeled runs: it
    learns each variant's typical (baseline, late) entropy pair and classifies by
    nearest centroid.
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
        return "Visitation Entropy Trend"

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
        """Detect the first sustained departure of rolling entropy from its own baseline."""
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.baseline_episodes + 3:
            return -1
        baseline = float(np.mean([self._entropy(counts[i]) for i in range(self.baseline_episodes)]))
        run_len = 0
        for t in range(self.baseline_episodes, counts.shape[0]):
            e = self._entropy(counts[t])
            if abs(e - baseline) > 0.3:
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
        baseline = float(np.mean([self._entropy(counts[i]) for i in range(self.baseline_episodes)]))
        window = counts[-self.late_window :] if counts.shape[0] >= self.late_window else counts
        late = float(np.mean([self._entropy(row) for row in window]))
        return np.array([baseline, late, late - baseline])

    @staticmethod
    def _entropy(counts_row: np.ndarray) -> float:
        total = counts_row.sum()
        if total <= 0:
            return 0.0
        p = counts_row / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))
