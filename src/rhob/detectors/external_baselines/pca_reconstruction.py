"""PCA reconstruction error -- classical representation-learning baseline (L1)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from rhob.detectors.posthoc import PosthocDetector, RunData


class PCAReconstructionDetector(PosthocDetector):
    """Learns a low-rank linear representation of legitimate-variant state-
    visitation histograms, then scores runs by reconstruction error.

    This is the canonical representation-learning baseline for
    distribution-shift / anomaly detection: fit an unsupervised low-dimensional
    representation of "normal" state occupancy, flag runs that don't project
    cleanly onto it. Fit on ``runs_b`` (legitimate) only, matching the
    unsupervised setting used by :class:`IsolationForestDetector`.
    """

    def __init__(self, n_components: int = 3, steady_window: int = 100):
        self.n_components = n_components
        self.steady_window = steady_window
        self.pca: Optional[PCA] = None
        self.error_scale: float = 1.0
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "PCA Reconstruction"

    def fit(self, runs_a: list[RunData], runs_b: list[RunData]) -> None:
        hists = [self._steady_hist(r) for r in runs_b]
        hists = [h for h in hists if h is not None]
        if len(hists) < 2:
            return
        X = np.array(hists)
        n_components = min(self.n_components, X.shape[0] - 1, X.shape[1])
        if n_components < 1:
            return
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X)
        recon = self.pca.inverse_transform(self.pca.transform(X))
        errors = np.linalg.norm(X - recon, axis=1)
        self.error_scale = float(np.std(errors) + 1e-6)
        self.is_trained = True

    def classify(self, run: RunData) -> float:
        if not self.is_trained or self.pca is None:
            return 0.5
        h = self._steady_hist(run)
        if h is None:
            return 0.5
        recon = self.pca.inverse_transform(self.pca.transform(h.reshape(1, -1)))[0]
        error = float(np.linalg.norm(h - recon))
        z = error / self.error_scale
        return float(1.0 / (1.0 + np.exp(-(z - 2.0))))

    def detect_onset(self, run: RunData) -> int:
        if not self.is_trained or self.pca is None:
            return -1
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.steady_window:
            return -1
        errors = []
        for t in range(self.steady_window, counts.shape[0]):
            window = counts[max(0, t - self.steady_window) : t].sum(axis=0)
            h = self._normalize(window)
            recon = self.pca.inverse_transform(self.pca.transform(h.reshape(1, -1)))[0]
            errors.append((t, float(np.linalg.norm(h - recon))))
        threshold = 3.0 * self.error_scale
        run_len = 0
        for t, err in errors:
            if err > threshold:
                run_len += 1
                if run_len >= 3:
                    return t - 2
            else:
                run_len = 0
        return -1

    def _steady_hist(self, run: RunData) -> Optional[np.ndarray]:
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.steady_window:
            return None
        window = counts[-self.steady_window :].sum(axis=0)
        return self._normalize(window)

    @staticmethod
    def _normalize(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        if total <= 0:
            return np.ones_like(counts, dtype=np.float64) / len(counts)
        return counts.astype(np.float64) / total
