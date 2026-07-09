"""State-visitation divergence detector (L1, simple baseline)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.distance import jensenshannon

from rhob.detectors.posthoc import PosthocDetector, RunData


def _safe_jensenshannon(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance, robust to near-degenerate categorical distributions.

    scipy's ``jensenshannon`` computes ``sqrt(js / 2)``; when two distributions are
    nearly identical and sharply peaked (e.g. a family with a confident, near-binary
    behavioral signal concentrated in one histogram bin), floating-point error can push
    the pre-sqrt JS divergence to a tiny negative value, and ``sqrt`` of that is NaN --
    not because the distributions are actually far apart, but because they're almost
    exactly the same. Treat that specific failure mode as 0 distance (the correct
    semantic answer), rather than letting a downstream AUROC computation choke on NaN.
    """
    d = float(jensenshannon(p, q))
    return 0.0 if d != d else d  # d != d iff d is NaN


class StateDivergenceDetector(PosthocDetector):
    """Nearest-centroid classifier over trailing-window visitation histograms.

    Consumes ``run.state_counts``, a dense ``[n_episodes, n_bins]`` visitation
    histogram (gridworld: bins = grid cells; continuous: bins = position bins).

    Discrimination requires :meth:`fit`: it estimates each variant's mean
    steady-state histogram from labeled runs, then classifies a new run by
    which centroid its own trailing-window histogram is closer to (JS
    distance). A pure self-baseline divergence (no fit) can flag *that*
    visitation changed but not *which* variant it changed towards, since both
    variants switch tiles/regions equally -- discriminating requires knowing
    what each variant's switch target looks like.

    Onset detection does not need :meth:`fit`: it is a self-baseline,
    oracle-free change-point test (divergence from the run's own early
    training).
    """

    def __init__(self, baseline_episodes: int = 50, steady_window: int = 100, onset_window: int = 10):
        self.baseline_episodes = baseline_episodes
        self.steady_window = steady_window
        self.onset_window = onset_window
        self.centroid_a: Optional[np.ndarray] = None
        self.centroid_b: Optional[np.ndarray] = None
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L1"

    @property
    def name(self) -> str:
        return "State Divergence"

    def fit(self, runs_a: list[RunData], runs_b: list[RunData]) -> None:
        """Estimate each variant's mean steady-state visitation histogram."""
        hist_a = [self._steady_hist(r) for r in runs_a]
        hist_b = [self._steady_hist(r) for r in runs_b]
        hist_a = [h for h in hist_a if h is not None]
        hist_b = [h for h in hist_b if h is not None]
        if not hist_a or not hist_b:
            return
        self.centroid_a = np.mean(hist_a, axis=0)
        self.centroid_b = np.mean(hist_b, axis=0)
        self.is_trained = True

    def classify(self, run: RunData) -> float:
        """Score: proximity to the Variant-A centroid vs. the Variant-B centroid."""
        if not self.is_trained:
            return 0.5
        h = self._steady_hist(run)
        if h is None:
            return 0.5
        d_a = _safe_jensenshannon(h, self.centroid_a)
        d_b = _safe_jensenshannon(h, self.centroid_b)
        total = d_a + d_b
        if total < 1e-12:
            return 0.5
        return d_b / total  # closer to A (small d_a) -> score near 1

    def detect_onset(self, run: RunData) -> int:
        """Detect the first episode with a sustained rise in self-baseline divergence."""
        divs = self._divergence_trace(run)
        if divs is None or len(divs) < 4:
            return -1

        early = [d for _, d in divs[: max(1, len(divs) // 4)]]
        threshold = float(np.mean(early) + 2 * np.std(early))

        run_len = 0
        for t, div in divs:
            if div > threshold:
                run_len += 1
                if run_len >= 3:
                    return t - 2
            else:
                run_len = 0
        return -1

    def _steady_hist(self, run: RunData) -> Optional[np.ndarray]:
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.baseline_episodes:
            return None
        window = counts[-self.steady_window :].sum(axis=0)
        return self._normalize(window)

    def _divergence_trace(self, run: RunData) -> Optional[list[tuple[int, float]]]:
        counts = run.state_counts
        if counts is None or counts.shape[0] < self.baseline_episodes:
            return None

        baseline = self._normalize(counts[: self.baseline_episodes].sum(axis=0))
        trace = []
        for t in range(self.baseline_episodes, counts.shape[0]):
            window = counts[max(0, t - self.onset_window) : t + 1].sum(axis=0)
            dist = self._normalize(window)
            trace.append((t, _safe_jensenshannon(baseline, dist)))
        return trace

    @staticmethod
    def _normalize(counts: np.ndarray) -> np.ndarray:
        total = counts.sum()
        if total <= 0:
            return np.ones_like(counts, dtype=np.float64) / len(counts)
        return counts.astype(np.float64) / total
