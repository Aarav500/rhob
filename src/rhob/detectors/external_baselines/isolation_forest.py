"""Isolation Forest (Liu, Ting & Zhou, 2008) -- unsupervised anomaly detection (L2)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from rhob.detectors.posthoc import PosthocDetector, RunData

#: The late window's mean, standard deviation, range and last-minus-first need two points.
MIN_EPISODES = 2


class IsolationForestDetector(PosthocDetector):
    """Unsupervised anomaly score over windowed behavioral-trace statistics.

    Fits an Isolation Forest on the legitimate-variant's feature distribution
    only (the classical unsupervised-anomaly-detection setting: model
    "normal", flag deviations) -- it never sees the hacking variant during
    fitting, unlike the supervised L2 baselines. This is the standard
    scikit-learn implementation of Liu et al. (2008), unmodified.
    """

    def __init__(self, window: int = 100, n_estimators: int = 100, random_state: int = 0):
        self.window = window
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L2"

    @property
    def name(self) -> str:
        return "Isolation Forest"

    @property
    def min_episodes(self) -> int:
        """Two episodes of ``behav_trace``: a standard deviation and a range need two points.

        Below that ``_extract_features`` returns None for every run, ``fit`` trains nothing
        and ``classify`` returns 0.5. Above it the late window shrinks to the whole run
        when the run is shorter than ``window``, so no registered family is short enough
        for this to bind. (``window=1`` keeps the late window at one point at every length,
        so that detector is constant however long the run; nothing in the repository builds
        it.)
        """
        return MIN_EPISODES

    def fit(self, runs_a: list[RunData], runs_b: list[RunData]) -> None:
        """Fit on the legitimate variant only (runs_b), per the anomaly-detection setting."""
        feats = [self._extract_features(r) for r in runs_b]
        feats = [f for f in feats if f is not None]
        if not feats:
            return
        X = np.array(feats)
        self.model = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model.fit(X)
        self.is_trained = True

    def classify(self, run: RunData) -> float:
        if not self.is_trained or self.model is None:
            return 0.5
        feat = self._extract_features(run)
        if feat is None:
            return 0.5
        # decision_function: higher = more normal. Flip + squash to [0, 1].
        raw = float(self.model.decision_function(feat.reshape(1, -1))[0])
        return float(1.0 / (1.0 + np.exp(4.0 * raw)))

    def detect_onset(self, run: RunData) -> int:
        if not self.is_trained or self.model is None:
            return -1
        trace = run.behav_trace
        if trace is None or len(trace) < self.window + 3:
            return -1
        scores = []
        for t in range(self.window, len(trace)):
            partial = RunData(
                proxy_rewards=run.proxy_rewards[:t],
                true_rewards=run.true_rewards[:t],
                state_counts=None,
                behav_trace=trace[:t],
            )
            feat = self._extract_features(partial)
            if feat is None:
                continue
            raw = float(self.model.decision_function(feat.reshape(1, -1))[0])
            scores.append((t, raw))
        anomalous = [t for t, raw in scores if raw < 0]
        return anomalous[0] if anomalous else -1

    def _extract_features(self, run: RunData) -> Optional[np.ndarray]:
        if run.behav_trace is None or len(run.behav_trace) < self.min_episodes:
            return None
        w = self.window
        late = run.behav_trace[-w:] if len(run.behav_trace) >= w else run.behav_trace
        # The module constant, not ``self.min_episodes``: this guard is about the late
        # window (``window=1`` keeps it at one point), and the window's standard
        # deviation and range need two points whatever length a subclass declares.
        if len(late) < MIN_EPISODES:
            return None
        return np.array(
            [
                float(np.mean(late)),
                float(np.std(late)),
                float(np.max(late) - np.min(late)),
                float(late[-1] - late[0]),
            ]
        )
