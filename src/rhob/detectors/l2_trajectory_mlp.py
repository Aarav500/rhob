"""Trajectory MLP detector (L2, strongest L2 baseline)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rhob.detectors.posthoc import PosthocDetector, RunData

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

FEATURE_DIM = 6


class TrajectoryMLPNet(nn.Module):
    """MLP for trajectory feature classification."""

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class TrajectoryMLPDetector(PosthocDetector):
    """Neural network trained on behavioral trajectory features.

    Features are drawn only from behaviour (``behav_trace``) and proxy reward
    -- the same information a detector could observe in deployment.
    ``true_rewards`` is oracle-only (see :class:`~rhob.detectors.posthoc.RunData`)
    and is never used as a feature; using it would trivially "solve" detection
    by reading the answer instead of inferring it.
    """

    def __init__(self, hidden: int = 128, window: int = 100, seed: int = 0):
        if not HAS_TORCH:
            raise ImportError("torch required for TrajectoryMLPDetector")
        self.hidden = hidden
        self.window = window
        self.seed = seed
        self.model: Optional[TrajectoryMLPNet] = None
        self.is_trained = False
        self.feature_stats: Optional[dict] = None

    @property
    def access_level(self) -> str:
        return "L2"

    @property
    def name(self) -> str:
        return "Trajectory MLP"

    def fit(
        self,
        runs_a: list[RunData],
        runs_b: list[RunData],
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
    ) -> None:
        """Train on labeled runs.

        Seeds torch's global RNG before constructing/training the model:
        weight initialization and the per-epoch ``torch.randperm`` shuffle
        both draw from it, so an unseeded fit() is a different, unreproducible
        model every call. This was severe enough in practice that five
        identical fit() calls on the same data produced held-out AUROCs
        ranging from 0.00 to 1.00 on the same family -- any previously
        reported single-run transfer number for this detector was not a
        reproducible measurement.
        """
        torch.manual_seed(self.seed)
        self.model = TrajectoryMLPNet(FEATURE_DIM, self.hidden)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        all_features = []
        all_labels = []
        for run in runs_a:
            feat = self._extract_features(run)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(1.0)
        for run in runs_b:
            feat = self._extract_features(run)
            if feat is not None:
                all_features.append(feat)
                all_labels.append(0.0)

        if not all_features:
            return

        all_features = np.array(all_features)
        self.feature_stats = {
            "mean": all_features.mean(axis=0),
            "std": all_features.std(axis=0) + 1e-6,
        }
        all_features = (all_features - self.feature_stats["mean"]) / self.feature_stats["std"]

        X = torch.tensor(all_features, dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                loss = criterion(self.model(X[idx]), y[idx])
                loss.backward()
                optimizer.step()

        self.is_trained = True

    def classify(self, run: RunData) -> float:
        """Score: predicted P(hacking) from MLP."""
        if not self.is_trained or self.model is None:
            return 0.5
        feat = self._extract_features(run)
        if feat is None:
            return 0.5
        return self._predict(feat)

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as the first episode where P(hacking) jumps.

        Re-applies feature extraction on a growing prefix of the run -- valid
        here because both source arrays (``proxy_rewards``, ``behav_trace``)
        are indexed by episode, unlike a raw trajectory.
        """
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
            scores.append((t, self._predict(feat)))

        if len(scores) < 2:
            return -1

        max_increase = 0.0
        onset_t = -1
        for i in range(1, len(scores)):
            increase = scores[i][1] - scores[i - 1][1]
            if increase > max_increase:
                max_increase = increase
                onset_t = scores[i][0]
        return onset_t if max_increase > 0.15 else -1

    def _predict(self, feat: np.ndarray) -> float:
        x = feat.reshape(1, -1)
        if self.feature_stats:
            x = (x - self.feature_stats["mean"]) / self.feature_stats["std"]
        with torch.no_grad():
            score = float(self.model(torch.tensor(x, dtype=torch.float32)).item())
        return max(0.0, min(1.0, score))

    def _extract_features(self, run: RunData) -> Optional[np.ndarray]:
        """Behavioral + proxy feature vector (no true-reward access)."""
        if run.behav_trace is None or len(run.behav_trace) < 2:
            return None
        w = self.window
        late_behav = run.behav_trace[-w:] if len(run.behav_trace) >= w else run.behav_trace
        late_proxy = run.proxy_rewards[-w:] if len(run.proxy_rewards) >= w else run.proxy_rewards
        if len(late_behav) < 2:
            return None

        return np.array(
            [
                float(np.mean(late_behav)),
                float(np.std(late_behav)),
                float(np.max(late_behav) - np.min(late_behav)),
                float(np.mean(late_proxy)),
                float(np.std(late_proxy)),
                float(late_behav[-1] - late_behav[0]),
            ]
        )
