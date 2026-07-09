"""Neural network detector on reward history (L0, strongest L0 baseline)."""

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

if HAS_TORCH:
    class RewardMLPNet(nn.Module):
        """Small MLP for reward history classification."""

        def __init__(self, window_size: int = 50, hidden: int = 64):
            super().__init__()
            self.window_size = window_size
            self.fc1 = nn.Linear(window_size, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.fc3 = nn.Linear(hidden, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x


class RewardMLPDetector(PosthocDetector):
    """Neural network trained on reward history windows.

    This is the STRONGEST possible L0 detector. Failure here confirms
    the information barrier: reward alone is insufficient.
    """

    def __init__(self, window_size: int = 50, hidden: int = 64, seed: int = 0):
        if not HAS_TORCH:
            raise ImportError("torch required for RewardMLPDetector")
        self.window_size = window_size
        self.hidden = hidden
        self.seed = seed
        self.model: Optional[RewardMLPNet] = None
        self.is_trained = False

    @property
    def access_level(self) -> str:
        return "L0"

    @property
    def name(self) -> str:
        return "Reward MLP"

    def fit(
        self,
        runs_a: list[RunData],
        runs_b: list[RunData],
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.001,
    ) -> None:
        """Train on labeled runs (consistent ``fit(runs_a, runs_b)`` signature shared
        by every trainable detector -- the caller passes ``RunData``, not raw arrays)."""
        if not HAS_TORCH:
            raise ImportError("torch required")
        torch.manual_seed(self.seed)
        self.model = RewardMLPNet(self.window_size, self.hidden)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        all_rewards = []
        all_labels = []
        for run in runs_a:
            r = run.proxy_rewards
            if len(r) >= self.window_size:
                all_rewards.append(r[-self.window_size :])
                all_labels.append(1.0)
        for run in runs_b:
            r = run.proxy_rewards
            if len(r) >= self.window_size:
                all_rewards.append(r[-self.window_size :])
                all_labels.append(0.0)

        if not all_rewards:
            return

        X = torch.tensor(np.array(all_rewards), dtype=torch.float32)
        y = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)

        for _ in range(epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i : i + batch_size]
                x_batch = X[idx]
                y_batch = y[idx]
                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        self.is_trained = True

    def classify(self, run: RunData) -> float:
        """Score: predicted P(hacking) from MLP."""
        if not self.is_trained or self.model is None:
            return 0.5
        if len(run.proxy_rewards) < self.window_size:
            return 0.5
        window = run.proxy_rewards[-self.window_size :]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = float(self.model(x).item())
        return max(0.0, min(1.0, score))

    def detect_onset(self, run: RunData) -> int:
        """Detect onset as first episode where predicted P(hacking) jumps."""
        if not self.is_trained or self.model is None:
            return -1
        if len(run.proxy_rewards) < self.window_size:
            return -1

        scores = []
        for t in range(self.window_size, len(run.proxy_rewards)):
            window = run.proxy_rewards[max(0, t - self.window_size + 1) : t + 1]
            if len(window) < self.window_size:
                continue
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score = float(self.model(x).item())
            scores.append((t, score))

        if not scores:
            return -1

        # Detect jump: find largest increase
        max_increase = 0.0
        onset_t = -1
        for i in range(1, len(scores)):
            increase = scores[i][1] - scores[i - 1][1]
            if increase > max_increase:
                max_increase = increase
                onset_t = scores[i][0]

        return onset_t if max_increase > 0.1 else -1
