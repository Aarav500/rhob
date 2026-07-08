"""Post-hoc detectors for offline evaluation on completed runs.

These detectors consume a full run's recorded per-episode signals and produce:
- A discrimination score (hacking probability) for binary classification
- An onset episode prediction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RunData:
    """A completed run's per-episode signals, access-level-appropriate.

    ``behav_trace`` is the pair's anti-symmetric L2 feature: ``mean(row) -
    mean(col)`` for gridworld, ``mean_x - centre`` for continuous. Both are
    centred at 0 pre-onset and take opposite signs across variants post-onset
    -- the same convention the admission gate itself uses to certify L2.

    ``true_rewards`` is oracle-only: the ground truth used to build onset
    labels and score the benchmark, never a legitimate detector input. No
    detector's ``classify``/``detect_onset`` may read it -- a detector that did
    would trivially "solve" hacking detection by reading the answer.
    """

    proxy_rewards: np.ndarray  # [n_episodes] proxy reward per episode (L0+)
    true_rewards: np.ndarray  # [n_episodes] oracle only -- not a detector input
    state_counts: Optional[np.ndarray]  # [n_episodes, n_bins] state-visitation histogram (L1+)
    behav_trace: Optional[np.ndarray]  # [n_episodes] anti-symmetric behavioral feature (L2)


class PosthocDetector(ABC):
    """Base class for post-hoc detectors."""

    @property
    @abstractmethod
    def access_level(self) -> str:
        """'L0', 'L1', or 'L2'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        raise NotImplementedError

    @abstractmethod
    def classify(self, run: RunData) -> float:
        """Score a run for hacking likelihood. Higher = more likely hacking."""
        raise NotImplementedError

    @abstractmethod
    def detect_onset(self, run: RunData) -> int:
        """Predict onset episode. Return -1 if no onset detected."""
        raise NotImplementedError
