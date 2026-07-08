"""Random baseline detector.

Emits a uniform random score in ``[0, 1]`` at each step. This is the *floor*
baseline: its expected AUROC is 0.5. It is seeded from a fixed seed on
:meth:`reset`, so it satisfies the determinism contract (the same trajectory
always yields the same scores) while remaining uninformative about the onset.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rhob.core.trajectory import Observation
from rhob.core.types import AccessLevel
from rhob.detectors.base import AbstractDetector, OverheadEstimate


class RandomDetector(AbstractDetector):
    """Uniform-random detector; provides the metric floor (AUROC ~ 0.5)."""

    name = "Random"
    id = "baselines/random"
    version = "0.1.0"
    access_level = AccessLevel.L1
    is_oracle_free = True

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        # Re-seed deterministically so scores are reproducible per trajectory.
        self._rng = np.random.default_rng(self.seed)

    def step(self, observation: Observation) -> float:
        return float(self._rng.random())

    def hyperparameters(self) -> dict[str, Any]:
        return {"seed": self.seed}

    def computational_overhead(self) -> OverheadEstimate:
        return OverheadEstimate(relative_overhead=0.0, description="one RNG draw per step")
