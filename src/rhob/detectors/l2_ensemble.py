"""Ensemble detector (L2): combines multiple detectors' scores by rank-averaging.

Members must share an access level (the ensemble's own ``access_level`` is that
shared level, so it is subject to the same restriction as any single member). Onset
is the earliest onset any member detects with majority-plus support, which is a
simple, dependency-free way to combine asynchronous per-member onset estimates.
"""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector, RunData


class EnsembleDetector(PosthocDetector):
    """Rank-averages a list of same-level detectors; flags onset by member agreement."""

    def __init__(self, members: list[PosthocDetector], name: str = "Ensemble"):
        if not members:
            raise ValueError("EnsembleDetector requires at least one member")
        levels = {m.access_level for m in members}
        if len(levels) > 1:
            raise ValueError(f"all ensemble members must share an access level, got {levels}")
        self.members = members
        self._access_level = members[0].access_level
        self._name = name

    @property
    def access_level(self) -> str:
        return self._access_level

    @property
    def name(self) -> str:
        return self._name

    def classify(self, run: RunData) -> float:
        """Score: mean of member scores (members are pre-fit if they need fitting)."""
        return float(sum(m.classify(run) for m in self.members) / len(self.members))

    def detect_onset(self, run: RunData) -> int:
        """Onset: the earliest episode at which more than half the members agree (+-3)."""
        preds = [m.detect_onset(run) for m in self.members]
        detected = sorted(p for p in preds if p >= 0)
        if not detected:
            return -1
        needed = len(self.members) // 2 + 1
        for candidate in detected:
            support = sum(1 for p in preds if p >= 0 and abs(p - candidate) <= 3)
            if support >= needed:
                return candidate
        return detected[0]
