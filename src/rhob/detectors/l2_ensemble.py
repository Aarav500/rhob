"""Ensemble detector (L2): combines multiple detectors' scores by their plain mean.

The mean is of the members' raw scores, not of their ranks, so a member whose scores
spread wider (a probability near 0 or 1 rather than near 0.5) weighs more in the
ensemble's ranking.

Members must share an access level (the ensemble's own ``access_level`` is that
shared level, so it is subject to the same restriction as any single member). Onset
is the earliest onset any member detects with majority-plus support, which is a
simple, dependency-free way to combine asynchronous per-member onset estimates.

The ensemble has no ``fit`` of its own, so ``rhob.v3.benchmark._evaluate_cell`` scores it
directly, without cross-validation. A member that needs fitting must be fitted before it
is passed in, and if it was fitted on the runs the ensemble is then scored on, that score
is in-sample for that member. A cross-validated figure needs a wrapper whose ``fit``
refits the members on each training fold.
"""

from __future__ import annotations

from rhob.detectors.posthoc import PosthocDetector, RunData


class EnsembleDetector(PosthocDetector):
    """Averages the scores of a list of same-level detectors; flags onset by member agreement."""

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

    @property
    def min_episodes(self) -> int:
        """The largest of the members' minimums: the mean is a score only when every term is.

        A member below its own minimum contributes its fallback constant (0.5 for most,
        NaN for Reward Skewness) instead of a score. Averaged in, the constant dilutes
        the ensemble toward chance, and a NaN fails the cell outright, so the ensemble
        needs every member's horizon.
        """
        return max(int(getattr(m, "min_episodes", 0) or 0) for m in self.members)

    def observe_cell(self, runs: list[RunData]) -> None:
        """Forward the cell's unlabeled runs to every member that wants them.

        Without this, a member that orients itself from the population (see
        :meth:`~rhob.detectors.posthoc.PosthocDetector.observe_cell`) would silently
        keep its default orientation inside an ensemble and contribute a
        sign-scrambled score to the average -- worse than the same detector alone.
        """
        for member in self.members:
            member.observe_cell(runs)

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
