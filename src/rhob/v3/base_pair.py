"""MatchedPair: one admitted (or candidate) matched-proxy environment pair.

A pair is represented by two rollout closures---one per variant---plus metadata.
Keeping the environment internals behind closures lets every family (gridworld,
continuous, future families) expose the same interface to the benchmark without a
shared environment base class. Each closure maps a seed to a completed run's
:class:`~rhob.detectors.posthoc.RunData` and its oracle onset label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rhob.detectors.posthoc import RunData
from rhob.v3.taxonomy import EnvironmentComplexity, HackingMechanism

# (RunData, onset_episode) for one rolled-out run.
RolloutResult = tuple[RunData, int]


@dataclass(frozen=True)
class MatchedPair:
    """A matched-proxy pair: hacking variant A vs. legitimate foil B.

    Attributes:
        family: Registered family name that produced this pair.
        mechanism: The hacking mechanism instantiated.
        complexity: The environment complexity class.
        difficulty: Target behavioral separability (L2-AUROC) this pair was built for.
        n_episodes: Episodes per run.
        rollout_hacking: ``seed -> (RunData, onset)`` for variant A (hacking).
        rollout_legit: ``seed -> (RunData, onset)`` for variant B (legitimate).
        params: Family-specific parameters (e.g. ``{"d": 0.75}``) for provenance.
    """

    family: str
    mechanism: HackingMechanism
    complexity: EnvironmentComplexity
    difficulty: float
    n_episodes: int
    rollout_hacking: Callable[[int], RolloutResult]
    rollout_legit: Callable[[int], RolloutResult]
    params: dict

    def rollout(
        self, n_seeds: int, seed_base: int = 0
    ) -> tuple[list[RunData], list[RunData], list[int]]:
        """Roll out ``n_seeds`` runs of each variant.

        Returns:
            ``(runs_a, runs_b, onsets_a)`` -- hacking runs, legitimate runs, and the
            oracle onset labels for the hacking runs (used only for onset scoring).
        """
        runs_a, runs_b, onsets_a = [], [], []
        for s in range(n_seeds):
            ra, oa = self.rollout_hacking(seed_base + s)
            rb, _ = self.rollout_legit(seed_base + 1000 + s)
            runs_a.append(ra)
            runs_b.append(rb)
            onsets_a.append(oa)
        return runs_a, runs_b, onsets_a
