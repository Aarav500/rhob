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
from rhob.v3.sign_randomization import apply_behav_sign, behav_sign
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
        rollout_hacking: ``seed -> (RunData, onset)`` for variant A (hacking), in the
            family's *own* authored coordinate (positive = hacking, see CONTRIBUTING).
        rollout_legit: ``seed -> (RunData, onset)`` for variant B (legitimate), same.
        params: Family-specific parameters (e.g. ``{"d": 0.75}``) for provenance.
        layout_seed: The ``seed`` this pair's layout was generated at. Stamped by
            :meth:`~rhob.v3.base_family.BaseFamily.generate_pair_at`; families need not
            set it. Half of the key the evaluation-time behavioral sign is drawn from
            (see :mod:`rhob.v3.sign_randomization`).
    """

    family: str
    mechanism: HackingMechanism
    complexity: EnvironmentComplexity
    difficulty: float
    n_episodes: int
    rollout_hacking: Callable[[int], RolloutResult]
    rollout_legit: Callable[[int], RolloutResult]
    params: dict
    layout_seed: int = 0

    def rollout(
        self, n_seeds: int, seed_base: int = 0, randomize_sign: bool = True
    ) -> tuple[list[RunData], list[RunData], list[int]]:
        """Roll out ``n_seeds`` runs of each variant, ready for evaluation.

        This is where the benchmark's behavioral coordinate is fixed, and therefore
        where the sign randomization is applied: it is the one call every evaluation
        path goes through (``Benchmark.evaluate``'s rollout cache,
        ``scripts/cross_family_transfer.py``, the admission gate) and it sees *both*
        variants of the pair at once, which a per-run hook could not. The detector is
        handed the result and never the sign -- see :mod:`rhob.v3.sign_randomization`
        for why the flip has to happen here rather than in a detector-visible flag.

        The two ``rollout_*`` closures are deliberately left alone. They stay in the
        family's authored coordinate, so a family's own tests can still assert the one
        property that survives randomization: that the pair is internally anti-symmetric
        *before* the benchmark chooses an orientation for it.

        Args:
            n_seeds: Runs per variant.
            seed_base: First rollout seed; the legitimate variant is offset by 1000.
            randomize_sign: Apply the per-``(family, layout_seed)`` coordinate flip to
                both variants. Default ``True``. Pass ``False`` only to measure the
                un-randomized baseline -- the before/after comparison, and the admission
                gate, which certifies the family's own behavioral separation rather than
                any detector's ability to orient itself.

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
        if randomize_sign:
            sign = behav_sign(self.family, self.layout_seed)
            runs_a = [apply_behav_sign(r, sign) for r in runs_a]
            runs_b = [apply_behav_sign(r, sign) for r in runs_b]
        return runs_a, runs_b, onsets_a
