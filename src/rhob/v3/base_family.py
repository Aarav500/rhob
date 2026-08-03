"""BaseFamily: the abstract generator every environment family implements.

A family produces matched-proxy pairs at any requested difficulty. The difficulty
argument is the *target* behavioral separability (L2-AUROC) in ``(0.5, 1.0)``;
families with a continuous knob honor it exactly, while inherently single-point
families (e.g. the tabular gridworld, whose L2 is binary) expose a one-element range.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from rhob.v3.base_pair import MatchedPair
from rhob.v3.taxonomy import DifficultyTier, EnvironmentComplexity, HackingMechanism


class BaseFamily(ABC):
    """Abstract base for environment family generators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique registered family name."""

    @property
    @abstractmethod
    def mechanism(self) -> HackingMechanism:
        """The hacking mechanism this family instantiates."""

    @property
    @abstractmethod
    def complexity(self) -> EnvironmentComplexity:
        """The environment complexity class."""

    @abstractmethod
    def difficulty_range(self) -> tuple[float, float]:
        """Inclusive ``(min, max)`` target-L2 this family can generate.

        A single-point family returns ``(x, x)``.
        """

    @abstractmethod
    def generate_pair(self, difficulty: float, seed: int = 0) -> MatchedPair:
        """Generate one pair targeting the given behavioral separability (L2-AUROC)."""

    def generate_pair_at(self, difficulty: float, seed: int = 0) -> MatchedPair:
        """:meth:`generate_pair`, with the layout seed recorded on the returned pair.

        Families build their :class:`~rhob.v3.base_pair.MatchedPair` themselves and are
        not required to know that the benchmark keys anything off the layout seed, so
        stamping it is done here, once, instead of in 33 constructors.

        Every path that varies the layout (the admission gate's ``n_layouts`` sweep,
        anything sweeping ``seed=``) should call this rather than :meth:`generate_pair`,
        so that two layouts of one family get independent behavioral orientations. A
        pair built by :meth:`generate_pair` directly keeps ``layout_seed=0`` and is
        still randomized -- just at the family's layout-0 orientation, which is what the
        benchmark suite itself uses (:meth:`FamilyRegistry.generate_suite` generates
        every scored cell at the default ``seed=0``).
        """
        return replace(self.generate_pair(difficulty, seed=seed), layout_seed=int(seed))

    def supported_tiers(self) -> list[DifficultyTier]:
        """The named tiers that fall within this family's difficulty range."""
        lo, hi = self.difficulty_range()
        return [t for t in DifficultyTier.all() if lo - 1e-9 <= float(t) <= hi + 1e-9]

    def default_difficulties(self) -> list[float]:
        """Difficulties used when a caller asks for ``"all"``.

        Named tiers inside the range, or---for a single-point / tier-free range
        (e.g. the gridworld anchor at L2=1.0)---the range endpoints.
        """
        tiers = [float(t) for t in self.supported_tiers()]
        if tiers:
            return tiers
        lo, hi = self.difficulty_range()
        return sorted({round(lo, 4), round(hi, 4)})

    def generate_difficulty_sweep(self, difficulties: list[float] | None = None) -> list[MatchedPair]:
        """Generate one pair per requested difficulty (defaults to the family's tiers)."""
        if difficulties is None:
            difficulties = self.default_difficulties()
        return [self.generate_pair(d) for d in difficulties]
