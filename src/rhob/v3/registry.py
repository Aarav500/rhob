"""FamilyRegistry: the central registry of environment family generators.

Distinct from the frozen ``rhob.environments.registry`` (which registers the
Milestone-1 streaming environments); this one registers v3 matched-proxy family
generators keyed by name.
"""

from __future__ import annotations

from rhob.v3.base_family import BaseFamily
from rhob.v3.base_pair import MatchedPair


class FamilyRegistry:
    """Registry of all environment family generators."""

    _families: dict[str, BaseFamily] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator: instantiate and register a :class:`BaseFamily` subclass."""

        def decorator(family_cls: type[BaseFamily]) -> type[BaseFamily]:
            instance = family_cls()
            if name in cls._families:
                raise ValueError(f"family {name!r} already registered")
            cls._families[name] = instance
            return family_cls

        return decorator

    @classmethod
    def list_families(cls) -> list[str]:
        return sorted(cls._families)

    @classmethod
    def get(cls, name: str) -> BaseFamily:
        if name not in cls._families:
            raise KeyError(f"unknown family {name!r}; registered: {cls.list_families()}")
        return cls._families[name]

    @classmethod
    def resolve(cls, families: str | list[str]) -> list[BaseFamily]:
        """Resolve ``"all"`` or a name/list of names to family instances."""
        if families == "all":
            return [cls._families[n] for n in cls.list_families()]
        if isinstance(families, str):
            families = [families]
        return [cls.get(n) for n in families]

    @classmethod
    def generate_suite(
        cls,
        families: str | list[str] = "all",
        difficulties: str | list[float] = "all",
        layout_seed: int = 0,
    ) -> list[MatchedPair]:
        """Generate the evaluation suite: one pair per (family, difficulty).

        ``layout_seed`` selects the environment layout every cell is built at, and
        therefore the behavioral orientation each family is scored under (see
        :mod:`rhob.v3.sign_randomization`); ``generate_pair_at`` stamps it onto the pair
        so the orientation is derived from it rather than assumed.

        The default ``0`` reproduces the historical single-draw suite: one layout per
        (family, difficulty) for the whole benchmark. That is a single sample, not a
        population, and an AUROC measured from it carries the sampling error of one
        draw -- at the published ``n_seeds=5`` the Mann-Whitney null SE is
        ``sqrt((n+m+1)/(12nm))`` = 0.19, which is wider than most differences the
        leaderboard reports. Vary this argument (with ``Benchmark.evaluate``'s
        ``seed_base``) to replicate the suite over independent draws and put an
        interval on those numbers; ``scripts/replicate_leaderboard.py`` does exactly
        that.
        """
        pairs: list[MatchedPair] = []
        for fam in cls.resolve(families):
            if difficulties == "all":
                diffs = fam.default_difficulties()
            elif isinstance(difficulties, (int, float)):
                diffs = [float(difficulties)]
            else:
                diffs = [float(d) for d in difficulties]
            lo, hi = fam.difficulty_range()
            for d in diffs:
                if lo - 1e-9 <= d <= hi + 1e-9:
                    pairs.append(fam.generate_pair_at(d))
        return pairs
