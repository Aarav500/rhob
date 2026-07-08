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
    ) -> list[MatchedPair]:
        """Generate the evaluation suite: one pair per (family, difficulty)."""
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
                    pairs.append(fam.generate_pair(d))
        return pairs
