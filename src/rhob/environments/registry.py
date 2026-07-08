"""Global environment registry.

A minimal registry mapping environment IDs to their classes, enabling discovery
(``list_environments``), instantiation (``get_environment``), and tier lookup
(used by the RHOB-Score aggregation).
"""

from __future__ import annotations

from typing import Type

from rhob.environments.base import AbstractEnvironment
from rhob.environments.tier1.gridworld_wireheading import GridWorldWireheading
from rhob.core.types import Tier

_REGISTRY: dict[str, Type[AbstractEnvironment]] = {}


def register_environment(env_class: Type[AbstractEnvironment]) -> Type[AbstractEnvironment]:
    """Register an environment class by its ``id`` (usable as a decorator)."""
    if not env_class.id:
        raise ValueError(f"{env_class.__name__} has no 'id' attribute set")
    _REGISTRY[env_class.id] = env_class
    return env_class


def get_environment(env_id: str, **kwargs) -> AbstractEnvironment:
    """Instantiate a registered environment by ID."""
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment {env_id!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[env_id](**kwargs)


def list_environments() -> list[str]:
    """Return the sorted list of registered environment IDs."""
    return sorted(_REGISTRY)


def get_environment_tier(env_id: str) -> Tier:
    """Return the difficulty tier of a registered environment."""
    if env_id not in _REGISTRY:
        raise KeyError(f"Unknown environment {env_id!r}")
    return _REGISTRY[env_id].tier


def get_environment_card(env_id: str):
    """Return the :class:`EnvironmentCard` for a registered environment."""
    return get_environment(env_id).describe()


# --- Built-in registrations ---
register_environment(GridWorldWireheading)
