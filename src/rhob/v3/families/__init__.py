"""Registered environment families. Importing this package registers them all."""

from rhob.v3.families import (  # noqa: F401
    continuous_camping,
    distributional_shift,
    goal_misgeneralization,
    gridworld_camping,
    novelty_farming,
    orbit_chirality,
    physics_exploitation,
    proxy_correlation_gaming,
    shortcut_exploitation,
)

__all__ = [
    "gridworld_camping",
    "continuous_camping",
    "proxy_correlation_gaming",
    "shortcut_exploitation",
    "novelty_farming",
    "orbit_chirality",
    "goal_misgeneralization",
    "physics_exploitation",
    "distributional_shift",
]
