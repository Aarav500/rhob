"""Fundamental type definitions for RHOB.

This module defines the core enumerations and type aliases used throughout the
benchmark. These types are frozen per major version (see the API stability
guarantees in the engineering specification).
"""

from __future__ import annotations

from enum import Enum, IntEnum

# --- Type aliases (semantic documentation over ``str``) ---
EnvironmentID = str  # e.g. "tier1/gridworld_wireheading"
DetectorID = str  # e.g. "baselines/cusum"
RunID = str  # e.g. "tier1/gridworld_wireheading:seed=0"


class AccessLevel(IntEnum):
    """Information constraint imposed on a detector.

    Access levels form a *total order* ``L1 < L2 < L3 < L4``. A detector
    declared at level ``Lk`` may only observe fields available at that level.
    Using :class:`IntEnum` makes the ordering comparisons (``>=``) meaningful,
    which the access filter relies on.

    Levels
    ------
    L1
        Reward-only. The detector sees the scalar proxy reward and timestep.
    L2
        Trajectory. Adds behavioural / policy-level features.
    L3
        Gradient. Adds gradient-norm and KL-from-reference signals.
    L4
        Full. Adds hidden internal environment state.
    """

    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.name


class HackingType(str, Enum):
    """Category of reward hacking exhibited by an environment.

    Values follow the taxonomy synthesised in the gap analysis (Skalse et al.,
    Krakovna et al., Pan et al., Everitt et al., Langosco et al.).
    """

    PROXY_GAMING = "proxy_gaming"
    REWARD_TAMPERING = "reward_tampering"
    SPECIFICATION_GAMING = "specification_gaming"
    GOAL_MISGENERALIZATION = "goal_misgeneralization"
    REWARD_MODEL_OVEROPTIMIZATION = "reward_model_overoptimization"
    EMERGENT_EXPLOITATION = "emergent_exploitation"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


class Tier(str, Enum):
    """Environment difficulty tier.

    The tier determines the weight an environment receives in the aggregate
    RHOB-Score (harder tiers are weighted more heavily).
    """

    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    ADVERSARIAL = "adversarial"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# Tier weights for the RHOB-Score (engineering spec Section 7.2).
TIER_WEIGHTS: dict[Tier, float] = {
    Tier.TIER1: 1.0,
    Tier.TIER2: 1.5,
    Tier.TIER3: 2.0,
    Tier.ADVERSARIAL: 2.5,
}
