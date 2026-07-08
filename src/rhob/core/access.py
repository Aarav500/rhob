"""Access-level enforcement.

The :class:`AccessFilter` converts a full training step into an
:class:`~rhob.core.trajectory.Observation` that exposes *only* the fields
permitted at the detector's declared access level. Fields above the level are
set to ``None``.

Key invariant
-------------
A detector declared at ``L1`` will **never** receive behavioural (L2) features,
regardless of what the environment recorded. Enforcement is *structural* (the
filter constructs a fresh, immutable observation) rather than by convention, so
it is impossible to accidentally leak higher-level information.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rhob.core.trajectory import Observation
from rhob.core.types import AccessLevel


class AccessFilter:
    """Restricts a training step to a given access level."""

    def __init__(self, access_level: AccessLevel):
        self.access_level = AccessLevel(access_level)

    def filter(
        self,
        t: int,
        reward_proxy: float,
        policy_features: Optional[np.ndarray] = None,
    ) -> Observation:
        """Build an access-filtered :class:`Observation`.

        Parameters
        ----------
        t
            Training-step index.
        reward_proxy
            The proxy reward at this step (always available, L1).
        policy_features
            The L2 behavioural summary, or ``None``. Included only if the
            declared access level is ``L2`` or higher.
        """
        features = policy_features if self.access_level >= AccessLevel.L2 else None
        # Defensive copy so a detector cannot mutate shared arrays.
        if features is not None:
            features = np.array(features, dtype=np.float64, copy=True)
            features.setflags(write=False)
        return Observation(
            t=int(t),
            access_level=self.access_level,
            reward=float(reward_proxy),
            policy_features=features,
        )
