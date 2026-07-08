"""Abstract detector contract.

A detector consumes a stream of access-filtered
:class:`~rhob.core.trajectory.Observation` objects (one per training step) and
emits, at each step, a hacking probability in ``[0, 1]``.

Contract invariants (verified by the evaluation runner and the contract tests):

* **Determinism** -- the same sequence of observations must yield identical
  scores, bit-for-bit.
* **Bounded output** -- every score lies in ``[0.0, 1.0]`` (no NaN/inf).
* **Reset isolation** -- scores after :meth:`reset` must not depend on any
  previously seen trajectory.
* **Access compliance** -- a detector declared at ``Lk`` must not read fields
  above level ``k`` (structurally enforced by the access filter).
* **Streaming cost** -- :meth:`step` should run in roughly O(1) amortized time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rhob.core.trajectory import Observation
from rhob.core.types import AccessLevel


@dataclass
class OverheadEstimate:
    """Rough per-step computational overhead of a detector."""

    relative_overhead: float  # fraction of base training cost (e.g. 0.03 = 3%)
    description: str = ""


class AbstractDetector(ABC):
    """Base class every RHOB detector must subclass."""

    # --- Identity (set by subclass) ---
    name: str = ""
    id: str = ""
    version: str = "0.1.0"
    access_level: AccessLevel = AccessLevel.L1
    is_oracle_free: bool = True

    # --- Lifecycle ---
    @abstractmethod
    def reset(self) -> None:
        """Clear all internal state. Called before each new trajectory."""
        raise NotImplementedError

    def configure(self, **kwargs: Any) -> None:
        """Apply configuration. Called once before evaluation (optional)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # --- Core interface ---
    @abstractmethod
    def step(self, observation: Observation) -> float:
        """Consume one observation; return a hacking probability in ``[0, 1]``."""
        raise NotImplementedError

    # --- Metadata ---
    def hyperparameters(self) -> dict[str, Any]:
        """Return current hyperparameter values (for reproducibility)."""
        return {}

    def computational_overhead(self) -> OverheadEstimate:
        """Report estimated computational overhead vs. base training."""
        return OverheadEstimate(relative_overhead=0.0, description="unspecified")

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"{type(self).__name__}(id={self.id!r}, access={self.access_level})"
