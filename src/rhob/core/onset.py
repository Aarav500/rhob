"""Onset label model.

An :class:`OnsetLabel` is the ground-truth annotation produced by an
environment's oracle. It records *when* reward hacking begins (the onset
timestep ``t*``), how confident the oracle is, and how severe the divergence is.

Onset labels are used for *evaluation only* -- they are never exposed to
detectors (see :mod:`rhob.core.access`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rhob.core.types import HackingType


@dataclass
class OnsetLabel:
    """Ground-truth reward-hacking onset annotation.

    Attributes
    ----------
    onset_step
        The labelled onset timestep ``t*`` (training-step index at which the
        true return begins to degrade while the proxy return keeps rising).
    confidence
        Oracle confidence in ``[0, 1]``. Derived from the strength of the
        two-sample test that identified the onset.
    hacking_type
        The category of hacking exhibited on this run.
    detection_method
        How the oracle determined the onset (e.g. ``"two_sample"``).
    confidence_interval
        Uncertainty in ``t*`` as an inclusive ``(low, high)`` step range.
    severity
        Rate of true-reward degradation after onset (larger = more severe).
    """

    onset_step: int
    confidence: float
    hacking_type: HackingType
    detection_method: str = "two_sample"
    confidence_interval: tuple[int, int] = (0, 0)
    severity: float = 0.0

    def __post_init__(self) -> None:
        if self.onset_step < 0:
            raise ValueError(f"onset_step must be non-negative, got {self.onset_step}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        # Normalise a plain string hacking_type into the enum.
        if not isinstance(self.hacking_type, HackingType):
            self.hacking_type = HackingType(self.hacking_type)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary (for HDF5 attributes / JSON)."""
        return {
            "onset_step": int(self.onset_step),
            "confidence": float(self.confidence),
            "hacking_type": self.hacking_type.value,
            "detection_method": self.detection_method,
            "confidence_interval": [
                int(self.confidence_interval[0]),
                int(self.confidence_interval[1]),
            ],
            "severity": float(self.severity),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OnsetLabel":
        """Reconstruct an :class:`OnsetLabel` from :meth:`to_dict` output."""
        ci = data.get("confidence_interval", (0, 0))
        return cls(
            onset_step=int(data["onset_step"]),
            confidence=float(data["confidence"]),
            hacking_type=HackingType(data["hacking_type"]),
            detection_method=str(data.get("detection_method", "two_sample")),
            confidence_interval=(int(ci[0]), int(ci[1])),
            severity=float(data.get("severity", 0.0)),
        )
