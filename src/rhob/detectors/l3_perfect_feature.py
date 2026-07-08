"""Perfect-feature oracle detector (L3, ceiling measurement).

Identical to :class:`~rhob.detectors.l2_behavioral_threshold.BehavioralThresholdDetector`
except it is explicitly labeled L3 (ceiling), not L2, to keep the "is the information
there?" ceiling measurement separate from L2 results attributed to a genuine access
level. Reads the exact anti-symmetric feature every family is designed around.
"""

from __future__ import annotations

from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector


class PerfectFeatureOracleDetector(BehavioralThresholdDetector):
    """The near-oracle behavioral-threshold detector, reported at its true level: L3."""

    @property
    def access_level(self) -> str:
        return "L3"

    @property
    def name(self) -> str:
        return "Perfect Feature Oracle"
