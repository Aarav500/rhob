"""Perfect-feature detector: a relabelled duplicate of the L2 behavioral baseline.

**This is not an oracle and it is not an L3 measurement.** It is
:class:`~rhob.detectors.l2_behavioral_threshold.BehavioralThresholdDetector` under a
different name, and it is retained only as a cross-check, never as evidence about what
L3 access buys. It is excluded from every per-access-level aggregate; see
:mod:`rhob.detectors.redundancy` and
:mod:`rhob.v3.leaderboard.access_summary`.

The class was originally introduced on the theory that reading "the exact feature every
family is designed around" was itself a form of oracle knowledge deserving its own
ceiling row. That reasoning does not survive contact with the access filter. This class
overrides only ``access_level`` and ``name``; ``classify`` and ``detect_onset`` are
inherited unchanged, so the single signal it reads is ``RunData.behav_trace`` -- which
:func:`rhob.v3.access.restrict` already hands to any L2 detector. It never touches
``true_rewards``, the one channel L3 adds. Nothing about its behaviour is L3.

Measured consequence, from the committed ``leaderboard/v5_leaderboard.json``: this
detector and Behavioral Threshold agree on 33 of 33 families (zero families differ) and
on the overall figure, both 0.9750 across 123 cells. Counting it made the L3 aggregate
look like n=2, mean 0.9790 -- an "oracle ceiling" one of whose two members was an L2
result wearing a hat. With it excluded, L3 is the one genuine oracle
(:class:`~rhob.detectors.l3_true_reward_oracle.TrueRewardOracleDetector`, 0.9830), which
sits 0.0080 above the best L2 detector (0.9750): there is no meaningful oracle gap.

``name`` and ``access_level`` are deliberately left as they are. The committed
leaderboard artifacts, the paper tables and the public Space all key off
``"Perfect Feature Oracle"`` at ``"L3"``; silently renaming or re-levelling the class
would desynchronize the code from data on disk without fixing anything, since the
double-count is removed at the aggregation step and re-filing the duplicate under L2
would merely move the double-count there (L2 mean 0.7213 -> 0.7530 for no new
information). The label is inert to measurement in any case: L2 and L3 both expose
``behav_trace``, so this detector scores identically either way.
"""

from __future__ import annotations

from rhob.detectors.l2_behavioral_threshold import BehavioralThresholdDetector


class PerfectFeatureOracleDetector(BehavioralThresholdDetector):
    """Duplicate of the L2 Behavioral Threshold baseline, kept as a labelled cross-check."""

    #: Name of the detector this one duplicates. Consumers that aggregate by access
    #: level must skip any detector with this attribute set; the authoritative list is
    #: :data:`rhob.detectors.redundancy.DUPLICATE_DIAGNOSTICS`.
    duplicate_of = "Behavioral Threshold"

    @property
    def access_level(self) -> str:
        """Reported as ``"L3"`` for artifact continuity only -- see the module docstring.

        The signal actually consumed (``behav_trace``) is available at L2. This value
        does not license reading the detector as an L3 result.
        """
        return "L3"

    @property
    def name(self) -> str:
        return "Perfect Feature Oracle"
