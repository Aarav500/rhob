"""Registry of detectors that duplicate another detector instead of measuring something new.

A detector listed here is kept in the suite and kept in the committed leaderboard
artifacts -- it is still a useful cross-check that two code paths agree -- but it must
never be counted a second time in any per-access-level aggregate, because doing so
reports the same measurement twice and inflates the level it is filed under.

The one current entry, ``Perfect Feature Oracle``, was added after the audit finding
that RHOB's headline "oracle ceiling" was a relabelled copy of its own L2 baseline.
The duplication is both structural and measured:

* ``PerfectFeatureOracleDetector`` (``l3_perfect_feature.py``) subclasses
  ``BehavioralThresholdDetector`` (``l2_behavioral_threshold.py``) and overrides only
  ``access_level`` and ``name``. It inherits ``classify`` and ``detect_onset``
  verbatim, so it consumes exactly one signal, ``RunData.behav_trace`` -- which
  ``rhob.v3.access.restrict`` already exposes at L2. It never reads
  ``true_rewards``, the only channel L3 actually adds.
* In ``leaderboard/v5_leaderboard.json`` the two detectors agree on 33 of 33
  families to the last recorded digit, and on the overall figure (both 0.9750
  over 123 cells). Zero families differ.

Consequence for the aggregate: with the duplicate counted, L3 reads
n=2, mean 0.9790, max 0.9830. With it excluded, L3 is the single genuine oracle
(True Reward Oracle, 0.9830). The duplicate is *not* re-filed under L2 either --
it would then double-count Behavioral Threshold inside L2 and lift that level's
mean from 0.7213 to 0.7530 while adding no new information. A duplicate belongs
to no level's aggregate.
"""

from __future__ import annotations

# Maps a duplicate detector's reported ``name`` to the ``name`` of the detector it
# duplicates. Keyed by name (not class) so that leaderboard JSON, which stores only
# names, can be filtered without importing and instantiating the detector suite.
DUPLICATE_DIAGNOSTICS: dict[str, str] = {
    "Perfect Feature Oracle": "Behavioral Threshold",
}


def duplicate_source(detector_name: str) -> str | None:
    """Return the name of the detector ``detector_name`` duplicates, or ``None``.

    ``None`` means the detector contributes an independent measurement and belongs
    in its access level's aggregate.
    """
    return DUPLICATE_DIAGNOSTICS.get(detector_name)


def is_duplicate_diagnostic(detector_name: str) -> bool:
    """Whether ``detector_name`` is a known duplicate and must be left out of aggregates."""
    return detector_name in DUPLICATE_DIAGNOSTICS
