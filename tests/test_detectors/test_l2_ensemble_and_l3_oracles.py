"""Tests for the L2 ensemble detector and the L3 oracle ceiling detectors."""

from __future__ import annotations

import numpy as np
import pytest

from rhob.detectors import (
    BehavioralThresholdDetector,
    EnsembleDetector,
    PerfectFeatureOracleDetector,
    RunData,
    TrueRewardOracleDetector,
    duplicate_source,
    is_duplicate_diagnostic,
)


def _run(behav_sign: float, true_low: bool) -> RunData:
    behav = np.concatenate([np.zeros(20), np.full(60, behav_sign)])
    true = np.concatenate([np.full(20, 0.5), np.full(60, 0.1 if true_low else 0.9)])
    return RunData(np.ones(80) * 0.5, true, None, behav)


def test_ensemble_requires_shared_access_level():
    a = BehavioralThresholdDetector()
    b = TrueRewardOracleDetector()  # L3, mismatched with a's L2
    with pytest.raises(ValueError):
        EnsembleDetector([a, b])


def test_ensemble_averages_member_scores():
    members = [BehavioralThresholdDetector(steady_window=60), BehavioralThresholdDetector(steady_window=30)]
    ens = EnsembleDetector(members, name="test-ensemble")
    run = _run(behav_sign=0.8, true_low=True)
    assert ens.access_level == "L2"
    assert ens.name == "test-ensemble"
    score = ens.classify(run)
    assert score > 0


def test_ensemble_onset_by_majority_agreement():
    members = [BehavioralThresholdDetector(baseline_episodes=20) for _ in range(3)]
    ens = EnsembleDetector(members)
    run = _run(behav_sign=0.8, true_low=True)
    onset = ens.detect_onset(run)
    assert 15 < onset < 25


def test_true_reward_oracle_classifies_and_localizes():
    detector = TrueRewardOracleDetector(baseline_episodes=20)
    assert detector.access_level == "L3"
    hacking = _run(behav_sign=0.0, true_low=True)
    legit = _run(behav_sign=0.0, true_low=False)
    # classify() must return a hacking-*likelihood* (PosthocDetector contract:
    # "higher = more likely hacking"), so the low-true-reward (hacking) run
    # must score HIGHER, not lower, than the high-true-reward (legit) run.
    assert detector.classify(hacking) > detector.classify(legit)
    onset = detector.detect_onset(hacking)
    assert 15 < onset < 25


def test_perfect_feature_oracle_keeps_its_published_labels():
    """Name and reported level are frozen: the committed leaderboard artifacts key off them.

    They are retained for artifact continuity only. The detector is a duplicate of the
    L2 baseline and is kept out of the L3 aggregate at the aggregation step instead --
    see ``test_perfect_feature_oracle_is_a_relabelled_l2_duplicate`` below and
    ``rhob.v3.leaderboard.access_summary``.
    """
    detector = PerfectFeatureOracleDetector()
    assert detector.access_level == "L3"
    assert detector.name == "Perfect Feature Oracle"
    run = _run(behav_sign=0.8, true_low=True)
    assert detector.classify(run) > 0


def test_perfect_feature_oracle_is_a_relabelled_l2_duplicate():
    """Lock in the audited fact that justifies excluding it from the L3 aggregate.

    Structural: it overrides only ``access_level`` and ``name``, inheriting the L2
    baseline's ``classify``/``detect_onset`` verbatim, so it reads ``behav_trace`` and
    nothing else -- never ``true_rewards``, the one channel L3 adds over L2.
    Behavioural: it therefore scores identically to the baseline on every run. In
    ``leaderboard/v5_leaderboard.json`` the two agree on 33/33 families and on the
    overall figure (both 0.9750 over 123 cells). If this test ever fails, the detector
    has become an independent measurement and must be re-examined before it is either
    re-levelled or admitted back into an access-level aggregate.
    """
    oracle = PerfectFeatureOracleDetector()
    baseline = BehavioralThresholdDetector()

    assert issubclass(PerfectFeatureOracleDetector, BehavioralThresholdDetector)
    for method in ("classify", "detect_onset"):
        assert method not in vars(PerfectFeatureOracleDetector), (
            f"{method} is now overridden; the duplication claim no longer holds"
        )

    for behav_sign, true_low in ((0.8, True), (-0.8, False), (0.0, True), (0.35, False)):
        run = _run(behav_sign=behav_sign, true_low=true_low)
        assert oracle.classify(run) == baseline.classify(run)
        assert oracle.detect_onset(run) == baseline.detect_onset(run)

    # And it is registered as such, so every aggregation path can see it.
    assert is_duplicate_diagnostic("Perfect Feature Oracle")
    assert duplicate_source("Perfect Feature Oracle") == "Behavioral Threshold"


def test_true_reward_oracle_is_not_a_duplicate():
    """The one genuine L3 detector must not be caught by the duplicate registry."""
    assert not is_duplicate_diagnostic(TrueRewardOracleDetector().name)
    assert duplicate_source("Behavioral Threshold") is None
