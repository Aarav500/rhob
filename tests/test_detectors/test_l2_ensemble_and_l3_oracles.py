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
    assert detector.classify(hacking) < detector.classify(legit)
    onset = detector.detect_onset(hacking)
    assert 15 < onset < 25


def test_perfect_feature_oracle_is_l3_not_l2():
    detector = PerfectFeatureOracleDetector()
    assert detector.access_level == "L3"
    assert detector.name == "Perfect Feature Oracle"
    run = _run(behav_sign=0.8, true_low=True)
    assert detector.classify(run) > 0
